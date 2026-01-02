import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
import cv2

from opt import get_opts
from dataset import VideoDataset
from models.implicit_model import ImplicitVideo_Hash, Deform_Hash3d_Warp, AnnealedHash

def save_video_cv2(save_path, frames, fps=15):
    if len(frames) == 0: return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
    print(f"Video saved to {save_path}")

def load_segment_model(ckpt_path, hash_config, hparams, device):
    """
    加载单个片段的模型
    """
    canonical_model = ImplicitVideo_Hash(hash_config).to(device)
    deform_model = Deform_Hash3d_Warp(hash_config).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    canonical_model.load_state_dict(checkpoint['canonical'])
    deform_model.load_state_dict(checkpoint['deform'])
    
    canonical_model.eval()
    deform_model.eval()
    
    return canonical_model, deform_model

def main():
    hparams = get_opts()
    hparams.batch_size = 1
    
    video_name = hparams.exp_name
    save_dir = os.path.join('results', video_name)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(f'cuda:{hparams.gpu}' if torch.cuda.is_available() else 'cpu')

    # Data
    dataset = VideoDataset(hparams.root_dir, hparams.img_wh, flow_dir=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # === 1. 恢复分割信息 ===
    model_dir = hparams.weight_path if os.path.isdir(hparams.weight_path) else os.path.dirname(hparams.weight_path)
    meta_path = os.path.join(model_dir, 'segments_meta.npy')
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Segment meta file not found at {meta_path}.")
    
    meta_data = np.load(meta_path, allow_pickle=True).item()
    segments = meta_data['segments'] # list of (start, end)
    
    print(f"Loaded segmentation meta: {len(segments)} segments. Running HARD STITCHING mode.")

    # === 2. 加载所有模型 ===
    with open('configs/hash.json') as f:
        hash_config = json.load(f)

    models_list = []
    target_ckpt_name = "ckpt_final.pt" 
    
    print("Pre-loading all segment models...")
    for k in range(len(segments)):
        seg_dir = os.path.join(model_dir, f'segment_{k}')
        ckpt_path = os.path.join(seg_dir, target_ckpt_name)
        
        if not os.path.exists(ckpt_path):
            ckpts = sorted([f for f in os.listdir(seg_dir) if f.endswith('.pt')])
            if len(ckpts) > 0:
                ckpt_path = os.path.join(seg_dir, ckpts[-1])
            else:
                raise FileNotFoundError(f"No checkpoints found in {seg_dir}")
        
        canon, deform = load_segment_model(ckpt_path, hash_config, hparams, device)
        models_list.append((canon, deform))

    # Annealing
    anneal_model = AnnealedHash(in_channels=3, 
                                annealed_step=hparams.annealed_step, 
                                annealed_begin_step=hparams.annealed_begin_step).to(device)

    # Grid
    W, H = hparams.img_wh
    Y, X = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    grid_coords = torch.stack([X, Y], dim=-1).view(-1, 2).to(device) 
    
    frames_buffer = []
    
    test_step = hparams.annealed_step + 100 
    
    # 构建查找表
    frame_to_segment = np.zeros(len(dataset), dtype=int)
    # 同时存储每个 segment 的 start，以便计算局部时间
    segment_starts = np.zeros(len(dataset), dtype=int)
    segment_lengths = np.zeros(len(dataset), dtype=int)

    for k, (start, end) in enumerate(segments):
        safe_end = min(end, len(dataset))
        frame_to_segment[start:safe_end] = k
        segment_starts[start:safe_end] = start
        segment_lengths[start:safe_end] = end - start
    
    print("Starting Inference (Local Time)...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            idx = batch['idx'].to(device) # Shape (1)
            global_t_idx = idx.item()
            
            # 1. 确定归属片段
            if global_t_idx < len(frame_to_segment):
                seg_id = frame_to_segment[global_t_idx]
                seg_start = segment_starts[global_t_idx]
                seg_len = segment_lengths[global_t_idx]
            else:
                seg_id = len(models_list) - 1
                seg_start = segments[-1][0]
                seg_len = segments[-1][1] - seg_start
            
            # 2. 获取模型
            canon_model, deform_model = models_list[seg_id]
            
            B = 1
            N_pixels = H * W
            
            # === 修改点: 计算局部时间 ===
            # (t - t_start) / (len - 1)
            local_time_val = (float(global_t_idx) - seg_start) / max(float(seg_len) - 1.0, 1.0)
            # 限制在 [0, 1] 之间，虽然理论上不会超
            local_time_val = np.clip(local_time_val, 0.0, 1.0)
            
            time_val = torch.tensor(local_time_val, device=device).view(B, 1, 1).repeat(1, N_pixels, 1)
            
            batch_grid = grid_coords.unsqueeze(0) 
            deform_input = torch.cat([batch_grid, time_val], dim=-1).view(-1, 3)

            # 3. 单模型推理
            with torch.cuda.amp.autocast():
                # Deform
                deform_out = deform_model(deform_input, step=test_step, aneal_func=anneal_model.forward)
                deform_out = deform_out.view(B, N_pixels, 2)
                
                # Warp
                canonical_coords_2d = batch_grid + deform_out 
                canonical_coords_flat = canonical_coords_2d.view(-1, 2)
                
                # Canon
                rgb_pred = canon_model(canonical_coords_flat) # (HW, 3)
                
                # Reshape
                rgb_final = rgb_pred.view(H, W, 3)
            
            pred_img = rgb_final.cpu().float().numpy()
            pred_img = np.clip(pred_img, 0, 1) * 255
            pred_img = pred_img.astype(np.uint8)
            
            frames_buffer.append(pred_img)

    video_save_path = os.path.join(save_dir, f"{video_name}_hard_stitched_local.mp4")
    fps = getattr(hparams, 'fps', 15)
    save_video_cv2(video_save_path, frames_buffer, fps=fps)

if __name__ == '__main__':
    main()