import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import numpy as np
import cv2

from opt import get_opts
from dataset import VideoDataset
from models.implicit_model import ImplicitVideo_Hash, Deform_Hash3d_Warp, AnnealedHash
from losses import MSELoss
from utils import get_scheduler, get_optimizer
from utils.image_utils import save_image

# 引入分割模块
from motion_segmentation import AdaptiveMotionSegmenter

def warp_flow(flow_next, flow_curr):
    """
    通过 flow_curr 对 flow_next 进行 warp，以合成 flow_curr + warped_flow_next
    """
    B, _, H, W = flow_curr.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1).to(flow_curr.device)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W).to(flow_curr.device)
    grid = torch.cat([xx, yy], 1) # (B, 2, H, W)
    
    flow_curr_norm = flow_curr.clone()
    flow_curr_norm[:, 0, :, :] /= (W / 2.0)
    flow_curr_norm[:, 1, :, :] /= (H / 2.0)
    
    vgrid = (grid + flow_curr_norm).permute(0, 2, 3, 1) 
    
    warped_flow_next = F.grid_sample(flow_next, vgrid, align_corners=True, mode='bilinear', padding_mode='border')
    
    return warped_flow_next

def train_segment(hparams, segment_idx, segment_indices, full_dataset, grid_coords, hash_config):
    device = torch.device(f'cuda:{hparams.gpu}' if torch.cuda.is_available() else 'cpu')
    
    seg_model_save_path = os.path.join(hparams.model_save_path, f'segment_{segment_idx}')
    os.makedirs(seg_model_save_path, exist_ok=True)
    
    # 获取当前片段的起止帧索引，用于计算局部时间
    seg_start = segment_indices[0]
    seg_end = segment_indices[-1] + 1 # exclusive end
    seg_len = seg_end - seg_start
    
    print(f"\n[{segment_idx}] Training Segment: Frames {seg_start} to {seg_end-1} (Len: {seg_len})")

    train_dataset = Subset(full_dataset, segment_indices)
    dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=4)

    canonical_model = ImplicitVideo_Hash(hash_config).to(device)
    deform_model = Deform_Hash3d_Warp(hash_config).to(device)
    anneal_model = AnnealedHash(in_channels=3, 
                                annealed_step=hparams.annealed_step, 
                                annealed_begin_step=hparams.annealed_begin_step).to(device)

    models_to_opt = [canonical_model, deform_model]
    optimizer = get_optimizer(hparams, models_to_opt)
    scheduler = get_scheduler(hparams, optimizer)
    scaler = torch.cuda.amp.GradScaler()
    criterion_mse = MSELoss()

    steps_per_segment = hparams.num_steps 
    progress_bar = tqdm(range(steps_per_segment), desc=f"Seg {segment_idx}")
    data_iter = iter(dataloader)

    W, H = hparams.img_wh
    
    # Hyperparameters
    lambda_mf = 0.05 

    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        imgs = batch['img'].to(device)
        idx = batch['idx'].to(device) # 全局索引
        B = imgs.shape[0]

        # === 修改点 1: 计算局部时间 ===
        # 将全局索引映射到 [0, 1] 区间，基于当前片段长度
        # (idx - seg_start) / (seg_len - 1)
        # 加上 1e-8 防止除以0
        local_time = (idx.float() - seg_start) / (max(seg_len - 1, 1))
        time_val = local_time.view(B, 1, 1).repeat(1, H*W, 1) 
        
        batch_grid = grid_coords.unsqueeze(0).repeat(B, 1, 1) 
        deform_input = torch.cat([batch_grid, time_val], dim=-1).view(-1, 3) 
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # 1. Forward Deformation (t)
            deform_out = deform_model(deform_input, step=step, aneal_func=anneal_model.forward)
            deform_out = deform_out.view(B, H*W, 2)
            
            D_p_t = deform_out 
            
            canonical_coords_2d = batch_grid + deform_out 
            canonical_coords_flat = canonical_coords_2d.view(-1, 2)
            
            # 2. Forward Canonical (Appearance)
            rgb_pred = canonical_model(canonical_coords_flat)
            rgb_pred = rgb_pred.view(B, H*W, 3)
            
            gt_rgb = imgs.permute(0, 2, 3, 1).reshape(B, H*W, 3)
            loss_rgb = criterion_mse(rgb_pred, gt_rgb)
            
            total_loss = loss_rgb

            # =========================================================
            # Multi-Frame Flow Consistency Loss (L_mf)
            # =========================================================
            if 'flow_fwd' in batch and 'flow_next' in batch:
                flow_t_t1 = batch['flow_fwd'].to(device)   
                flow_t1_t2 = batch['flow_next'].to(device) 
                
                warped_flow_next = warp_flow(flow_t1_t2, flow_t_t1)
                flow_t_t2 = flow_t_t1 + warped_flow_next 
                
                flow_t_t2_flat = flow_t_t2.permute(0, 2, 3, 1).reshape(B, H*W, 2)
                
                # --- Compute D(p + F_{t->t+2}, t+2) ---
                flow_norm_x = flow_t_t2_flat[:, :, 0] / (W / 2.0)
                flow_norm_y = flow_t_t2_flat[:, :, 1] / (H / 2.0)
                flow_norm = torch.stack([flow_norm_x, flow_norm_y], dim=-1)
                
                pos_t2 = batch_grid + flow_norm 
                
                # === 修改点 2: L_mf 中的时间也要用局部时间 ===
                # 下一时刻 t+2 的全局索引
                next_global_idx = idx + 2
                
                # 如果 t+2 超出了当前片段范围，我们需要截断或者忽略
                # 这里简单处理：如果 t+2 还在当前片段内，就计算 loss；否则忽略
                # (B, 1)
                valid_mask = (next_global_idx < seg_end).float().view(B, 1)
                
                if valid_mask.sum() > 0:
                    local_time_t2 = (next_global_idx.float() - seg_start) / (max(seg_len - 1, 1))
                    time_val_t2 = local_time_t2.view(B, 1, 1).repeat(1, H*W, 1)
                    
                    deform_input_t2 = torch.cat([pos_t2, time_val_t2], dim=-1).view(-1, 3)
                    
                    D_p_plus_flow_t2 = deform_model(deform_input_t2, step=step, aneal_func=anneal_model.forward)
                    D_p_plus_flow_t2 = D_p_plus_flow_t2.view(B, H*W, 2)
                    
                    mask = (pos_t2.abs() <= 1.0).all(dim=-1) 
                    
                    if mask.sum() > 0:
                        diff = D_p_t - D_p_plus_flow_t2
                        loss_mf = (diff.norm(dim=-1) * mask).sum() / (mask.sum() + 1e-6)
                        # 乘以 valid_mask 确保不计算越界帧的 loss
                        total_loss += lambda_mf * loss_mf * valid_mask.mean()
                else:
                    loss_mf = torch.tensor(0.0).to(device)

            else:
                 loss_mf = torch.tensor(0.0).to(device)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 100 == 0:
            progress_bar.set_postfix({'loss': loss_rgb.item(), 'l_mf': loss_mf.item()})

        # Save Checkpoint
        if step % hparams.save_model_iters == 0 and step > 0:
            torch.save({
                'canonical': canonical_model.state_dict(),
                'deform': deform_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'segment_info': {'start': seg_start, 'end': seg_end}
            }, os.path.join(seg_model_save_path, f'ckpt_{step}.pt'))
            
    torch.save({
        'canonical': canonical_model.state_dict(),
        'deform': deform_model.state_dict(),
    }, os.path.join(seg_model_save_path, f'ckpt_final.pt'))

def main():
    hparams = get_opts()
    
    os.makedirs(hparams.log_save_path, exist_ok=True)
    os.makedirs(hparams.model_save_path, exist_ok=True)

    # ================= 优先加载 segmentation.json =================
    json_path = os.path.join(hparams.root_dir, 'segmentation.json')
    
    if os.path.exists(json_path):
        print(f"Loading pre-computed segmentation info from {json_path}")
        with open(json_path, 'r') as f:
            seg_data = json.load(f)
        segments = [(s['start_idx'], s['end_idx']) for s in seg_data['segments']]
        tau = seg_data['threshold_tau']
        total_frames = seg_data['total_frames']
        print(f"Loaded {len(segments)} segments (Threshold: {tau:.2f})")
    else:
        print("Warning: segmentation.json not found. Running online segmentation (default threshold 1.5).")
        import glob
        files = sorted(glob.glob(os.path.join(hparams.root_dir, "*.jpg")) + glob.glob(os.path.join(hparams.root_dir, "*.png")))
        frames = []
        for f in files:
            img = cv2.imread(f)
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        analysis_frames = np.array(frames)

        segmenter = AdaptiveMotionSegmenter(threshold_factor=1.5, min_segment_length=10)
        segments, motion_scores, tau = segmenter.segment_video(analysis_frames)
        total_frames = len(analysis_frames)
        print(f"Online segmentation result: {len(segments)} segments.")

    # 保存元数据供 Test 阶段使用
    np.save(os.path.join(hparams.model_save_path, 'segments_meta.npy'), {
        'segments': segments,
        'tau': tau,
        'total_frames': total_frames
    })

    # 2. 全量数据集
    full_dataset = VideoDataset(hparams.root_dir, hparams.img_wh, hparams.flow_dir)
    
    device = torch.device(f'cuda:{hparams.gpu}' if torch.cuda.is_available() else 'cpu')
    W, H = hparams.img_wh
    Y, X = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    grid_coords = torch.stack([X, Y], dim=-1).view(-1, 2).to(device)

    with open('configs/hash.json') as f:
        hash_config = json.load(f)

    # 3. 循环训练
    for k, (start, end) in enumerate(segments):
        segment_indices = list(range(start, end))
        train_segment(
            hparams, 
            segment_idx=k, 
            segment_indices=segment_indices, 
            full_dataset=full_dataset, 
            grid_coords=grid_coords, 
            hash_config=hash_config
        )

if __name__ == '__main__':
    main()