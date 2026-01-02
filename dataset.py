import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from glob import glob
from utils.image_utils import resize_image

class VideoDataset(Dataset):
    def __init__(self, root_dir, img_wh, flow_dir=None):
        self.root_dir = root_dir
        self.img_wh = img_wh
        self.flow_dir = flow_dir

        self.image_paths = sorted(glob(os.path.join(root_dir, '*.png')) + 
                                  glob(os.path.join(root_dir, '*.jpg')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        # Check for flow files
        self.flow_paths = []
        if self.flow_dir:
            self.flow_paths = sorted(glob(os.path.join(self.flow_dir, '*.npy')))
            if len(self.flow_paths) == 0:
                print("Warning: No flow files found.")

    def __len__(self):
        return len(self.image_paths)

    def load_flow(self, idx):
        """Helper to load and resize flow"""
        if idx >= len(self.flow_paths):
            return None
            
        flow_path = self.flow_paths[idx]
        flow = np.load(flow_path) # Expecting (2, H, W) or (H, W, 2)
        
        if flow.shape[0] != 2: 
            flow = flow.transpose(2, 0, 1)
        
        flow_tensor = torch.from_numpy(flow).float()
        
        # Resize flow
        if flow_tensor.shape[1] != self.img_wh[1] or flow_tensor.shape[2] != self.img_wh[0]:
            orig_h, orig_w = flow_tensor.shape[1], flow_tensor.shape[2]
            flow_tensor = torch.nn.functional.interpolate(flow_tensor.unsqueeze(0), size=(self.img_wh[1], self.img_wh[0]), mode='bilinear', align_corners=True).squeeze(0)
            flow_tensor[0] *= (self.img_wh[0] / orig_w)
            flow_tensor[1] *= (self.img_wh[1] / orig_h)
            
        return flow_tensor

    def __getitem__(self, idx):
        # Load Image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_image(img, (self.img_wh[0], self.img_wh[1]))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1) # (3, H, W)

        sample = {
            'img': img_tensor,
            'idx': idx,
            'img_path': img_path
        }

        # Load Flow: flow_fwd (t -> t+1) and flow_next (t+1 -> t+2)
        # 这里的 flow_fwd 是当前帧到下一帧的光流
        if self.flow_dir:
            # 1. Flow t -> t+1
            flow_fwd = self.load_flow(idx)
            if flow_fwd is not None:
                sample['flow_fwd'] = flow_fwd
            
            # 2. Flow t+1 -> t+2 (用于计算 Multi-Frame Consistency)
            # 只有当 idx+1 也存在有效光流时才加载
            flow_next = self.load_flow(idx + 1)
            if flow_next is not None:
                sample['flow_next'] = flow_next

        return sample