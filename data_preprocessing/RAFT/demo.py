import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)
        
        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # Compute flow
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            # Save raw flow for training
            # flow_up is (1, 2, H, W)
            flow_np = flow_up[0].cpu().numpy()
            np.save(os.path.join(args.outdir, f'{i:06d}.npy'), flow_np)
            
            # Visualization (Optional)
            if not args.no_viz:
                flo = flow_up[0].permute(1,2,0).cpu().numpy()
                flo_img = flow_viz.flow_to_image(flo)
                img_flo = np.concatenate([image1[0].permute(1,2,0).cpu().numpy(), flo_img], axis=0)
                cv2.imwrite(os.path.join(args.outdir, f'viz_{i:06d}.png'), img_flo[:, :, [2,1,0]]) # BGR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", required=True)
    parser.add_argument('--path', help="dataset path", required=True)
    parser.add_argument('--outdir', help="output path", required=True)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use alternate correlation')
    parser.add_argument('--no_viz', action='store_true', help='skip visualization')
    args = parser.parse_args()

    demo(args)