import argparse
import os
import json
import cv2
import numpy as np
from motion_segmentation import AdaptiveMotionSegmenter

def load_images_from_dir(directory):
    """从目录加载图片序列作为视频帧"""
    files = sorted([f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
    frames = []
    print(f"Loading {len(files)} frames from {directory}...")
    for f in files:
        path = os.path.join(directory, f)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.array(frames), files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Path to video frames')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save segmentation info json')
    parser.add_argument('--threshold_factor', type=float, default=0.1)
    args = parser.parse_args()

    # 1. 加载数据
    frames, filenames = load_images_from_dir(args.root_dir)
    
    # 2. 运行分割模块
    segmenter = AdaptiveMotionSegmenter(threshold_factor=args.threshold_factor, min_segment_length=10)
    segments, motion_scores, tau = segmenter.segment_video(frames)
    
    # 3. 组织元数据
    # 我们不仅保存索引，最好保存文件名，方便后续Dataset读取
    segment_info = []
    for idx, (start, end) in enumerate(segments):
        # 注意: end 在切片中是 exclusive 的，但文件名列表索引是 inclusive
        # 这里保存为 [start_idx, end_idx) 格式
        segment_data = {
            "id": idx,
            "start_idx": int(start),
            "end_idx": int(end),
            "filenames": filenames[start:end]
        }
        segment_info.append(segment_data)

    output_data = {
        "total_frames": len(frames),
        "threshold_tau": float(tau),
        "segments": segment_info
    }

    # 4. 保存为 JSON
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Segmentation done! Found {len(segments)} segments. Info saved to {args.save_path}")

if __name__ == "__main__":
    main()