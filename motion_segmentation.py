import cv2
import numpy as np
import matplotlib.pyplot as plt

class AdaptiveMotionSegmenter:
    def __init__(self, threshold_factor=1.5, min_segment_length=10):
        """
        初始化自适应运动分割器
        :param threshold_factor: 动态阈值系数 (tau = mean + factor * std)
        :param min_segment_length: 防止视频被切分得过碎，设置最小片段长度
        """
        self.threshold_factor = threshold_factor
        self.min_segment_length = min_segment_length

    def compute_motion_magnitudes(self, frames):
        """
        实现公式 (1): 计算每一步的帧间运动幅度 Mt
        Mt = (1/|Omega|) * sum( || I_t(p) - I_{t-1}(p) ||_2 )
        """
        motion_magnitudes = [0.0] # 第一帧没有前一帧，设为0
        
        print("正在计算帧间运动幅度...")
        for t in range(1, len(frames)):
            # 转换为浮点数进行计算，防止uint8溢出
            curr_frame = frames[t].astype(np.float32)
            prev_frame = frames[t-1].astype(np.float32)
            
            # 计算像素级差分向量
            diff = curr_frame - prev_frame
            
            # 计算 L2 范数 (对 RGB 通道求欧几里得距离)
            # axis=2 表示在颜色通道维度上求范数
            norm_diff = np.linalg.norm(diff, axis=2)
            
            # 对所有像素求平均 (1/|Omega| * sum(...))
            mt = np.mean(norm_diff)
            motion_magnitudes.append(mt)
            
        return np.array(motion_magnitudes)

    def get_dynamic_threshold(self, motion_magnitudes):
        """
        基于视频运动统计信息动态确定阈值 tau
        这里使用统计学方法: Mean + k * Std
        """
        # 忽略第一帧的0值
        valid_magnitudes = motion_magnitudes[1:]
        
        mean_val = np.mean(valid_magnitudes)
        std_val = np.std(valid_magnitudes)
        
        tau = mean_val + self.threshold_factor * std_val
        return tau

    def segment_video(self, frames):
        """
        核心流程：计算运动 -> 确定阈值 -> 分割视频
        返回: 
        - segments: 列表，每个元素为 [start_index, end_index]
        - motion_magnitudes: 计算出的运动幅度数组
        - tau: 计算出的阈值
        """
        n_frames = len(frames)
        if n_frames < 2:
            return [[0, n_frames-1]], np.array([0]), 0

        # 1. 计算运动幅度 Mt
        mts = self.compute_motion_magnitudes(frames)
        
        # 2. 动态确定阈值 tau
        tau = self.get_dynamic_threshold(mts)
        
        # 3. 识别峰值并分割
        # 寻找 Mt > tau 的帧作为潜在分割点
        boundaries = [0] # 起始帧
        
        last_boundary = 0
        for t in range(1, n_frames):
            # 如果运动过快 (Mt > tau) 且距离上一个分割点足够远 (避免过碎)
            if mts[t] > tau and (t - last_boundary) >= self.min_segment_length:
                # 这是一个运动剧烈的时刻，将其视为新片段的开始
                boundaries.append(t)
                last_boundary = t
        
        # 确保包含最后一帧
        if boundaries[-1] != n_frames:
            boundaries.append(n_frames)
            
        # 构建片段区间 [start, end)
        segments = []
        for i in range(len(boundaries) - 1):
            segments.append((boundaries[i], boundaries[i+1]))
            
        return segments, mts, tau

    def compute_blending_weights(self, total_frames, segments, sigma_scale=0.5):
        """
        实现公式 (2) 的辅助部分: 计算高斯混合权重 w_k(t)
        
        :param total_frames: 视频总帧数
        :param segments: 分割列表 [(start, end), ...]
        :param sigma_scale: 控制高斯分布宽度的系数
        :return: 权重矩阵，形状为 (K, T)，K是片段数，T是总帧数
        """
        K = len(segments)
        T = total_frames
        weights = np.zeros((K, T))
        
        time_steps = np.arange(T)
        
        for k, (start, end) in enumerate(segments):
            # 片段的时间中点
            midpoint = (start + end - 1) / 2.0
            
            # 片段长度
            length = end - start
            
            # 高斯分布的标准差，与片段长度相关，确保覆盖该片段
            sigma = length * sigma_scale
            
            # 计算高斯权重 w_k(t) = exp( - (t - center)^2 / (2 * sigma^2) )
            w_k = np.exp(-0.5 * ((time_steps - midpoint) / sigma) ** 2)
            
            weights[k, :] = w_k
            
        # 归一化权重，确保每个时间点 t，所有 K 个片段的权重之和为 1
        # sum_weights = np.sum(weights, axis=0)
        # weights = weights / (sum_weights + 1e-8)
        
        return weights

# ================= 使用示例 =================

def load_video_frames(video_path, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换 BGR 到 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

# 假设有一个名为 'input_video.mp4' 的视频
# frames = load_video_frames('input_video.mp4') 

# 为了演示，我们生成一些模拟数据
# 模拟 100 帧，中间有一段快速运动
print("生成模拟视频数据...")
frames = np.zeros((100, 64, 64, 3), dtype=np.uint8)
# 0-30帧: 静止
frames[0:30] = np.random.randint(0, 50, (30, 64, 64, 3))
# 30-40帧: 剧烈运动 (噪声变化大)
frames[30:40] = np.random.randint(0, 255, (10, 64, 64, 3)) 
# 40-100帧: 缓慢运动
for i in range(40, 100):
    frames[i] = frames[i-1] + np.random.randint(-5, 5, (64, 64, 3))
frames = np.clip(frames, 0, 255).astype(np.uint8)


# 1. 实例化分割器
segmenter = AdaptiveMotionSegmenter(threshold_factor=2.0, min_segment_length=5)

# 2. 执行分割
segments, motion_scores, tau = segmenter.segment_video(frames)

print(f"\n动态阈值 (tau): {tau:.2f}")
print(f"检测到的分割片段 (K={len(segments)}):")
for i, seg in enumerate(segments):
    print(f"  Segment {i+1}: Frames {seg[0]} to {seg[1]}")

# 3. 计算混合权重 (用于后续重建)
weights = segmenter.compute_blending_weights(len(frames), segments)

# 4. 可视化结果
plt.figure(figsize=(12, 6))

# 绘制运动幅度
plt.subplot(2, 1, 1)
plt.plot(motion_scores, label='Motion Magnitude $M_t$', color='blue')
plt.axhline(y=tau, color='red', linestyle='--', label=f'Threshold $\\tau$ ({tau:.1f})')
for seg in segments:
    plt.axvline(x=seg[0], color='green', linestyle=':', alpha=0.5)
plt.title('Motion Analysis and Segmentation Boundaries')
plt.legend()
plt.ylabel('Magnitude')

# 绘制混合权重
plt.subplot(2, 1, 2)
for k in range(weights.shape[0]):
    plt.plot(weights[k], label=f'Weight $w_{k+1}(t)$')
plt.title('Temporal Blending Weights $w_k(t)$')
plt.xlabel('Frame Index $t$')
plt.ylabel('Weight')
plt.legend()

plt.tight_layout()
plt.show()