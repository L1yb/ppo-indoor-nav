import numpy as np
import torch
from geometry_msgs.msg import Twist

class MotionPrimitives:
    def __init__(self, radar_range=5.0, num_beams=180):
        self.radar_range = radar_range
        self.num_beams = num_beams
        self.safety_threshold = 0.3  # 安全停止距离

    def preprocess_radar(self, raw_scan):
        """预处理二维雷达数据
        Args:
            raw_scan (list): 原始雷达数据 [距离1, 角度1, 距离2, 角度2,...]
        Returns:
            torch.Tensor: 处理后的状态向量 (batch_size, state_dim)
        """
        # 转换为极坐标矩阵
        ranges = np.array(raw_scan[::2])
        angles = np.deg2rad(np.array(raw_scan[1::2]))
        
        # 生成笛卡尔坐标
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # 生成障碍物特征向量
        # 更新前向检测区域为±30度
        forward_mask = (np.abs(angles) < np.deg2rad(30))
        
        # 新增侧方障碍物检测区域（±60-90度）
        side_mask = (np.abs(angles) > np.deg2rad(60)) & (np.abs(angles) < np.deg2rad(90))
        
        # 计算障碍物密度特征
        forward_density = np.sum(ranges[forward_mask] < self.radar_range*0.5) / len(ranges[forward_mask]) if len(ranges[forward_mask]) > 0 else 0.0
        side_density = np.sum(ranges[side_mask] < self.radar_range*0.8) / len(ranges[side_mask]) if len(ranges[side_mask]) > 0 else 0.0
        
        return torch.FloatTensor([
            min_forward_dist / self.radar_range,
            np.mean(ranges) / self.radar_range,
            np.std(ranges) / self.radar_range,
            np.min(ranges) / self.radar_range,
            forward_density,
            side_density
        ])

    def action_to_cmdvel(self, action_idx):
        """将动作索引转换为具体的运动参数
        Args:
            action_idx (int): PPO网络输出的动作索引
        Returns:
            Twist: ROS速度指令
        """
        cmd = Twist()
        # 运动基元定义（需根据实际小车动力学调整）
        primitives = [
            (0.2, 0.0),    # 低速直行
            (0.15, 0.3),  # 小左转
            (0.15, -0.3), # 小右转
            (0.1, 0.6),   # 中左转
            (0.1, -0.6),  # 中右转
            (0.0, 1.0),   # 原地左转（保留）
            (0.0, -1.0)   # 原地右转（保留）
        ]
        linear, angular = primitives[action_idx]
        cmd.linear.x = linear
        cmd.angular.z = angular
        return cmd

    def calculate_reward(self, state, new_state):
        """计算即时奖励
        Args:
            state (torch.Tensor): 当前状态
            new_state (torch.Tensor): 新状态
        Returns:
            float: 奖励值
        """
        progress_reward = (state[0] - new_state[0]) * 10  # 前进奖励
        collision_penalty = -10 if new_state[0] < self.safety_threshold else 0  # 使用前向最小距离判断
        smooth_penalty = -0.3 * torch.abs(state[4] - new_state[4])
        return float(progress_reward + collision_penalty + smooth_penalty)

    def safety_check(self, state):
        return state[0].item() < self.safety_threshold  # 使用前向最小距离判断