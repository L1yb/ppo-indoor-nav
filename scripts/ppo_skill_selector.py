#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions import Categorical
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import os

# 确保CUDA可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SkillActor(nn.Module):
    def __init__(self, state_dim, skill_num):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, skill_num),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class SkillCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class PPOSkillSelector:
    def __init__(self):
        rospy.init_node('ppo_skill_selector')
        
        # 参数设置
        self.state_dim = 12  # 雷达特征(8) + 目标距离角度(2) + 机器人朝向(2)
        self.skill_num = 5   # 基本技能数量: 前进、左小转、右小转、左大转、右大转
        self.lr = 3e-4
        self.gamma = 0.99
        self.lam = 0.95      # GAE lambda参数
        self.clip_ratio = 0.2
        self.max_grad_norm = 0.5
        self.batch_size = 64
        
        # 初始化网络
        self.actor = SkillActor(self.state_dim, self.skill_num).to(device)
        self.critic = SkillCritic(self.state_dim).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)
        
        # 如果存在模型则加载
        self.model_path = os.path.expanduser("~/catkin_ws/src/ppo_indoor_navigation/model")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.load_model()
        
        # 初始化订阅和发布
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 状态变量
        self.laser_data = None
        self.robot_pos = None
        self.robot_yaw = None
        self.goal_pos = None
        self.last_state = None
        self.last_action = None
        
        # 缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # 技能定义
        self.skills = [
            (0.3, 0.0),     # 直行
            (0.2, 0.3),     # 小左转
            (0.2, -0.3),    # 小右转
            (0.15, 0.6),    # 大左转
            (0.15, -0.6)    # 大右转
        ]
        
        # 奖励参数
        self.goal_radius = 0.5  # 目标达成半径
        self.obstacle_radius = 0.4  # 障碍物检测半径
        
        # 训练控制参数
        self.training = True
        self.update_interval = 500  # 更新间隔
        self.save_interval = 5000   # 保存间隔
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0
        
        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.loginfo("PPO技能选择器已初始化")
        
    def laser_callback(self, msg):
        """处理激光雷达数据"""
        self.laser_data = np.array(msg.ranges)
        # 替换inf值
        self.laser_data[np.isinf(self.laser_data)] = msg.range_max
        
    def odom_callback(self, msg):
        """处理里程计数据"""
        pos = msg.pose.pose.position
        self.robot_pos = np.array([pos.x, pos.y])
        
        # 提取朝向
        quat = msg.pose.pose.orientation
        _, _, self.robot_yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    
    def goal_callback(self, msg):
        """处理目标点数据"""
        self.goal_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        rospy.loginfo(f"收到新目标点: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})")
        
    def get_state(self):
        """构建状态表示"""
        if self.laser_data is None or self.robot_pos is None or self.goal_pos is None:
            return None
            
        # 雷达特征提取 (降维到8个方向)
        laser_bins = []
        bin_size = len(self.laser_data) // 8
        for i in range(8):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size
            laser_bins.append(np.min(self.laser_data[start_idx:end_idx]))
        laser_features = np.array(laser_bins) / 5.0  # 归一化
        
        # 目标点相对位置 (距离和角度)
        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        goal_dist = np.sqrt(dx*dx + dy*dy)
        goal_angle = np.arctan2(dy, dx) - self.robot_yaw
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))  # 归一化到[-pi, pi]
        
        # 构建完整状态向量
        state = np.concatenate([
            laser_features,  # 8维雷达特征
            [goal_dist/10.0, goal_angle/np.pi],  # 归一化的目标距离和角度
            [np.cos(self.robot_yaw), np.sin(self.robot_yaw)]  # 机器人朝向
        ])
        
        return torch.FloatTensor(state).to(device)
    
    def select_skill(self, state):
        """选择技能（动作）"""
        with torch.no_grad():
            probs = self.actor(state)
            value = self.critic(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
    
    def execute_skill(self, skill_idx):
        """执行选定的技能"""
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = self.skills[skill_idx]
        self.cmd_vel_pub.publish(cmd)
    
    def calculate_reward(self, state, next_state, action):
        """计算奖励"""
        # 从状态中提取相关信息
        state_np = state.cpu().numpy()
        next_state_np = next_state.cpu().numpy()
        
        laser_features = state_np[:8]
        goal_dist = state_np[8] * 10.0  # 还原原始距离
        
        next_laser_features = next_state_np[:8]
        next_goal_dist = next_state_np[8] * 10.0
        
        # 计算各项奖励
        # 1. 接近目标奖励
        progress_reward = (goal_dist - next_goal_dist) * 10.0
        
        # 2. 碰撞惩罚
        collision_penalty = -20.0 if np.min(next_laser_features) < 0.1 else 0.0
        
        # 3. 接近障碍物惩罚
        obstacle_penalty = -0.5 * np.exp(-5 * np.min(next_laser_features))
        
        # 4. 目标达成奖励
        goal_reward = 50.0 if next_goal_dist < self.goal_radius else 0.0
        
        # 5. 平滑性奖励 (鼓励适当时使用直行)
        smoothness_reward = 0.2 if action == 0 and np.min(next_laser_features) > 0.5 else 0.0
        
        total_reward = progress_reward + collision_penalty + obstacle_penalty + goal_reward + smoothness_reward
        done = (next_goal_dist < self.goal_radius) or (np.min(next_laser_features) < 0.1)
        
        return total_reward, done
    
    def compute_gae(self):
        """计算广义优势估计(GAE)"""
        values = torch.cat(self.values).flatten()
        rewards = torch.tensor(self.rewards, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # 计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # 终止状态的值
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            
        # 计算目标回报
        returns = advantages + values
        
        return advantages, returns
    
    def update_policy(self):
        """更新PPO策略"""
        if len(self.states) < self.batch_size:
            return
            
        # 计算GAE
        advantages, returns = self.compute_gae()
        
        # 准备数据
        states = torch.cat(self.states)
        actions = torch.tensor(self.actions, device=device)
        old_log_probs = torch.cat(self.log_probs)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮训练
        for epoch in range(5):
            # 计算策略比率
            logits = self.actor(states)
            dist = Categorical(logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算PPO比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Actor损失
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # Critic损失
            values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(values, returns)
            
            # 梯度更新
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # 记录训练信息
        rospy.loginfo(f"策略更新 - Episode: {self.episode_count}, Step: {self.step_count}, Reward: {self.episode_reward:.2f}")
        self.episode_reward = 0
        self.episode_count += 1
        
        # 定期保存模型
        if self.step_count % self.save_interval == 0:
            self.save_model()
    
    def control_loop(self, event):
        """主控制循环"""
        state = self.get_state()
        if state is None:
            return
            
        # 选择动作
        action, log_prob, value = self.select_skill(state)
        
        # 执行动作
        self.execute_skill(action)
        
        # 训练流程
        if self.training and self.last_state is not None:
            # 计算奖励
            reward, done = self.calculate_reward(self.last_state, state, self.last_action)
            self.episode_reward += reward
            
            # 存储经验
            self.states.append(self.last_state.unsqueeze(0))
            self.actions.append(self.last_action)
            self.rewards.append(reward)
            self.log_probs.append(self.last_log_prob.unsqueeze(0))
            self.values.append(self.last_value)
            self.dones.append(done)
            
            # 检查是否更新模型
            self.step_count += 1
            if self.step_count % self.update_interval == 0 or done:
                self.update_policy()
                
            # 目标达成通知
            if done and reward > 0:
                rospy.loginfo("目标点到达成功！")
        
        # 保存当前状态作为下一步的上一状态
        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = value

    def save_model(self):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'step_count': self.step_count
        }, os.path.join(self.model_path, 'ppo_skill_model.pt'))
        rospy.loginfo(f"模型已保存: {self.model_path}/ppo_skill_model.pt")
        
    def load_model(self):
        """加载模型"""
        model_file = os.path.join(self.model_path, 'ppo_skill_model.pt')
        if os.path.exists(model_file):
            try:
                checkpoint = torch.load(model_file, map_location=device)
                self.actor.load_state_dict(checkpoint['actor'])
                self.critic.load_state_dict(checkpoint['critic'])
                self.step_count = checkpoint['step_count']
                rospy.loginfo(f"成功加载模型: {model_file}")
            except Exception as e:
                rospy.logwarn(f"加载模型失败: {e}")
        else:
            rospy.loginfo("没有找到预训练模型，将使用新模型")

if __name__ == '__main__':
    try:
        skill_selector = PPOSkillSelector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 