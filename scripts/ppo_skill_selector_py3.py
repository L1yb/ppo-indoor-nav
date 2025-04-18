#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions import Categorical
from std_msgs.msg import Float32MultiArray, Int32
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
        rospy.init_node('ppo_skill_selector_py3')
        
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
        self.model_path = os.path.expanduser("~/catkin_ws/src/ppo_indoor_nav/model")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.load_model()
        
        # 初始化订阅和发布
        self.state_sub = rospy.Subscriber('/ppo/state', Float32MultiArray, self.state_callback)
        self.action_pub = rospy.Publisher('/ppo/action', Int32, queue_size=10)
        self.hint_pub = rospy.Publisher('/ppo/hint', Float32MultiArray, queue_size=10)
        
        # 状态变量
        self.current_state = None
        self.last_state = None
        self.last_action = None
        
        # 缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
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
        
        # 在初始化中增加
        self.exploration_temp = 1.0  # 探索温度参数
        
        # 添加降低探索的逻辑
        if self.step_count % 1000 == 0 and self.exploration_temp > 0.5:
            self.exploration_temp *= 0.95  # 逐渐降低探索
        
        rospy.loginfo("PPO技能选择器(Python 3)已初始化")
        
    def state_callback(self, msg):
        """处理从桥接节点接收的状态数据"""
        state_array = np.array(msg.data)
        self.current_state = torch.FloatTensor(state_array).to(device)
        
        # 执行PPO决策
        self.process_state()
        
    def process_state(self):
        """处理当前状态并做出决策"""
        if self.current_state is None:
            return
            
        # 选择动作
        action, log_prob, value = self.select_skill(self.current_state)
        
        # 发布动作
        action_msg = Int32()
        action_msg.data = action
        self.action_pub.publish(action_msg)
        
        # 训练流程
        if self.training and self.last_state is not None:
            # 计算奖励
            reward, done = self.calculate_reward(self.last_state, self.current_state, self.last_action)
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
                rospy.loginfo(f"目标点到达成功！")
            
            # 在process_state方法中增加碰撞后的重置逻辑
            if done and np.min(self.current_state.cpu().numpy()[:8]) < 0.1:  # 检测到碰撞
                rospy.loginfo("检测到碰撞，准备重置机器人位置")
                # 发布重置信号到仿真环境
                reset_msg = Int32()
                reset_msg.data = 1
                self.reset_pub.publish(reset_msg)
                
                # 清空当前状态
                self.last_state = None
                self.current_state = None
                self.episode_reward = 0
        
        # 保存当前状态作为下一步的上一状态
        self.last_state = self.current_state
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = value
        
        # 在process_state方法中增加提示发布
        next_laser_features = self.current_state.cpu().numpy()[:8]
        if np.min(next_laser_features) < 0.3:
            hint_msg = Float32MultiArray()
            # 根据激光雷达数据计算最佳逃离方向
            best_direction = np.argmax(next_laser_features)
            hint_msg.data = [best_direction]
            self.hint_pub.publish(hint_msg)
    
    def select_skill(self, state):
        """选择技能（动作）"""
        with torch.no_grad():
            probs = self.actor(state)
            value = self.critic(state)
            
            # 安全检查：如果前方障碍物很近，降低向前动作的概率
            laser_data = state.cpu().numpy()[:8]
            if np.min(laser_data[:3]) < 0.5:  # 前方三个方向有障碍物
                probs_np = probs.cpu().numpy()
                probs_np[0] *= 0.1  # 降低前进概率
                probs = torch.from_numpy(probs_np).to(device)
                
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
    
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
        obstacle_penalty = -2.0 * np.exp(-3 * np.min(next_laser_features))  # 增大惩罚系数
        
        # 3. 接近障碍物惩罚
        obstacle_trend_penalty = -1.0 if np.min(next_laser_features) < np.min(laser_features) else 0.0
        
        # 4. 目标达成奖励
        goal_reward = 50.0 if next_goal_dist < self.goal_radius else 0.0
        
        # 5. 平滑性奖励 (鼓励适当时使用直行)
        smoothness_reward = 0.2 if action == 0 and np.min(next_laser_features) > 0.5 else 0.0
        
        total_reward = progress_reward + obstacle_penalty + obstacle_trend_penalty + goal_reward + smoothness_reward
        done = (next_goal_dist < self.goal_radius) or (np.min(next_laser_features) < 0.1)
        
        return total_reward, done
    
    def compute_gae(self):
        """计算广义优势估计(GAE)"""
        values = torch.cat(self.values).flatten()
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
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