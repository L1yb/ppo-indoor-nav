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
        self.model_path = os.path.expanduser("~/catkin_ws/src/ppo-indoor-nav/model")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.load_model()
        
        # 初始化订阅和发布
        self.state_sub = rospy.Subscriber('/ppo/state', Float32MultiArray, self.state_callback)
        self.action_pub = rospy.Publisher('/ppo/action', Int32, queue_size=10)
        self.hint_pub = rospy.Publisher('/ppo/hint', Float32MultiArray, queue_size=10)
        # 添加重置发布者
        self.reset_pub = rospy.Publisher('/ppo/reset', Int32, queue_size=10)
        
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
        
        # 探索和引导参数
        self.exploration_temp = 1.0  # 探索温度参数
        self.initial_exploration_rate = 0.95  # 提高初始探索率，使干预更有效
        self.min_exploration_rate = 0.3  # 提高最小探索率，确保长期干预效果
        self.exploration_decay = 0.9998  # 降低衰减速度，使干预持续更长时间
        self.current_exploration_rate = self.initial_exploration_rate
        
        # 路径跟踪和修正参数
        self.prev_goal_dists = []  # 历史目标距离
        self.max_history_length = 10  # 历史长度限制
        self.wrong_direction_count = 0  # 错误方向计数
        self.max_wrong_direction = 5  # 允许的最大错误方向次数
        
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
            # 提取目标距离信息用于轨迹监控
            next_state_np = self.current_state.cpu().numpy()
            next_goal_dist = next_state_np[8] * 10.0
            
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
            if done and next_goal_dist < self.goal_radius:
                rospy.loginfo(f"目标点到达成功！累计奖励: {self.episode_reward:.2f}")
                # 重置轨迹监控参数
                self.wrong_direction_count = 0
                self.prev_goal_dists = []
                
                # 发送停止命令
                stop_action = Int32()
                stop_action.data = 99  # 特殊动作代码，表示停止
                self.action_pub.publish(stop_action)
                rospy.loginfo("已发送停止命令")
            
            # 碰撞后的重置逻辑
            if done and np.min(self.current_state.cpu().numpy()[:8]) < 0.1:  # 检测到碰撞
                rospy.loginfo(f"检测到碰撞，重置机器人位置。累计奖励: {self.episode_reward:.2f}")
                # 发布重置信号
                reset_msg = Int32()
                reset_msg.data = 1
                self.reset_pub.publish(reset_msg)
                
                # 重置状态和轨迹监控
                self.last_state = None
                self.current_state = None
                self.episode_reward = 0
                self.wrong_direction_count = 0
                self.prev_goal_dists = []
                return  # 避免继续处理
        
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
            state_np = state.cpu().numpy()
            probs = self.actor(state)
            value = self.critic(state)
            
            # 提取状态信息
            laser_data = state_np[:8]
            goal_dist = state_np[8] * 10.0  # 目标距离
            goal_angle = state_np[9] * np.pi  # 目标角度 (-1到1之间)
            
            # 早期训练或根据探索率进行启发式引导
            if self.step_count < 10000 or np.random.random() < self.current_exploration_rate:
                probs_np = probs.cpu().numpy()
                
                # 1. 安全检查：如果前方障碍物很近，调整概率
                if np.min(laser_data[:3]) < 0.5:  # 前方三个方向有障碍物
                    # 选择左转还是右转取决于哪边更开阔
                    if np.mean(laser_data[0:4]) > np.mean(laser_data[4:8]):
                        probs_np[1] = max(probs_np[1] * 2, 0.4)  # 增加左小转概率
                    else:
                        probs_np[2] = max(probs_np[2] * 2, 0.4)  # 增加右小转概率
                    probs_np[0] *= 0.1  # 大幅降低前进概率
                
                # 2. 根据目标角度调整转向概率 - 增强版
                if abs(goal_angle) < 0.3:  # 几乎正前方
                    if np.min(laser_data[:3]) > 0.7:  # 前方无障碍
                        probs_np[0] = 0.8  # 大幅增加前进概率
                        # 其他动作概率降低
                        probs_np[1:] *= 0.2
                        rospy.loginfo_throttle(2.0, "目标在正前方，优先直行")
                elif abs(goal_angle) < 0.8:  # 偏左/偏右方向
                    if goal_angle > 0:  # 目标在左侧
                        probs_np[1] = 0.7  # 大幅增加左小转概率
                        probs_np[0] = 0.2  # 保持一定前进概率
                        probs_np[2:] *= 0.1  # 其他动作概率大幅降低
                        rospy.loginfo_throttle(2.0, "目标在左前方，增加左转概率")
                    else:  # 目标在右侧
                        probs_np[2] = 0.7  # 大幅增加右小转概率
                        probs_np[0] = 0.2  # 保持一定前进概率
                        probs_np[1] = 0.05  # 左转概率降低
                        probs_np[3:] *= 0.1  # 其他动作概率大幅降低
                        rospy.loginfo_throttle(2.0, "目标在右前方，增加右转概率")
                else:  # 目标在较大角度偏离
                    if goal_angle > 0:  # 目标在左侧
                        probs_np[3] = 0.75  # 使用左大转
                        probs_np[1] = 0.2  # 辅助使用左小转
                        probs_np[0] = 0.05  # 几乎不前进
                        probs_np[2] = 0.0  # 禁止右转
                        probs_np[4] = 0.0
                        rospy.loginfo_throttle(2.0, "目标在左侧大角度，强制左转")
                    else:  # 目标在右侧
                        probs_np[4] = 0.75  # 使用右大转
                        probs_np[2] = 0.2  # 辅助使用右小转
                        probs_np[0] = 0.05  # 几乎不前进
                        probs_np[1] = 0.0  # 禁止左转
                        probs_np[3] = 0.0
                        rospy.loginfo_throttle(2.0, "目标在右侧大角度，强制右转")
                
                # 3. 如果检测到持续偏离轨道，强制修正
                if self.wrong_direction_count > self.max_wrong_direction:
                    if goal_angle > 0:  # 需要向左转
                        probs_np = np.zeros_like(probs_np)
                        probs_np[3] = 0.9  # 强制左大转
                    else:  # 需要向右转
                        probs_np = np.zeros_like(probs_np)
                        probs_np[4] = 0.9  # 强制右大转
                    rospy.loginfo("检测到持续偏离，执行强制修正动作")
                    self.wrong_direction_count = 0
                
                # 归一化概率
                probs_np = probs_np / np.sum(probs_np)
                probs = torch.from_numpy(probs_np).to(device)
            
            # 更新探索率
            self.current_exploration_rate = max(
                self.current_exploration_rate * self.exploration_decay,
                self.min_exploration_rate
            )
            
            # 使用策略选择动作
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # 输出调试信息
            if self.step_count % 50 == 0:
                action_names = ["前进", "左小转", "右小转", "左大转", "右大转"]
                rospy.loginfo(f"选择动作: {action_names[action.item()]}, 目标距离: {goal_dist:.2f}m, 目标角度: {goal_angle:.2f}rad")
                rospy.loginfo(f"探索率: {self.current_exploration_rate:.3f}, 前方障碍: {np.min(laser_data[:3]):.2f}m")
                rospy.loginfo(f"动作概率: {probs.cpu().numpy()}")
            
        return action.item(), log_prob, value
    
    def calculate_reward(self, state, next_state, action):
        """计算奖励"""
        # 从状态中提取相关信息
        state_np = state.cpu().numpy()
        next_state_np = next_state.cpu().numpy()
        
        laser_features = state_np[:8]
        goal_dist = state_np[8] * 10.0  # 还原原始距离
        goal_angle = state_np[9] * np.pi  # 目标角度
        
        next_laser_features = next_state_np[:8]
        next_goal_dist = next_state_np[8] * 10.0
        next_goal_angle = next_state_np[9] * np.pi
        
        # 计算各项奖励
        # 1. 接近目标奖励 (增强系数)
        progress_reward = (goal_dist - next_goal_dist) * 15.0
        
        # 2. 碰撞惩罚 (增强惩罚)
        obstacle_penalty = -3.0 * np.exp(-3 * np.min(next_laser_features))
        
        # 3. 接近障碍物惩罚
        obstacle_trend_penalty = -1.5 if np.min(next_laser_features) < np.min(laser_features) else 0.0
        
        # 4. 目标达成奖励
        goal_reward = 50.0 if next_goal_dist < self.goal_radius else 0.0
        
        # 5. 平滑性奖励 (鼓励适当时使用直行)
        smoothness_reward = 0.3 if action == 0 and np.min(next_laser_features) > 0.5 else 0.0
        
        # 6. 新增：朝向目标奖励 (重要)
        angle_diff = abs(next_goal_angle) - abs(goal_angle)
        direction_reward = 6.0 if angle_diff < -0.1 else -3.5 if angle_diff > 0.1 else 0.0
        
        # 7. 新增：保持良好朝向奖励
        heading_reward = 2.0 if abs(next_goal_angle) < 0.3 and action == 0 else 0.0
        
        # 更新历史目标距离
        self.prev_goal_dists.append(next_goal_dist)
        if len(self.prev_goal_dists) > self.max_history_length:
            self.prev_goal_dists.pop(0)
            
        # 检查是否持续偏离目标
        if len(self.prev_goal_dists) >= 5:
            # 计算最近一段距离的变化趋势
            recent_progress = self.prev_goal_dists[0] - self.prev_goal_dists[-1]
            if recent_progress < -0.2:  # 距离在增加，说明偏离
                self.wrong_direction_count += 1
                direction_reward -= 3.0  # 额外惩罚
                rospy.logwarn_throttle(2.0, f"检测到偏离目标，错误计数: {self.wrong_direction_count}")
            else:
                self.wrong_direction_count = max(0, self.wrong_direction_count - 1)
        
        # 总奖励
        total_reward = progress_reward + obstacle_penalty + obstacle_trend_penalty + goal_reward + smoothness_reward + direction_reward + heading_reward
        
        # 日志记录
        if self.step_count % 50 == 0:
            rospy.loginfo(f"奖励详情 - 进度: {progress_reward:.2f}, 方向: {direction_reward:.2f}, 障碍: {obstacle_penalty:.2f}, 目标: {goal_reward:.2f}")
        
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
        actor_losses = []
        critic_losses = []
        entropies = []
        ratios = []
        
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
            
            # 记录本轮训练信息
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            ratios.append(ratio.mean().item())
            
        # 计算平均训练指标
        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        avg_critic_loss = sum(critic_losses) / len(critic_losses)
        avg_entropy = sum(entropies) / len(entropies)
        avg_ratio = sum(ratios) / len(ratios)
        
        # 获取样本动作分布
        with torch.no_grad():
            sample_probs = self.actor(states[:5])
            
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # 记录详细训练信息
        rospy.loginfo(f"策略更新 - Episode: {self.episode_count}, Step: {self.step_count}, Reward: {self.episode_reward:.2f}")
        rospy.loginfo(f"训练详情 - Actor损失: {avg_actor_loss:.4f}, Critic损失: {avg_critic_loss:.4f}, 熵: {avg_entropy:.4f}")
        rospy.loginfo(f"训练统计 - 策略比率: {avg_ratio:.4f}, 优势均值: {advantages.mean().item():.4f}, 优势标准差: {advantages.std().item():.4f}")
        rospy.loginfo(f"学习率: {self.lr}, 样本动作概率: {sample_probs.cpu().numpy().mean(axis=0)}")
        
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