#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32
from tf.transformations import euler_from_quaternion

class ROSPPOBridge:
    def __init__(self):
        rospy.init_node('ros_ppo_bridge')
        
        # 订阅机器人传感器数据
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        # 发布传感器数据到PPO节点
        self.state_pub = rospy.Publisher('/ppo/state', Float32MultiArray, queue_size=10)
        
        # 订阅PPO决策
        self.action_sub = rospy.Subscriber('/ppo/action', Int32, self.action_callback)
        
        # 发布控制命令到机器人
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 状态变量
        self.laser_data = None
        self.robot_pos = None
        self.robot_yaw = None
        self.goal_pos = None
        
        # 技能定义
        self.skills = [
            (0.3, 0.0),     # 直行
            (0.2, 0.3),     # 小左转
            (0.2, -0.3),    # 小右转
            (0.15, 0.6),    # 大左转
            (0.15, -0.6)    # 大右转
        ]
        
        # 主循环
        rospy.Timer(rospy.Duration(0.1), self.main_loop)
        rospy.loginfo("ROS-PPO桥接节点已启动")
    
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
        rospy.loginfo("收到新目标点: ({:.2f}, {:.2f})".format(self.goal_pos[0], self.goal_pos[1]))
    
    def action_callback(self, msg):
        """接收PPO算法选择的动作并执行"""
        action_idx = msg.data
        if 0 <= action_idx < len(self.skills):
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = self.skills[action_idx]
            self.cmd_vel_pub.publish(cmd)
    
    def get_state(self):
        """构建状态表示并发布"""
        if self.laser_data is None or self.robot_pos is None or self.goal_pos is None:
            return
            
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
        
        # 发布状态
        state_msg = Float32MultiArray()
        state_msg.data = state.tolist()
        self.state_pub.publish(state_msg)
    
    def main_loop(self, event):
        """主循环，定期发布状态"""
        self.get_state()

if __name__ == '__main__':
    try:
        bridge = ROSPPOBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 