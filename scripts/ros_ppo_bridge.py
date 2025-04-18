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
        
        # 订阅重置信号
        self.reset_sub = rospy.Subscriber('/ppo/reset', Int32, self.reset_callback)
        
        # 发布控制命令到机器人
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # 状态变量
        self.laser_data = None
        self.robot_pos = None
        self.robot_yaw = None
        self.goal_pos = None
        
        # 初始位置和目标位置（用于重置）
        self.initial_pos = None
        self.initial_yaw = None
        
        # 技能定义
        self.skills = [
            (0.3, 0.0),     # 直行
            (0.2, 0.4),     # 小左转 - 角速度从0.3增加到0.4
            (0.2, -0.4),    # 小右转 - 角速度从-0.3增加到-0.4
            (0.15, 0.8),    # 大左转 - 角速度从0.6增加到0.8
            (0.15, -0.8)    # 大右转 - 角速度从-0.6增加到-0.8
        ]
        
        # 主循环
        rospy.Timer(rospy.Duration(0.1), self.main_loop)
        rospy.loginfo("ROS-PPO桥接节点已启动")
    
    def laser_callback(self, msg):
        """处理激光雷达数据"""
        # print(msg.ranges)
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
        
        # 保存初始位置（如果尚未保存）
        if self.initial_pos is None:
            self.initial_pos = self.robot_pos.copy()
            self.initial_yaw = self.robot_yaw
            rospy.loginfo("保存初始位置: ({:.2f}, {:.2f}), 朝向: {:.2f}".format(
                self.initial_pos[0], self.initial_pos[1], self.initial_yaw))
    
    def goal_callback(self, msg):
        """处理目标点数据"""
        self.goal_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        rospy.loginfo("收到新目标点: ({:.2f}, {:.2f})".format(self.goal_pos[0], self.goal_pos[1]))
    
    def action_callback(self, msg):
        """接收PPO算法选择的动作并执行"""
        action_idx = msg.data
        
        # 处理停止命令
        if action_idx == 99:  # 特殊动作代码，表示停止
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            # 多次发送停止命令确保小车停止
            for i in range(3):
                self.cmd_vel_pub.publish(cmd)
                rospy.sleep(0.1)
            rospy.loginfo("小车已停止")
            return
            
        if 0 <= action_idx < len(self.skills):
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = self.skills[action_idx]
            self.cmd_vel_pub.publish(cmd)
    
    def reset_callback(self, msg):
        """处理重置信号"""
        if msg.data == 1:
            self.reset_robot_position()
    
    def reset_robot_position(self):
        """重置机器人位置"""
        rospy.loginfo("收到重置信号，重置机器人位置")
        
        # 记录重置前的位置信息
        rospy.loginfo("重置前位置: ({:.2f}, {:.2f})".format(
            self.robot_pos[0], self.robot_pos[1]))
        
        # 尝试多种方法重置机器人位置
        reset_success = False
        
        # 方法1: 使用Gazebo服务重置模型位置
        try:
            # 导入Gazebo相关消息
            from gazebo_msgs.msg import ModelState
            from gazebo_msgs.srv import SetModelState
            
            # 使用确定的模型名称
            robot_model_name = 'mycar'  # 根据launch文件中的指定名称
            rospy.loginfo("使用机器人模型名称: {}".format(robot_model_name))
            
            # 设置模型状态
            model_state = ModelState()
            model_state.model_name = robot_model_name
            model_state.pose.position.x = self.initial_pos[0]
            model_state.pose.position.y = self.initial_pos[1]
            model_state.pose.position.z = 0.0
            
            # 从yaw角度计算四元数
            from tf.transformations import quaternion_from_euler
            quat = quaternion_from_euler(0, 0, self.initial_yaw)
            model_state.pose.orientation.x = quat[0]
            model_state.pose.orientation.y = quat[1]
            model_state.pose.orientation.z = quat[2]
            model_state.pose.orientation.w = quat[3]
            
            # 重置速度
            model_state.twist.linear.x = 0
            model_state.twist.linear.y = 0
            model_state.twist.linear.z = 0
            model_state.twist.angular.x = 0
            model_state.twist.angular.y = 0
            model_state.twist.angular.z = 0
            
            # 设置参考框架
            model_state.reference_frame = 'world'
            
            # 调用服务设置模型状态
            rospy.wait_for_service('/gazebo/set_model_state', timeout=2.0)
            set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_model_state(model_state)
            
            if resp.success:
                rospy.loginfo("通过Gazebo服务成功重置机器人位置: ({:.2f}, {:.2f})".format(
                    self.initial_pos[0], self.initial_pos[1]))
                reset_success = True
            else:
                rospy.logwarn("Gazebo重置服务调用失败: {}".format(resp.status_message))
        
        except Exception as e:
            rospy.logwarn("方法1失败: {}".format(e))
        
        # 方法2: 发布ROS话题进行重置
        if not reset_success:
            try:
                # 尝试发布到/gazebo/set_model_state话题
                from gazebo_msgs.msg import ModelState
                model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
                
                model_state = ModelState()
                model_state.model_name = 'mycar'  # 使用launch文件中指定的名称
                model_state.pose.position.x = self.initial_pos[0]
                model_state.pose.position.y = self.initial_pos[1]
                model_state.pose.position.z = 0.0
                
                # 设置朝向
                from tf.transformations import quaternion_from_euler
                quat = quaternion_from_euler(0, 0, self.initial_yaw)
                model_state.pose.orientation.x = quat[0]
                model_state.pose.orientation.y = quat[1]
                model_state.pose.orientation.z = quat[2]
                model_state.pose.orientation.w = quat[3]
                
                # 发布多次以确保消息传递
                for i in range(3):
                    model_state_pub.publish(model_state)
                    rospy.sleep(0.1)
                
                rospy.loginfo("通过发布ModelState话题重置机器人位置")
                reset_success = True
            except Exception as e:
                rospy.logwarn("方法2失败: {}".format(e))
        
        # 方法3: 通过/initialpose重置AMCL定位
        if not reset_success:
            try:
                # 发布到/initialpose话题 (用于AMCL定位重置)
                from geometry_msgs.msg import PoseWithCovarianceStamped
                
                init_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)
                
                init_pose = PoseWithCovarianceStamped()
                init_pose.header.stamp = rospy.Time.now()
                init_pose.header.frame_id = "map"
                
                init_pose.pose.pose.position.x = self.initial_pos[0]
                init_pose.pose.pose.position.y = self.initial_pos[1]
                
                from tf.transformations import quaternion_from_euler
                quat = quaternion_from_euler(0, 0, self.initial_yaw)
                init_pose.pose.pose.orientation.x = quat[0]
                init_pose.pose.pose.orientation.y = quat[1]
                init_pose.pose.pose.orientation.z = quat[2]
                init_pose.pose.pose.orientation.w = quat[3]
                
                # 设置协方差矩阵 (对角线元素)
                cov = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
                init_pose.pose.covariance = cov
                
                # 发布初始位姿
                init_pose_pub.publish(init_pose)
                rospy.sleep(0.5)  # 给一点时间让消息传递
                
                rospy.loginfo("通过/initialpose话题重置机器人位置")
                reset_success = True
                
            except Exception as e:
                rospy.logwarn("方法3失败: {}".format(e))
        
        # 最后手段: 通过发送空速度停止机器人
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        # 如果所有方法都失败，发出停止命令并通知
        if not reset_success:
            rospy.logwarn("所有重置方法失败，已停止机器人。可能需要手动重置机器人位置。")
            
            # 多次发送停止命令确保机器人停止
            for i in range(5):
                self.cmd_vel_pub.publish(cmd)
                rospy.sleep(0.1)
        
        return reset_success
    
    def get_state(self):
        """构建状态表示并发布"""
        if self.laser_data is None or self.robot_pos is None or self.goal_pos is None:
            if(self.laser_data is None):
                print("laser is none")
            if(self.robot_pos is None):
                print("pos is none")
            if(self.goal_pos is None):
                print("goal is none")
            # print("state is none")
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
        # print(state_msg)
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