#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PoseStamped
import sys

def send_goal():
    rospy.init_node('goal_sender', anonymous=True)
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
    
    # 等待publisher初始化
    rospy.sleep(1)
    
    # 创建目标点
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = rospy.Time.now()
    
    # 命令行参数解析
    if len(sys.argv) >= 3:
        goal.pose.position.x = float(sys.argv[1])
        goal.pose.position.y = float(sys.argv[2])
    else:
        # 默认目标点
        goal.pose.position.x = 2.0
        goal.pose.position.y = 2.0
    
    # 朝向（四元数，这里设置朝向为沿x轴正方向）
    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = 0.0
    goal.pose.orientation.w = 1.0
    
    # 发布目标点
    pub.publish(goal)
    rospy.loginfo(f"已发送目标点: ({goal.pose.position.x}, {goal.pose.position.y})")

if __name__ == '__main__':
    try:
        send_goal()
    except rospy.ROSInterruptException:
        pass 