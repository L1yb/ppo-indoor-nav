#!/bin/bash
# 启动PPO导航系统

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

# 检测ROS版本
ROS_VERSION=$(rosversion -d)
echo "检测到ROS版本: $ROS_VERSION"

# 输出帮助信息
echo "===== PPO技能选择导航系统 ====="
echo "此脚本将启动PPO技能选择器节点"
echo "确保你的ROS环境已正确设置，小车底盘和传感器已启动"

# 设置ROS工作空间
source ~/catkin_ws/devel/setup.bash

# 如果是Melodic，确保使用Python 3
if [ "$ROS_VERSION" = "melodic" ]; then
    echo "ROS Melodic环境，强制使用Python 3..."
    export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
    # 检查是否安装了必要的Python包
    python3 -c "import torch" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "错误: PyTorch未安装或未对Python 3可用"
        echo "请运行: pip3 install torch==1.7.1"
        echo "您可以参考README中的Ubuntu 18.04 (ROS Melodic)安装指南"
        exit 1
    fi
fi

# 启动PPO导航节点
echo "正在启动PPO技能选择器..."
roslaunch ppo_indoor_navigation ppo_skill_nav.launch

# 使用方法提示
echo ""
echo "===== 使用方法 ====="
echo "在新终端中运行以下命令发送目标点："
echo "rosrun ppo_indoor_navigation send_goal.py X坐标 Y坐标"
echo "例如: rosrun ppo_indoor_navigation send_goal.py 2.0 3.0" 