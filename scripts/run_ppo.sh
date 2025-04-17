#!/bin/bash

# 找到conda的环境
# source ~/miniconda3/etc/profile.d/conda.sh
source /home/lyb/anaconda3/condabin/conda

# 激活Python 3.8环境
conda activate scene  # 请替换为您实际的环境名称

# 设置ROS路径
export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/lib/python2.7/dist-packages

# 启动PPO节点
python3 ~/catkin_ws/src/ppo_indoor_navigation/scripts/ppo_skill_selector_py3.py 