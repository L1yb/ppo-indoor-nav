# PPO 室内导航系统

基于 PPO 强化学习算法的机器人室内导航系统，可以自动选择合适的技能到达目标点。

## 功能特点

1. 基于 PPO（近端策略优化）算法实现技能选择
2. 使用激光雷达、里程计和目标点信息作为状态输入
3. 自动学习选择最佳导航动作（直行、左转、右转等）
   ~~4. 支持避障和平滑导航~~
4. 训练后的模型可以保存和加载

## 系统依赖

- Ubuntu 18.04 + ROS Melodic
- Python 3.8
- PyTorch 2.1.1

## 使用方法

先启动松灵小车

```bash
# 首次启动运行一下初始化
rosrun scout_bringup setup_can2usb.bash

roslaunch scout_bringup scout_minimal.launch
# 启动雷达
sudo chmod 777 /dev/ttyUSB0
roslaunch rplidar_ros single_lidar.launch

```

### 启动系统

no 直接运行脚本:

```bash
cd ~/catkin_ws/src/ppo_indoor_navigation
./run_ppo_nav.sh
```

no 或者手动启动:

```bash
roslaunch ppo_indoor_navigation ppo_skill_nav.launch
```

yes 启动步骤

```bash
roslaunch ppo_indoor_navigation ppo_navigation.launch start_ppo:=false

# 进入conda环境之后
export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/b/python2.7/dist-packages

python3 scripts/ppo_skill_selector_py3.py
```

### 发送目标点

```bash
rosrun ppo_indoor_navigation send_goal.py 2.0 3.0
```

其中 2.0 和 3.0 分别是目标点的 X 和 Y 坐标。

## 系统结构

- `scripts/ppo_skill_selector_py3.py`: PPO 技能选择器主节点
- `scripts/send_goal.py`: 发送目标点的工具脚本
- `ros_ppo_bridge.py`: ros 和 ppo 的桥梁，ppo 运行环境是 3.8，ros 环境是 2.7，他们之间通过话题进行通信
- `launch/ppo_skill_nav.launch`: 启动文件
- `model/`: 存储训练好的模型

## 技能定义

系统定义了 5 种基本导航技能:

1. 直行 (0.3, 0.0)
2. 小左转 (0.2, 0.3)
3. 小右转 (0.2, -0.3)
4. 大左转 (0.15, 0.6)
5. 大右转 (0.15, -0.6)

## 状态空间

- 激光雷达数据(降维为 8 个方向)
- 目标点相对距离和角度
- 机器人当前朝向

## 奖励设计

- 接近目标奖励
- 碰撞惩罚
- 接近障碍物惩罚
- 目标达成奖励
- 平滑性奖励
