# PPO室内导航系统

基于PPO强化学习算法的机器人室内导航系统，可以自动选择合适的技能到达目标点。

## 功能特点

1. 基于PPO（近端策略优化）算法实现技能选择
2. 使用激光雷达、里程计和目标点信息作为状态输入
3. 自动学习选择最佳导航动作（直行、左转、右转等）
4. 支持避障和平滑导航
5. 训练后的模型可以保存和加载

## 系统依赖

- Ubuntu 20.04 + ROS Noetic 或 Ubuntu 18.04 + ROS Melodic
- Python 3.6+
- PyTorch 1.7+
- Numpy, TF等

## 安装方法

### Ubuntu 20.04 (ROS Noetic)

1. 确保已安装ROS和Python依赖:

```bash
sudo apt update
sudo apt install python3-pip python3-catkin-tools
pip3 install torch numpy
```

2. 克隆代码到ROS工作空间:

```bash
cd ~/catkin_ws/src
git clone https://your-repo-url/ppo_indoor_navigation.git
```

3. 编译工作空间:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### Ubuntu 18.04 (ROS Melodic)

ROS Melodic默认使用Python 2.7，而我们的代码使用Python 3特性，需要进行以下修改：

1. 安装Python 3和必要的依赖:

```bash
sudo apt update
sudo apt install python3-pip python3-catkin-tools python3-dev
sudo apt install ros-melodic-tf2-geometry-msgs ros-melodic-catkin python3-catkin-pkg-modules
pip3 install torch==1.7.1 numpy rospkg catkin_pkg empy PyYAML
```

2. 确保ROS Melodic可以使用Python 3:

```bash
sudo apt install python3-catkin-tools python3-dev python3-numpy
```

3. 修改所有Python脚本头部:

```bash
# 脚本已经使用 #!/usr/bin/env python3，无需修改
# 确保脚本有执行权限
chmod +x ~/catkin_ws/src/ppo_indoor_navigation/scripts/*.py
chmod +x ~/catkin_ws/src/ppo_indoor_navigation/run_ppo_nav.sh
```

4. 如果您的Python版本低于3.6，需要转换f-string:

```bash
# 使用我们提供的转换工具将f-string转换为format()方法
cd ~/catkin_ws/src/ppo_indoor_navigation
./scripts/convert_fstrings.py scripts/
```

5. 编译工作空间时指定Python 3:

```bash
cd ~/catkin_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

6. 使用兼容性检查脚本验证环境:

```bash
rosrun ppo_indoor_navigation check_melodic_compat.py
```

## 使用方法

### 启动系统

直接运行脚本:

```bash
cd ~/catkin_ws/src/ppo_indoor_navigation
./run_ppo_nav.sh
```

或者手动启动:

```bash
roslaunch ppo_indoor_navigation ppo_skill_nav.launch
```

### 发送目标点

```bash
rosrun ppo_indoor_navigation send_goal.py 2.0 3.0
```

其中2.0和3.0分别是目标点的X和Y坐标。

## 系统结构

- `scripts/ppo_skill_selector.py`: PPO技能选择器主节点
- `scripts/send_goal.py`: 发送目标点的工具脚本
- `scripts/check_melodic_compat.py`: ROS Melodic环境兼容性检查脚本
- `scripts/convert_fstrings.py`: f-string转换工具（用于Python 3.6以下版本）
- `launch/ppo_skill_nav.launch`: 启动文件
- `model/`: 存储训练好的模型

## 技能定义

系统定义了5种基本导航技能:

1. 直行 (0.3, 0.0)
2. 小左转 (0.2, 0.3)
3. 小右转 (0.2, -0.3)
4. 大左转 (0.15, 0.6)
5. 大右转 (0.15, -0.6)

## 状态空间

- 激光雷达数据(降维为8个方向)
- 目标点相对距离和角度
- 机器人当前朝向

## 奖励设计

- 接近目标奖励
- 碰撞惩罚
- 接近障碍物惩罚
- 目标达成奖励
- 平滑性奖励

## 自定义配置

可以在`ppo_skill_selector.py`中修改PPO参数、技能定义和奖励函数等。

## Ubuntu 18.04 (ROS Melodic) 常见问题与解决方案

1. **ModuleNotFoundError: No module named 'torch'**:
   - 确保使用`pip3`安装了PyTorch: `pip3 install torch==1.7.1`
   - 确保使用Python 3运行脚本

2. **找不到tf模块**:
   - 安装Python 3版本的tf模块: `pip3 install transforms3d`
   - 确保已安装ROS相关包: `sudo apt install ros-melodic-tf ros-melodic-tf2-ros`

3. **启动脚本时/usr/bin/env: 'python3\r': No such file or directory**:
   - 这是由于Windows行尾符导致的问题，运行: `sed -i 's/\r$//' ~/catkin_ws/src/ppo_indoor_navigation/scripts/*.py`
   - 然后重新赋予执行权限: `chmod +x ~/catkin_ws/src/ppo_indoor_navigation/scripts/*.py`

4. **CUDA不可用或版本不兼容**:
   - 对于非CUDA环境，代码已经自动处理，使用CPU模式
   - 如果需要CUDA支持，安装与PyTorch 1.7.1兼容的CUDA版本(通常为CUDA 10.2或11.0)
   - 安装命令: `pip3 install torch==1.7.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html`

5. **cv_bridge兼容性问题**:
   - 如果使用了cv_bridge，可能需要为Python 3重新编译: 
   ```bash
   cd ~/catkin_ws/src
   git clone -b melodic https://github.com/ros-perception/vision_opencv.git
   cd ~/catkin_ws
   catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
   ```

6. **Python版本过低不支持f-string**:
   - 如果您的Python版本低于3.6，可以使用我们提供的转换工具:
   ```bash
   cd ~/catkin_ws/src/ppo_indoor_navigation
   ./scripts/convert_fstrings.py scripts/
   ``` 