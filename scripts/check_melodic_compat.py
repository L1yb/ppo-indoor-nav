#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS Melodic兼容性检查脚本
用于检查在Ubuntu 18.04 / ROS Melodic环境下运行PPO导航系统所需的依赖
"""

import sys
import os
import importlib
import subprocess
import platform

def print_status(message, status, details=None):
    """打印状态信息"""
    if status:
        print(f"[✓] {message}")
    else:
        print(f"[✗] {message}")
        if details:
            print(f"    - {details}")

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 6
    print_status(
        f"Python版本: {version.major}.{version.minor}.{version.micro}",
        is_valid,
        None if is_valid else "需要Python 3.6或更高版本"
    )
    return is_valid

def check_module(module_name, min_version=None):
    """检查Python模块是否已安装"""
    try:
        module = importlib.import_module(module_name)
        if min_version and hasattr(module, '__version__'):
            version = module.__version__
            is_valid = version >= min_version
            print_status(
                f"{module_name}版本: {version}",
                is_valid,
                None if is_valid else f"需要{module_name}>={min_version}"
            )
            return is_valid
        else:
            print_status(f"{module_name}已安装", True)
            return True
    except ImportError as e:
        print_status(f"{module_name}未安装", False, str(e))
        return False

def check_ros_version():
    """检查ROS版本"""
    try:
        proc = subprocess.Popen(['rosversion', '-d'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode == 0:
            ros_version = stdout.decode('utf-8').strip()
            is_melodic = ros_version == 'melodic'
            print_status(f"ROS版本: {ros_version}", True)
            if not is_melodic:
                print("    - 注意: 此检查脚本专为ROS Melodic设计，但您的环境也可能兼容")
            return True
        else:
            print_status("无法检测ROS版本", False, stderr.decode('utf-8').strip())
            return False
    except Exception as e:
        print_status("ROS环境异常", False, str(e))
        return False

def check_ubuntu_version():
    """检查Ubuntu版本"""
    try:
        if platform.system() != 'Linux':
            print_status(f"操作系统: {platform.system()}", False, "需要Linux系统")
            return False
            
        # 检查Ubuntu版本
        with open('/etc/os-release', 'r') as f:
            lines = f.readlines()
            version_info = {}
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    version_info[key] = value.strip().replace('"', '')
            
            if 'NAME' in version_info and 'Ubuntu' in version_info['NAME']:
                if 'VERSION_ID' in version_info:
                    version = version_info['VERSION_ID']
                    is_valid = version == '18.04'
                    print_status(f"Ubuntu版本: {version}", is_valid, None if is_valid else "推荐版本18.04")
                    return is_valid
                else:
                    print_status("无法检测Ubuntu版本", False)
                    return False
            else:
                print_status(f"操作系统: {version_info.get('NAME', '未知')}", False, "推荐使用Ubuntu 18.04")
                return False
    except Exception as e:
        print_status("系统版本检测失败", False, str(e))
        return False

def check_torch_cuda():
    """检查PyTorch CUDA支持"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        print_status(
            f"PyTorch CUDA支持: {'可用' if cuda_available else '不可用'}, 设备数: {device_count}",
            True
        )
        if not cuda_available:
            print("    - 注意: 系统将使用CPU模式运行，这可能会影响性能")
        return True
    except Exception as e:
        print_status("PyTorch CUDA检测失败", False, str(e))
        return False

def print_recommendations():
    """打印建议"""
    print("\n推荐安装命令:")
    print("1. 安装Python 3依赖:")
    print("   sudo apt update")
    print("   sudo apt install python3-pip python3-catkin-tools python3-dev")
    print("   sudo apt install ros-melodic-tf2-geometry-msgs ros-melodic-catkin python3-catkin-pkg-modules")
    print("\n2. 安装PyTorch:")
    print("   pip3 install torch==1.7.1 numpy")
    print("\n3. 安装ROS Python 3依赖:")
    print("   pip3 install rospkg catkin_pkg empy PyYAML")
    print("\n4. 编译工作空间:")
    print("   cd ~/catkin_ws")
    print("   catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3")
    print("   source devel/setup.bash")

def main():
    """主函数"""
    print("====== PPO导航系统 - ROS Melodic兼容性检查 ======\n")
    
    # 运行所有检查
    python_ok = check_python_version()
    ubuntu_ok = check_ubuntu_version()
    ros_ok = check_ros_version()
    
    print("\n--- 依赖检查 ---")
    torch_ok = check_module('torch', '1.7.0')
    numpy_ok = check_module('numpy')
    tf_ok = check_module('tf')
    rospy_ok = check_module('rospy')
    
    print("\n--- CUDA支持检查 ---")
    cuda_ok = check_torch_cuda() if torch_ok else False
    
    # 计算总体结果
    all_ok = python_ok and ubuntu_ok and ros_ok and torch_ok and numpy_ok and tf_ok and rospy_ok
    
    print("\n====== 检查结果 ======")
    if all_ok:
        print("环境配置正确，可以运行PPO导航系统。")
        if not cuda_ok:
            print("注意: 系统将使用CPU模式运行，这可能会影响性能。")
    else:
        print("环境配置不完整，请解决上述问题后再运行系统。")
        print_recommendations()
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main()) 