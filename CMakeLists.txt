cmake_minimum_required(VERSION 3.0.2)
project(ppo_indoor_navigation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  gazebo_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)