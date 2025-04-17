cmake_minimum_required(VERSION 3.0.2)
project(ppo_indoor_navigation)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
  tf
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)