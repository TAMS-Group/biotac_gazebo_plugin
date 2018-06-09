cmake_minimum_required(VERSION 2.8.3)
project(sim_biotac_motorhand)

set(ROS_BUILD_TYPE Release)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  controller_manager_msgs
  moveit_ros_planning_interface
  sr_robot_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(three_pod_grasp src/three_pod_grasp.cpp)
add_dependencies(three_pod_grasp ${catkin_EXPORTED_TARGETS})
target_link_libraries(three_pod_grasp ${catkin_LIBRARIES})
