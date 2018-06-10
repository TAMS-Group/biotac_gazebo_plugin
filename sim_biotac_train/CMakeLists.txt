cmake_minimum_required(VERSION 2.8.3)
project(sim_biotac_train)

set(ROS_BUILD_TYPE Release)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_srvs
    geometry_msgs
    sensor_msgs
    eigen_conversions
    apriltags
    rosbag
    kdl_conversions
    sr_robot_msgs
    sim_biotac
)

find_package(Eigen3 REQUIRED)

include_directories(include
    ${Boost_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${sim_biotac_INCLUDE_DIRS}
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

catkin_package(
    INCLUDE_DIRS

    CATKIN_DEPENDS
    roscpp
    rospy
    std_srvs
    geometry_msgs
    sensor_msgs
    eigen_conversions
    apriltags
    rosbag
    kdl_conversions
    sr_robot_msgs
    sim_biotac

    DEPENDS
    EIGEN3
)

add_executable(sim_biotac_train_adjust src/sim_biotac_train_adjust.cpp)
add_dependencies(sim_biotac_train_adjust ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(sim_biotac_train_adjust ${catkin_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

add_executable(sim_biotac_train_extract src/sim_biotac_train_extract.cpp)
add_dependencies(sim_biotac_train_extract ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(sim_biotac_train_extract ${catkin_LIBRARIES} yaml-cpp)
