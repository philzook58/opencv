cmake_minimum_required(VERSION 2.8)
project(aruco_testproject)
SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )
find_package(aruco REQUIRED )
find_package( OpenCV REQUIRED )
# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(aruco_simple aruco_opencv.cpp)

target_link_libraries(aruco_simple  ${aruco_LIBS} ${OpenCV_LIBS})
