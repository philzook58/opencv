cmake_minimum_required(VERSION 2.8)


project(myopencv_sample)


find_package(OpenCV REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})


set(the_target "myopencv_sample")


add_executable(${the_target} main.cpp) # add other .cpp

# and .h files here

target_link_libraries(${the_target} ${OpenCV_LIBS})