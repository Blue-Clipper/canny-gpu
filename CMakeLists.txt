cmake_minimum_required(VERSION 3.10)
project(canny-gpu)

find_package(CUDA QUIET REQUIRED)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(BUUILD_TYPE Debug)


cuda_add_executable(
    canny-gpu
    main.cu
    canny_kernel.cu
    canny_kernel.cuh
    io.h 
    io.cpp
)
target_link_libraries( canny-gpu ${OpenCV_LIBS} )    

