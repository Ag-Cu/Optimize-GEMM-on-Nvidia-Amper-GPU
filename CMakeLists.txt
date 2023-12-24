cmake_minimum_required(VERSION 3.10)

project(cuda_learn CXX C CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 \
-gencode=arch=compute_80,code=sm_80 \
-O3")

include_directories(./include)

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
set(SOURCES ${SRC_FILES})

add_executable(sgemm ${SOURCES})
find_library(CUBLAS_LIB cublas HINTS /usr/local/cuda/lib64)
target_link_libraries(sgemm ${CUBLAS_LIB})

