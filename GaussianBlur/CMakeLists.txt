cmake_minimum_required(VERSION 2.8)

# project name
project(Convolute)

#find the CUDA libraries
find_package(CUDA REQUIRED)

#Specify the CUDA include directories for the project
include_directories(${CUDA_INCLUDE_DIRS})

#create an executable from the given CUDA files
cuda_add_executable(Convolute main.cu)

#specify any additional libraries here(CUFFT and CUBLAS can be useful)
target_link_libraries(Convolute)
