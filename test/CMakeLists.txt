cmake_minimum_required(VERSION 3.10)

set(PARENT_PROJECT_NAME ${PROJECT_NAME})

project(test LANGUAGES CXX CUDA)
enable_language(CUDA)
find_package(CUDAToolkit)

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(TEST_GRID_SAMPLE test_grid_sample)

add_executable(${TEST_GRID_SAMPLE} test.cpp)

target_include_directories(${TEST_GRID_SAMPLE} PUBLIC ${PROJECT_INCLUDE_DIR} ${CUDA_ROOT}/include)
target_link_libraries(${TEST_GRID_SAMPLE} ${PARENT_PROJECT_NAME} CUDA::cudart)

set_target_properties(${TEST_GRID_SAMPLE} PROPERTIES CUDA_ARCHITECTURES "89")