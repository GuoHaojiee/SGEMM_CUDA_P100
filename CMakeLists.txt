cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# 确保 CUDA 可用
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
# P100 使用 sm_60（Pascal 架构）
set(CUDA_COMPUTE_CAPABILITY 60)

# Debug 模式下启用设备代码调试符号（会大幅降低性能）
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# 头文件搜索路径
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

# 源文件目录
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# 只编译 sgemm 主程序（内核 1~6、9、10）
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
