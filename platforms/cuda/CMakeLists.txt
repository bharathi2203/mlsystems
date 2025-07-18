cmake_minimum_required(VERSION 3.18)
project(SystemsPortfolioCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3")
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90") # Support RTX 20xx to H100

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../kernels)

# Common utilities
add_library(cuda_utils STATIC
    common/cuda_utils.cu
    common/benchmark.cu
    common/memory_utils.cu
)

# Function to add kernel executables
function(add_kernel_executable kernel_name kernel_path)
    add_executable(${kernel_name} ${kernel_path})
    target_link_libraries(${kernel_name} cuda_utils CUDA::cudart CUDA::cublas)
    set_target_properties(${kernel_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# Kernel executables
file(GLOB_RECURSE KERNEL_SOURCES "../../kernels/**/*.cu")
foreach(KERNEL_SOURCE ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SOURCE} NAME_WE)
    add_kernel_executable(${KERNEL_NAME} ${KERNEL_SOURCE})
endforeach()

# Test suite
add_executable(test_kernels test/test_runner.cu)
target_link_libraries(test_kernels cuda_utils CUDA::cudart)

# Benchmark suite
add_executable(benchmark_kernels benchmark/benchmark_runner.cu)
target_link_libraries(benchmark_kernels cuda_utils CUDA::cudart CUDA::cublas) 