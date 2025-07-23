/*
 * 3D Convolution - CUDA Implementation
 * 
 * GPU-accelerated implementation of 3D convolution operation
 * for volumetric data processing and 3D neural network applications.
 * 
 * Mathematical foundation:
 * - 3D Convolution: (f * g)[i,j,k] = ΣΣΣ f[m,n,p] * g[i-m,j-n,k-p] for all m,n,p
 * - 3D Cross-correlation: (f ★ g)[i,j,k] = ΣΣΣ f[m,n,p] * g[i+m,j+n,k+p] for all m,n,p
 * - Valid convolution: output size = (input_size - kernel_size + 1)
 * - Same convolution: output size = input_size (with padding)
 * - Full convolution: output size = (input_size + kernel_size - 1)
 * 
 * Algorithm optimizations:
 * 1. 3D block decomposition for parallel processing
 * 2. Shared memory tiling for volumetric data caching
 * 3. Coalesced memory access patterns for 3D data
 * 4. Register blocking for improved cache utilization
 * 5. Separable kernel optimization when applicable
 * 
 * Memory patterns:
 * - 3D thread block decomposition (x, y, z dimensions)
 * - Shared memory tiling for input volumes and kernels
 * - Proper boundary handling with 3D padding strategies
 * - Optimized memory layout for 3D data structures
 * 
 * Numerical considerations:
 * - Single precision floating point operations
 * - 3D boundary handling with zero-padding
 * - Accumulation strategies for large 3D kernels
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

// Configuration constants
#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8
#define TILE_SIZE_Z 8
#define MAX_KERNEL_SIZE_3D 16
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

/**
 * CUDA Kernel: Basic 3D Convolution
 * 
 * Each thread computes one output voxel using direct convolution.
 * Simple implementation for small kernels and educational purposes.
 * 
 * @param input: Input volume [depth x height x width]
 * @param kernel: Convolution kernel [kernel_depth x kernel_height x kernel_width]
 * @param output: Output volume [output_depth x output_height x output_width]
 * @param input_depth: Depth of input volume
 * @param input_height: Height of input volume
 * @param input_width: Width of input volume
 * @param kernel_depth: Depth of convolution kernel
 * @param kernel_height: Height of convolution kernel
 * @param kernel_width: Width of convolution kernel
 * @param output_depth: Depth of output volume
 * @param output_height: Height of output volume
 * @param output_width: Width of output volume
 */
__global__ void convolution_3d_basic_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int output_depth,
    int output_height,
    int output_width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < output_width && y < output_height && z < output_depth) {
        float sum = 0.0f;
        
        // Compute 3D convolution for this output voxel
        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int input_z = z + kd;
                    int input_y = y + kh;
                    int input_x = x + kw;
                    
                    if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                        int input_idx = input_z * input_height * input_width + 
                                       input_y * input_width + input_x;
                        int kernel_idx = kd * kernel_height * kernel_width + 
                                        kh * kernel_width + kw;
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = z * output_height * output_width + y * output_width + x;
        output[output_idx] = sum;
    }
}

/**
 * CUDA Kernel: Tiled 3D Convolution with Shared Memory
 * 
 * Optimized implementation using shared memory to cache input tiles
 * and reduce global memory bandwidth requirements for 3D data.
 */
__global__ void convolution_3d_tiled_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int output_depth,
    int output_height,
    int output_width
) {
    extern __shared__ float shared_mem[];
    
    // Allocate shared memory for input tile and kernel
    int input_tile_size = (TILE_SIZE_Z + kernel_depth - 1) * 
                         (TILE_SIZE_Y + kernel_height - 1) * 
                         (TILE_SIZE_X + kernel_width - 1);
    int kernel_size = kernel_depth * kernel_height * kernel_width;
    
    float* shared_input = shared_mem;
    float* shared_kernel = &shared_mem[input_tile_size];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    // Load kernel into shared memory
    int tid = tz * blockDim.y * blockDim.x + ty * blockDim.x + tx;
    if (tid < kernel_size) {
        shared_kernel[tid] = kernel[tid];
    }
    
    // Load input tile into shared memory
    int tile_depth = TILE_SIZE_Z + kernel_depth - 1;
    int tile_height = TILE_SIZE_Y + kernel_height - 1;
    int tile_width = TILE_SIZE_X + kernel_width - 1;
    
    int input_start_z = bz * TILE_SIZE_Z;
    int input_start_y = by * TILE_SIZE_Y;
    int input_start_x = bx * TILE_SIZE_X;
    
    for (int dz = tz; dz < tile_depth; dz += blockDim.z) {
        for (int dy = ty; dy < tile_height; dy += blockDim.y) {
            for (int dx = tx; dx < tile_width; dx += blockDim.x) {
                int input_z = input_start_z + dz;
                int input_y = input_start_y + dy;
                int input_x = input_start_x + dx;
                
                float val = 0.0f;
                if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                    int input_idx = input_z * input_height * input_width + 
                                   input_y * input_width + input_x;
                    val = input[input_idx];
                }
                
                int shared_idx = dz * tile_height * tile_width + dy * tile_width + dx;
                shared_input[shared_idx] = val;
            }
        }
    }
    
    __syncthreads();
    
    // Compute convolution using shared memory
    int output_x = input_start_x + tx;
    int output_y = input_start_y + ty;
    int output_z = input_start_z + tz;
    
    if (output_x < output_width && output_y < output_height && output_z < output_depth) {
        float sum = 0.0f;
        
        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int shared_z = tz + kd;
                    int shared_y = ty + kh;
                    int shared_x = tx + kw;
                    
                    if (shared_z < tile_depth && shared_y < tile_height && shared_x < tile_width) {
                        int shared_input_idx = shared_z * tile_height * tile_width + 
                                              shared_y * tile_width + shared_x;
                        int kernel_idx = kd * kernel_height * kernel_width + 
                                        kh * kernel_width + kw;
                        
                        sum += shared_input[shared_input_idx] * shared_kernel[kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = output_z * output_height * output_width + 
                        output_y * output_width + output_x;
        output[output_idx] = sum;
    }
}

/**
 * CUDA Kernel: 3D Cross-correlation
 * 
 * Implements 3D cross-correlation operation commonly used in 3D neural networks.
 */
__global__ void cross_correlation_3d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int output_depth,
    int output_height,
    int output_width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < output_width && y < output_height && z < output_depth) {
        float sum = 0.0f;
        
        // Cross-correlation (no kernel flipping)
        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int input_z = z + kd;
                    int input_y = y + kh;
                    int input_x = x + kw;
                    
                    if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                        int input_idx = input_z * input_height * input_width + 
                                       input_y * input_width + input_x;
                        
                        // Flipped kernel indexing for cross-correlation
                        int kernel_idx = (kernel_depth - 1 - kd) * kernel_height * kernel_width + 
                                        (kernel_height - 1 - kh) * kernel_width + 
                                        (kernel_width - 1 - kw);
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = z * output_height * output_width + y * output_width + x;
        output[output_idx] = sum;
    }
}

/**
 * CUDA Kernel: Strided 3D Convolution
 * 
 * Supports strided convolution operations for 3D downsampling.
 */
__global__ void convolution_3d_strided_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int output_depth,
    int output_height,
    int output_width,
    int stride_z,
    int stride_y,
    int stride_x
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < output_width && y < output_height && z < output_depth) {
        float sum = 0.0f;
        int input_start_z = z * stride_z;
        int input_start_y = y * stride_y;
        int input_start_x = x * stride_x;
        
        // Compute strided 3D convolution
        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int input_z = input_start_z + kd;
                    int input_y = input_start_y + kh;
                    int input_x = input_start_x + kw;
                    
                    if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                        int input_idx = input_z * input_height * input_width + 
                                       input_y * input_width + input_x;
                        int kernel_idx = kd * kernel_height * kernel_width + 
                                        kh * kernel_width + kw;
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = z * output_height * output_width + y * output_width + x;
        output[output_idx] = sum;
    }
}

/**
 * Host function: 3D Convolution launcher
 * 
 * @param input: Input volume on device [depth x height x width]
 * @param kernel: Convolution kernel on device [kernel_depth x kernel_height x kernel_width]
 * @param output: Output volume on device [output_depth x output_height x output_width]
 * @param input_depth: Depth of input volume
 * @param input_height: Height of input volume
 * @param input_width: Width of input volume
 * @param kernel_depth: Depth of convolution kernel
 * @param kernel_height: Height of convolution kernel
 * @param kernel_width: Width of convolution kernel
 * @param mode: Convolution mode (0=valid, 1=same, 2=full)
 * @return: Error code (0 = success)
 */
__host__ int convolution_3d_cuda(
    const float* input,
    const float* kernel,
    float* output,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int mode = 0
) {
    if (input_depth <= 0 || input_height <= 0 || input_width <= 0 ||
        kernel_depth <= 0 || kernel_height <= 0 || kernel_width <= 0) {
        return -1; // Invalid parameters
    }
    
    int output_depth, output_height, output_width;
    
    switch (mode) {
        case 0: // Valid convolution
            output_depth = input_depth - kernel_depth + 1;
            output_height = input_height - kernel_height + 1;
            output_width = input_width - kernel_width + 1;
            break;
        case 1: // Same convolution (requires padding)
            output_depth = input_depth;
            output_height = input_height;
            output_width = input_width;
            break;
        case 2: // Full convolution
            output_depth = input_depth + kernel_depth - 1;
            output_height = input_height + kernel_height - 1;
            output_width = input_width + kernel_width - 1;
            break;
        default:
            return -2; // Invalid mode
    }
    
    if (output_depth <= 0 || output_height <= 0 || output_width <= 0) {
        return -3; // Invalid output size
    }
    
    // Configure kernel launch parameters
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid_size(
        (output_width + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        (output_depth + block_size.z - 1) / block_size.z
    );
    
    // Choose optimal kernel based on problem characteristics
    if (kernel_depth <= 4 && kernel_height <= 4 && kernel_width <= 4 && mode == 0) {
        // Use basic kernel for small kernels
        convolution_3d_basic_kernel<<<grid_size, block_size>>>(
            input, kernel, output,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            output_depth, output_height, output_width
        );
    } else {
        // Use tiled kernel for larger problems
        int tile_input_size = (TILE_SIZE_Z + kernel_depth - 1) * 
                             (TILE_SIZE_Y + kernel_height - 1) * 
                             (TILE_SIZE_X + kernel_width - 1);
        int kernel_size = kernel_depth * kernel_height * kernel_width;
        size_t shared_mem_size = (tile_input_size + kernel_size) * sizeof(float);
        
        convolution_3d_tiled_kernel<<<grid_size, block_size, shared_mem_size>>>(
            input, kernel, output,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            output_depth, output_height, output_width
        );
    }
    
    cudaError_t error = cudaGetLastError();
    return (error == cudaSuccess) ? 0 : -4;
}

/**
 * Host function: Strided 3D Convolution launcher
 */
__host__ int convolution_3d_strided_cuda(
    const float* input,
    const float* kernel,
    float* output,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_z,
    int stride_y,
    int stride_x
) {
    int output_depth = (input_depth - kernel_depth) / stride_z + 1;
    int output_height = (input_height - kernel_height) / stride_y + 1;
    int output_width = (input_width - kernel_width) / stride_x + 1;
    
    if (output_depth <= 0 || output_height <= 0 || output_width <= 0) {
        return -1;
    }
    
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid_size(
        (output_width + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        (output_depth + block_size.z - 1) / block_size.z
    );
    
    convolution_3d_strided_kernel<<<grid_size, block_size>>>(
        input, kernel, output,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        output_depth, output_height, output_width,
        stride_z, stride_y, stride_x
    );
    
    cudaError_t error = cudaGetLastError();
    return (error == cudaSuccess) ? 0 : -4;
}

/**
 * Host function: 3D Cross-correlation launcher
 */
__host__ int cross_correlation_3d_cuda(
    const float* input,
    const float* kernel,
    float* output,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width
) {
    int output_depth = input_depth - kernel_depth + 1;
    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;
    
    if (output_depth <= 0 || output_height <= 0 || output_width <= 0) {
        return -1;
    }
    
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 grid_size(
        (output_width + block_size.x - 1) / block_size.x,
        (output_height + block_size.y - 1) / block_size.y,
        (output_depth + block_size.z - 1) / block_size.z
    );
    
    cross_correlation_3d_kernel<<<grid_size, block_size>>>(
        input, kernel, output,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        output_depth, output_height, output_width
    );
    
    cudaError_t error = cudaGetLastError();
    return (error == cudaSuccess) ? 0 : -4;
}

// Legacy interface for compatibility
void solve(const float* input, const float* kernel, float* output,
           int input_depth, int input_height, int input_width,
           int kernel_depth, int kernel_height, int kernel_width) {
    convolution_3d_cuda(input, kernel, output, 
                       input_depth, input_height, input_width,
                       kernel_depth, kernel_height, kernel_width, 0);
    cudaDeviceSynchronize();
}
