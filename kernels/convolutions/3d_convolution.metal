#include <metal_stdlib>
using namespace metal;

/*
 * 3D Convolution - Metal Implementation
 * 
 * GPU-accelerated implementation of 3D convolution operation
 * optimized for Apple Silicon GPUs using Metal Shading Language.
 * 
 * Mathematical foundation:
 * - 3D Convolution: (f * g)[i,j,k] = ΣΣΣ f[m,n,p] * g[i-m,j-n,k-p] for all m,n,p
 * - 3D Cross-correlation: (f ★ g)[i,j,k] = ΣΣΣ f[m,n,p] * g[i+m,j+n,k+p] for all m,n,p
 * - Valid convolution: output size = (input_size - kernel_size + 1)
 * - Same convolution: output size = input_size (with padding)
 * - Full convolution: output size = (input_size + kernel_size - 1)
 * 
 * Memory patterns:
 * - 3D thread block decomposition (x, y, z dimensions)
 * - Threadgroup memory tiling for volumetric data caching
 * - Proper boundary handling with 3D padding strategies
 * - Optimized memory layout for 3D data structures
 * 
 * Numerical considerations:
 * - Single precision floating point operations
 * - 3D boundary handling with zero-padding
 * - Accumulation strategies for large 3D kernels
 */

// Configuration constants
constant uint TILE_SIZE_X = 8;
constant uint TILE_SIZE_Y = 8;
constant uint TILE_SIZE_Z = 8;
constant uint MAX_KERNEL_SIZE_3D = 16;
constant uint THREADGROUP_SIZE_X = 8;
constant uint THREADGROUP_SIZE_Y = 8;
constant uint THREADGROUP_SIZE_Z = 8;

/**
 * Metal Kernel: Basic 3D Convolution
 * Each thread computes one output voxel using direct convolution.
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
kernel void convolution_3d_basic(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_depth [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& kernel_depth [[buffer(6)]],
    constant uint& kernel_height [[buffer(7)]],
    constant uint& kernel_width [[buffer(8)]],
    constant uint& output_depth [[buffer(9)]],
    constant uint& output_height [[buffer(10)]],
    constant uint& output_width [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;
    uint z = gid.z;
    
    if (x < output_width && y < output_height && z < output_depth) {
        float sum = 0.0f;
        
        // Compute 3D convolution for this output voxel
        for (uint kd = 0; kd < kernel_depth; kd++) {
            for (uint kh = 0; kh < kernel_height; kh++) {
                for (uint kw = 0; kw < kernel_width; kw++) {
                    uint input_z = z + kd;
                    uint input_y = y + kh;
                    uint input_x = x + kw;
                    
                    if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                        uint input_idx = input_z * input_height * input_width + 
                                        input_y * input_width + input_x;
                        uint kernel_idx = kd * kernel_height * kernel_width + 
                                         kh * kernel_width + kw;
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        uint output_idx = z * output_height * output_width + y * output_width + x;
        output[output_idx] = sum;
    }
}

/**
 * Metal Kernel: Tiled 3D Convolution with Threadgroup Memory
 * Optimized implementation using threadgroup memory for 3D data caching.
 */
kernel void convolution_3d_tiled(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_depth [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& kernel_depth [[buffer(6)]],
    constant uint& kernel_height [[buffer(7)]],
    constant uint& kernel_width [[buffer(8)]],
    constant uint& output_depth [[buffer(9)]],
    constant uint& output_height [[buffer(10)]],
    constant uint& output_width [[buffer(11)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    threadgroup float* shared_kernel [[threadgroup(1)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 thread_count [[threads_per_threadgroup]]
) {
    uint tx = tid.x;
    uint ty = tid.y;
    uint tz = tid.z;
    uint bx = gid.x;
    uint by = gid.y;
    uint bz = gid.z;
    
    // Calculate tile dimensions
    uint tile_depth = TILE_SIZE_Z + kernel_depth - 1;
    uint tile_height = TILE_SIZE_Y + kernel_height - 1;
    uint tile_width = TILE_SIZE_X + kernel_width - 1;
    
    // Load kernel into threadgroup memory
    uint tid_linear = tz * thread_count.y * thread_count.x + ty * thread_count.x + tx;
    uint kernel_size = kernel_depth * kernel_height * kernel_width;
    if (tid_linear < kernel_size) {
        shared_kernel[tid_linear] = kernel[tid_linear];
    }
    
    // Load input tile into threadgroup memory
    uint input_start_z = bz * TILE_SIZE_Z;
    uint input_start_y = by * TILE_SIZE_Y;
    uint input_start_x = bx * TILE_SIZE_X;
    
    for (uint dz = tz; dz < tile_depth; dz += thread_count.z) {
        for (uint dy = ty; dy < tile_height; dy += thread_count.y) {
            for (uint dx = tx; dx < tile_width; dx += thread_count.x) {
                uint input_z = input_start_z + dz;
                uint input_y = input_start_y + dy;
                uint input_x = input_start_x + dx;
                
                float val = 0.0f;
                if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                    uint input_idx = input_z * input_height * input_width + 
                                    input_y * input_width + input_x;
                    val = input[input_idx];
                }
                
                uint shared_idx = dz * tile_height * tile_width + dy * tile_width + dx;
                shared_input[shared_idx] = val;
            }
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute convolution using threadgroup memory
    uint output_x = input_start_x + tx;
    uint output_y = input_start_y + ty;
    uint output_z = input_start_z + tz;
    
    if (output_x < output_width && output_y < output_height && output_z < output_depth) {
        float sum = 0.0f;
        
        for (uint kd = 0; kd < kernel_depth; kd++) {
            for (uint kh = 0; kh < kernel_height; kh++) {
                for (uint kw = 0; kw < kernel_width; kw++) {
                    uint shared_z = tz + kd;
                    uint shared_y = ty + kh;
                    uint shared_x = tx + kw;
                    
                    if (shared_z < tile_depth && shared_y < tile_height && shared_x < tile_width) {
                        uint shared_input_idx = shared_z * tile_height * tile_width + 
                                               shared_y * tile_width + shared_x;
                        uint kernel_idx = kd * kernel_height * kernel_width + 
                                         kh * kernel_width + kw;
                        
                        sum += shared_input[shared_input_idx] * shared_kernel[kernel_idx];
                    }
                }
            }
        }
        
        uint output_idx = output_z * output_height * output_width + 
                         output_y * output_width + output_x;
        output[output_idx] = sum;
    }
}

/**
 * Metal Kernel: 3D Cross-correlation
 * Implements 3D cross-correlation operation (commonly used in 3D neural networks).
 */
kernel void cross_correlation_3d(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_depth [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& kernel_depth [[buffer(6)]],
    constant uint& kernel_height [[buffer(7)]],
    constant uint& kernel_width [[buffer(8)]],
    constant uint& output_depth [[buffer(9)]],
    constant uint& output_height [[buffer(10)]],
    constant uint& output_width [[buffer(11)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 thread_count [[threads_per_threadgroup]]
) {
    uint x = gid.x * thread_count.x + tid.x;
    uint y = gid.y * thread_count.y + tid.y;
    uint z = gid.z * thread_count.z + tid.z;
    
    // Load kernel into threadgroup memory
    uint tid_linear = tid.z * thread_count.y * thread_count.x + tid.y * thread_count.x + tid.x;
    uint kernel_size = kernel_depth * kernel_height * kernel_width;
    if (tid_linear < kernel_size) {
        shared_kernel[tid_linear] = kernel[tid_linear];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (x < output_width && y < output_height && z < output_depth) {
        float sum = 0.0f;
        
        // Cross-correlation (no kernel flipping)
        for (uint kd = 0; kd < kernel_depth; kd++) {
            for (uint kh = 0; kh < kernel_height; kh++) {
                for (uint kw = 0; kw < kernel_width; kw++) {
                    uint input_z = z + kd;
                    uint input_y = y + kh;
                    uint input_x = x + kw;
                    
                    if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                        uint input_idx = input_z * input_height * input_width + 
                                        input_y * input_width + input_x;
                        
                        // Flipped kernel indexing for cross-correlation
                        uint kernel_idx = (kernel_depth - 1 - kd) * kernel_height * kernel_width + 
                                         (kernel_height - 1 - kh) * kernel_width + 
                                         (kernel_width - 1 - kw);
                        
                        sum += input[input_idx] * shared_kernel[kernel_idx];
                    }
                }
            }
        }
        
        uint output_idx = z * output_height * output_width + y * output_width + x;
        output[output_idx] = sum;
    }
}

/**
 * Metal Kernel: Strided 3D Convolution
 * Supports strided convolution operations for 3D downsampling.
 */
kernel void convolution_3d_strided(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_depth [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& kernel_depth [[buffer(6)]],
    constant uint& kernel_height [[buffer(7)]],
    constant uint& kernel_width [[buffer(8)]],
    constant uint& output_depth [[buffer(9)]],
    constant uint& output_height [[buffer(10)]],
    constant uint& output_width [[buffer(11)]],
    constant uint& stride_z [[buffer(12)]],
    constant uint& stride_y [[buffer(13)]],
    constant uint& stride_x [[buffer(14)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 thread_count [[threads_per_threadgroup]]
) {
    uint x = gid.x * thread_count.x + tid.x;
    uint y = gid.y * thread_count.y + tid.y;
    uint z = gid.z * thread_count.z + tid.z;
    
    // Load kernel into threadgroup memory
    uint tid_linear = tid.z * thread_count.y * thread_count.x + tid.y * thread_count.x + tid.x;
    uint kernel_size = kernel_depth * kernel_height * kernel_width;
    if (tid_linear < kernel_size) {
        shared_kernel[tid_linear] = kernel[tid_linear];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (x < output_width && y < output_height && z < output_depth) {
        float sum = 0.0f;
        uint input_start_z = z * stride_z;
        uint input_start_y = y * stride_y;
        uint input_start_x = x * stride_x;
        
        // Compute strided 3D convolution
        for (uint kd = 0; kd < kernel_depth; kd++) {
            for (uint kh = 0; kh < kernel_height; kh++) {
                for (uint kw = 0; kw < kernel_width; kw++) {
                    uint input_z = input_start_z + kd;
                    uint input_y = input_start_y + kh;
                    uint input_x = input_start_x + kw;
                    
                    if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                        uint input_idx = input_z * input_height * input_width + 
                                        input_y * input_width + input_x;
                        uint kernel_idx = kd * kernel_height * kernel_width + 
                                         kh * kernel_width + kw;
                        
                        sum += input[input_idx] * shared_kernel[kernel_idx];
                    }
                }
            }
        }
        
        uint output_idx = z * output_height * output_width + y * output_width + x;
        output[output_idx] = sum;
    }
}

/**
 * Metal Kernel: SIMD-optimized 3D Convolution
 * Uses SIMD operations for improved arithmetic throughput on Apple Silicon.
 */
kernel void convolution_3d_simd(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_depth [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& kernel_depth [[buffer(6)]],
    constant uint& kernel_height [[buffer(7)]],
    constant uint& kernel_width [[buffer(8)]],
    constant uint& output_depth [[buffer(9)]],
    constant uint& output_height [[buffer(10)]],
    constant uint& output_width [[buffer(11)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 thread_count [[threads_per_threadgroup]]
) {
    uint x = gid.x * thread_count.x + tid.x;
    uint y = gid.y * thread_count.y + tid.y;
    uint z = gid.z * thread_count.z + tid.z;
    
    // Load kernel into threadgroup memory
    uint tid_linear = tid.z * thread_count.y * thread_count.x + tid.y * thread_count.x + tid.x;
    uint kernel_size = kernel_depth * kernel_height * kernel_width;
    if (tid_linear < kernel_size) {
        shared_kernel[tid_linear] = kernel[tid_linear];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (x < output_width && y < output_height && z < output_depth) {
        float4 sum = float4(0.0f);
        uint k = 0;
        
        // Vectorized computation for improved throughput
        for (uint kd = 0; kd < kernel_depth; kd++) {
            for (uint kh = 0; kh < kernel_height; kh++) {
                uint kw = 0;
                for (; kw + 3 < kernel_width; kw += 4) {
                    uint input_z = z + kd;
                    uint input_y = y + kh;
                    uint input_x = x + kw;
                    
                    if (input_z < input_depth && input_y < input_height && input_x + 3 < input_width) {
                        uint input_base = input_z * input_height * input_width + input_y * input_width;
                        float4 input_vec = float4(
                            input[input_base + input_x],
                            input[input_base + input_x + 1],
                            input[input_base + input_x + 2],
                            input[input_base + input_x + 3]
                        );
                        
                        uint kernel_base = kd * kernel_height * kernel_width + kh * kernel_width;
                        float4 kernel_vec = float4(
                            shared_kernel[kernel_base + kw],
                            shared_kernel[kernel_base + kw + 1],
                            shared_kernel[kernel_base + kw + 2],
                            shared_kernel[kernel_base + kw + 3]
                        );
                        
                        sum += input_vec * kernel_vec;
                    }
                }
                
                // Handle remaining elements
                for (; kw < kernel_width; kw++) {
                    uint input_z = z + kd;
                    uint input_y = y + kh;
                    uint input_x = x + kw;
                    
                    if (input_z < input_depth && input_y < input_height && input_x < input_width) {
                        uint input_idx = input_z * input_height * input_width + 
                                        input_y * input_width + input_x;
                        uint kernel_idx = kd * kernel_height * kernel_width + 
                                         kh * kernel_width + kw;
                        
                        sum.x += input[input_idx] * shared_kernel[kernel_idx];
                    }
                }
            }
        }
        
        float result = sum.x + sum.y + sum.z + sum.w;
        uint output_idx = z * output_height * output_width + y * output_width + x;
        output[output_idx] = result;
    }
}
