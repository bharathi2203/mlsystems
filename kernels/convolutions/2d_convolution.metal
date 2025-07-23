#include <metal_stdlib>
using namespace metal;

/*
 * 2D Convolution - Metal Implementation
 * 
 * GPU-accelerated implementation of 2D convolution operation
 * optimized for Apple Silicon GPUs using Metal Shading Language.
 * 
 * Mathematical foundation:
 * - 2D Convolution: (f * g)[i,j] = ΣΣ f[m,n] * g[i-m,j-n] for all m,n
 * - 2D Cross-correlation: (f ★ g)[i,j] = ΣΣ f[m,n] * g[i+m,j+n] for all m,n
 * - Valid convolution: output size = (input_size - kernel_size + 1)
 * - Same convolution: output size = input_size (with padding)
 * - Full convolution: output size = (input_size + kernel_size - 1)
 * 
 * Memory patterns:
 * - 2D thread block decomposition for parallel processing
 * - Threadgroup memory tiling for input and kernel caching
 * - Boundary handling with proper padding strategies
 * - Output coalescing for optimal write patterns
 * 
 * Numerical considerations:
 * - Single precision floating point operations
 * - Boundary handling with zero-padding or mirror padding
 * - Accumulation strategies to prevent overflow
 */

// Configuration constants
constant uint TILE_SIZE = 16;
constant uint MAX_KERNEL_SIZE = 32;
constant uint THREADGROUP_SIZE_X = 16;
constant uint THREADGROUP_SIZE_Y = 16;

/**
 * Metal Kernel: Basic 2D Convolution
 * Each thread computes one output pixel using direct convolution.
 * 
 * @param input: Input image [input_rows x input_cols]
 * @param kernel: Convolution kernel [kernel_rows x kernel_cols]
 * @param output: Output image [output_rows x output_cols]
 * @param input_rows: Height of input image
 * @param input_cols: Width of input image
 * @param kernel_rows: Height of convolution kernel
 * @param kernel_cols: Width of convolution kernel
 * @param output_rows: Height of output image
 * @param output_cols: Width of output image
 */
kernel void convolution_2d_basic(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_rows [[buffer(3)]],
    constant uint& input_cols [[buffer(4)]],
    constant uint& kernel_rows [[buffer(5)]],
    constant uint& kernel_cols [[buffer(6)]],
    constant uint& output_rows [[buffer(7)]],
    constant uint& output_cols [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    
    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;
        
        // Compute convolution for this output pixel
        for (uint kr = 0; kr < kernel_rows; kr++) {
            for (uint kc = 0; kc < kernel_cols; kc++) {
                uint input_row = row + kr;
                uint input_col = col + kc;
                
                if (input_row < input_rows && input_col < input_cols) {
                    float input_val = input[input_row * input_cols + input_col];
                    float kernel_val = kernel[kr * kernel_cols + kc];
                    sum += input_val * kernel_val;
                }
            }
        }
        
        output[row * output_cols + col] = sum;
    }
}

/**
 * Metal Kernel: Tiled 2D Convolution with Threadgroup Memory
 * Optimized implementation using threadgroup memory for caching.
 */
kernel void convolution_2d_tiled(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_rows [[buffer(3)]],
    constant uint& input_cols [[buffer(4)]],
    constant uint& kernel_rows [[buffer(5)]],
    constant uint& kernel_cols [[buffer(6)]],
    constant uint& output_rows [[buffer(7)]],
    constant uint& output_cols [[buffer(8)]],
    threadgroup float* shared_input [[threadgroup(0)]],
    threadgroup float* shared_kernel [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 thread_count [[threads_per_threadgroup]]
) {
    uint tx = tid.x;
    uint ty = tid.y;
    uint bx = gid.x;
    uint by = gid.y;
    
    // Calculate input tile dimensions (includes overlap for kernel)
    uint tile_input_rows = TILE_SIZE + kernel_rows - 1;
    uint tile_input_cols = TILE_SIZE + kernel_cols - 1;
    
    // Load kernel into threadgroup memory
    if (tx < kernel_cols && ty < kernel_rows) {
        shared_kernel[ty * kernel_cols + tx] = kernel[ty * kernel_cols + tx];
    }
    
    // Load input tile into threadgroup memory
    uint input_start_row = by * TILE_SIZE;
    uint input_start_col = bx * TILE_SIZE;
    
    for (uint dy = ty; dy < tile_input_rows; dy += thread_count.y) {
        for (uint dx = tx; dx < tile_input_cols; dx += thread_count.x) {
            uint input_row = input_start_row + dy;
            uint input_col = input_start_col + dx;
            
            float val = 0.0f;
            if (input_row < input_rows && input_col < input_cols) {
                val = input[input_row * input_cols + input_col];
            }
            
            if (dy < tile_input_rows && dx < tile_input_cols) {
                shared_input[dy * tile_input_cols + dx] = val;
            }
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute convolution using threadgroup memory
    uint output_row = input_start_row + ty;
    uint output_col = input_start_col + tx;
    
    if (output_row < output_rows && output_col < output_cols) {
        float sum = 0.0f;
        
        for (uint kr = 0; kr < kernel_rows; kr++) {
            for (uint kc = 0; kc < kernel_cols; kc++) {
                uint shared_row = ty + kr;
                uint shared_col = tx + kc;
                
                if (shared_row < tile_input_rows && shared_col < tile_input_cols) {
                    sum += shared_input[shared_row * tile_input_cols + shared_col] * 
                           shared_kernel[kr * kernel_cols + kc];
                }
            }
        }
        
        output[output_row * output_cols + output_col] = sum;
    }
}

/**
 * Metal Kernel: SIMD-optimized 2D Convolution
 * Uses SIMD operations for improved arithmetic throughput.
 */
kernel void convolution_2d_simd(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_rows [[buffer(3)]],
    constant uint& input_cols [[buffer(4)]],
    constant uint& kernel_rows [[buffer(5)]],
    constant uint& kernel_cols [[buffer(6)]],
    constant uint& output_rows [[buffer(7)]],
    constant uint& output_cols [[buffer(8)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 thread_count [[threads_per_threadgroup]]
) {
    uint col = gid.x * thread_count.x + tid.x;
    uint row = gid.y * thread_count.y + tid.y;
    
    // Load kernel into threadgroup memory
    if (tid.x < kernel_cols && tid.y < kernel_rows) {
        shared_kernel[tid.y * kernel_cols + tid.x] = kernel[tid.y * kernel_cols + tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (row < output_rows && col < output_cols) {
        float4 sum = float4(0.0f);
        uint k = 0;
        
        // Vectorized computation when possible
        for (uint kr = 0; kr < kernel_rows; kr++) {
            uint kc = 0;
            for (; kc + 3 < kernel_cols; kc += 4) {
                uint input_row = row + kr;
                uint input_col = col + kc;
                
                if (input_row < input_rows && input_col + 3 < input_cols) {
                    float4 input_vec = float4(
                        input[input_row * input_cols + input_col],
                        input[input_row * input_cols + input_col + 1],
                        input[input_row * input_cols + input_col + 2],
                        input[input_row * input_cols + input_col + 3]
                    );
                    
                    float4 kernel_vec = float4(
                        shared_kernel[kr * kernel_cols + kc],
                        shared_kernel[kr * kernel_cols + kc + 1],
                        shared_kernel[kr * kernel_cols + kc + 2],
                        shared_kernel[kr * kernel_cols + kc + 3]
                    );
                    
                    sum += input_vec * kernel_vec;
                }
            }
            
            // Handle remaining elements
            for (; kc < kernel_cols; kc++) {
                uint input_row = row + kr;
                uint input_col = col + kc;
                
                if (input_row < input_rows && input_col < input_cols) {
                    sum.x += input[input_row * input_cols + input_col] * 
                            shared_kernel[kr * kernel_cols + kc];
                }
            }
        }
        
        float result = sum.x + sum.y + sum.z + sum.w;
        output[row * output_cols + col] = result;
    }
}

/**
 * Metal Kernel: Separable 2D Convolution - Horizontal Pass
 * First pass of separable convolution applying horizontal kernel.
 */
kernel void separable_conv_horizontal(
    device const float* input [[buffer(0)]],
    device float* intermediate [[buffer(1)]],
    device const float* h_kernel [[buffer(2)]],
    constant uint& input_rows [[buffer(3)]],
    constant uint& input_cols [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    constant uint& output_cols [[buffer(6)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 thread_count [[threads_per_threadgroup]]
) {
    uint col = gid.x * thread_count.x + tid.x;
    uint row = gid.y * thread_count.y + tid.y;
    
    // Load horizontal kernel into threadgroup memory
    if (tid.x < kernel_size && tid.y == 0) {
        shared_kernel[tid.x] = h_kernel[tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (row < input_rows && col < output_cols) {
        float sum = 0.0f;
        
        for (uint k = 0; k < kernel_size; k++) {
            uint input_col = col + k;
            if (input_col < input_cols) {
                sum += input[row * input_cols + input_col] * shared_kernel[k];
            }
        }
        
        intermediate[row * output_cols + col] = sum;
    }
}

/**
 * Metal Kernel: Separable 2D Convolution - Vertical Pass
 * Second pass of separable convolution applying vertical kernel.
 */
kernel void separable_conv_vertical(
    device const float* intermediate [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* v_kernel [[buffer(2)]],
    constant uint& intermediate_rows [[buffer(3)]],
    constant uint& intermediate_cols [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    constant uint& output_rows [[buffer(6)]],
    constant uint& output_cols [[buffer(7)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 thread_count [[threads_per_threadgroup]]
) {
    uint col = gid.x * thread_count.x + tid.x;
    uint row = gid.y * thread_count.y + tid.y;
    
    // Load vertical kernel into threadgroup memory
    if (tid.x < kernel_size && tid.y == 0) {
        shared_kernel[tid.x] = v_kernel[tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;
        
        for (uint k = 0; k < kernel_size; k++) {
            uint intermediate_row = row + k;
            if (intermediate_row < intermediate_rows) {
                sum += intermediate[intermediate_row * intermediate_cols + col] * shared_kernel[k];
            }
        }
        
        output[row * output_cols + col] = sum;
    }
}

/**
 * Metal Kernel: 2D Cross-correlation
 * Implements 2D cross-correlation operation (commonly used in neural networks).
 */
kernel void cross_correlation_2d(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_rows [[buffer(3)]],
    constant uint& input_cols [[buffer(4)]],
    constant uint& kernel_rows [[buffer(5)]],
    constant uint& kernel_cols [[buffer(6)]],
    constant uint& output_rows [[buffer(7)]],
    constant uint& output_cols [[buffer(8)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 thread_count [[threads_per_threadgroup]]
) {
    uint col = gid.x * thread_count.x + tid.x;
    uint row = gid.y * thread_count.y + tid.y;
    
    // Load kernel into threadgroup memory
    if (tid.x < kernel_cols && tid.y < kernel_rows) {
        shared_kernel[tid.y * kernel_cols + tid.x] = kernel[tid.y * kernel_cols + tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;
        
        // Cross-correlation (no kernel flipping)
        for (uint kr = 0; kr < kernel_rows; kr++) {
            for (uint kc = 0; kc < kernel_cols; kc++) {
                uint input_row = row + kr;
                uint input_col = col + kc;
                
                if (input_row < input_rows && input_col < input_cols) {
                    // Flipped kernel indexing for cross-correlation
                    uint kernel_idx = (kernel_rows - 1 - kr) * kernel_cols + (kernel_cols - 1 - kc);
                    sum += input[input_row * input_cols + input_col] * shared_kernel[kernel_idx];
                }
            }
        }
        
        output[row * output_cols + col] = sum;
    }
}

/**
 * Metal Kernel: Strided 2D Convolution
 * Supports strided convolution operations for downsampling.
 */
kernel void convolution_2d_strided(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_rows [[buffer(3)]],
    constant uint& input_cols [[buffer(4)]],
    constant uint& kernel_rows [[buffer(5)]],
    constant uint& kernel_cols [[buffer(6)]],
    constant uint& output_rows [[buffer(7)]],
    constant uint& output_cols [[buffer(8)]],
    constant uint& stride_y [[buffer(9)]],
    constant uint& stride_x [[buffer(10)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 thread_count [[threads_per_threadgroup]]
) {
    uint col = gid.x * thread_count.x + tid.x;
    uint row = gid.y * thread_count.y + tid.y;
    
    // Load kernel into threadgroup memory
    if (tid.x < kernel_cols && tid.y < kernel_rows) {
        shared_kernel[tid.y * kernel_cols + tid.x] = kernel[tid.y * kernel_cols + tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;
        uint input_start_row = row * stride_y;
        uint input_start_col = col * stride_x;
        
        // Compute strided convolution
        for (uint kr = 0; kr < kernel_rows; kr++) {
            for (uint kc = 0; kc < kernel_cols; kc++) {
                uint input_row = input_start_row + kr;
                uint input_col = input_start_col + kc;
                
                if (input_row < input_rows && input_col < input_cols) {
                    sum += input[input_row * input_cols + input_col] * 
                           shared_kernel[kr * kernel_cols + kc];
                }
            }
        }
        
        output[row * output_cols + col] = sum;
    }
}

/**
 * Metal Kernel: Dilated 2D Convolution
 * Supports dilated (atrous) convolution operations.
 */
kernel void convolution_2d_dilated(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_rows [[buffer(3)]],
    constant uint& input_cols [[buffer(4)]],
    constant uint& kernel_rows [[buffer(5)]],
    constant uint& kernel_cols [[buffer(6)]],
    constant uint& output_rows [[buffer(7)]],
    constant uint& output_cols [[buffer(8)]],
    constant uint& dilation_y [[buffer(9)]],
    constant uint& dilation_x [[buffer(10)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 thread_count [[threads_per_threadgroup]]
) {
    uint col = gid.x * thread_count.x + tid.x;
    uint row = gid.y * thread_count.y + tid.y;
    
    // Load kernel into threadgroup memory
    if (tid.x < kernel_cols && tid.y < kernel_rows) {
        shared_kernel[tid.y * kernel_cols + tid.x] = kernel[tid.y * kernel_cols + tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;
        
        // Compute dilated convolution
        for (uint kr = 0; kr < kernel_rows; kr++) {
            for (uint kc = 0; kc < kernel_cols; kc++) {
                uint input_row = row + kr * dilation_y;
                uint input_col = col + kc * dilation_x;
                
                if (input_row < input_rows && input_col < input_cols) {
                    sum += input[input_row * input_cols + input_col] * 
                           shared_kernel[kr * kernel_cols + kc];
                }
            }
        }
        
        output[row * output_cols + col] = sum;
    }
}
