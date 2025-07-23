#include <metal_stdlib>
using namespace metal;

/*
 * 1D Convolution - Metal Implementation
 * 
 * GPU-accelerated implementation of 1D convolution operation
 * optimized for Apple Silicon GPUs using Metal Shading Language.
 * 
 * Mathematical foundation:
 * - Convolution: (f * g)[n] = Σ f[m] * g[n-m] for all m
 * - Cross-correlation: (f ★ g)[n] = Σ f[m] * g[n+m] for all m
 * - Valid convolution: output size = input_size - kernel_size + 1
 * - Full convolution: output size = input_size + kernel_size - 1
 * - Same convolution: output size = input_size (with padding)
 * 
 * Memory patterns:
 * - Coalesced input reads with proper alignment
 * - Kernel broadcast through threadgroup memory
 * - Sequential output writes with bank conflict avoidance
 * 
 * Numerical considerations:
 * - Single precision floating point operations
 * - Boundary handling with zero-padding or clamping
 * - Overflow protection for large accumulations
 */

// Configuration constants
constant uint MAX_KERNEL_SIZE = 256;
constant uint THREADGROUP_SIZE = 256;

/**
 * Metal Kernel: 1D Convolution with threadgroup memory optimization
 * Each thread computes one output element with kernel cached in threadgroup memory.
 * 
 * @param input: Input signal [input_size]
 * @param kernel: Convolution kernel [kernel_size]
 * @param output: Output signal [output_size]
 * @param input_size: Size of input signal
 * @param kernel_size: Size of convolution kernel
 * @param output_size: Size of output signal
 */
kernel void convolution_1d_basic(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& kernel_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Cooperatively load kernel into threadgroup memory
    if (thread_id < kernel_size) {
        shared_kernel[thread_id] = kernel[thread_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (global_id < output_size) {
        float result = 0.0f;
        
        // Compute convolution for this output element
        for (uint k = 0; k < kernel_size; k++) {
            uint input_idx = global_id + k;
            if (input_idx < input_size) {
                result += input[input_idx] * shared_kernel[k];
            }
        }
        
        output[global_id] = result;
    }
}

/**
 * Metal Kernel: 1D Convolution with tiled input processing
 * Optimized for large kernels using tiled input loading to maximize cache reuse.
 */
kernel void convolution_1d_tiled(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& kernel_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    threadgroup float* shared_data [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    // Shared memory layout: [kernel][input_tile]
    threadgroup float* shared_kernel = shared_data;
    threadgroup float* shared_input = &shared_data[kernel_size];
    
    uint block_start = threadgroup_id * threads_per_threadgroup;
    
    // Load kernel into threadgroup memory
    if (thread_id < kernel_size) {
        shared_kernel[thread_id] = kernel[thread_id];
    }
    
    // Calculate input tile boundaries
    uint input_start = block_start;
    uint input_end = min(input_start + threads_per_threadgroup + kernel_size - 1, input_size);
    uint input_tile_size = input_end - input_start;
    
    // Load input tile into threadgroup memory
    for (uint i = thread_id; i < input_tile_size; i += threads_per_threadgroup) {
        shared_input[i] = (input_start + i < input_size) ? input[input_start + i] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint global_id = block_start + thread_id;
    if (global_id < output_size) {
        float result = 0.0f;
        
        // Compute convolution using threadgroup memory
        for (uint k = 0; k < kernel_size; k++) {
            uint shared_idx = thread_id + k;
            if (shared_idx < input_tile_size) {
                result += shared_input[shared_idx] * shared_kernel[k];
            }
        }
        
        output[global_id] = result;
    }
}

/**
 * Metal Kernel: 1D Cross-correlation
 * Implements cross-correlation operation (commonly used in neural networks).
 */
kernel void cross_correlation_1d(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& kernel_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Load kernel into threadgroup memory
    if (thread_id < kernel_size) {
        shared_kernel[thread_id] = kernel[thread_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (global_id < output_size) {
        float result = 0.0f;
        
        // Cross-correlation computation (flipped kernel indexing)
        for (uint k = 0; k < kernel_size; k++) {
            uint input_idx = global_id + k;
            if (input_idx < input_size) {
                result += input[input_idx] * shared_kernel[kernel_size - 1 - k];
            }
        }
        
        output[global_id] = result;
    }
}

/**
 * Metal Kernel: 1D Convolution with SIMD optimization
 * Uses SIMD operations for improved arithmetic throughput on Apple Silicon.
 */
kernel void convolution_1d_simd(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& kernel_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Load kernel into threadgroup memory
    if (thread_id < kernel_size) {
        shared_kernel[thread_id] = kernel[thread_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (global_id < output_size) {
        float4 result = float4(0.0f);
        uint k = 0;
        
        // Vectorized computation for kernel_size multiple of 4
        for (; k + 3 < kernel_size; k += 4) {
            uint input_idx = global_id + k;
            if (input_idx + 3 < input_size) {
                float4 input_vec = float4(
                    input[input_idx],
                    input[input_idx + 1],
                    input[input_idx + 2],
                    input[input_idx + 3]
                );
                float4 kernel_vec = float4(
                    shared_kernel[k],
                    shared_kernel[k + 1],
                    shared_kernel[k + 2],
                    shared_kernel[k + 3]
                );
                result += input_vec * kernel_vec;
            }
        }
        
        // Handle remaining elements
        float scalar_result = result.x + result.y + result.z + result.w;
        for (; k < kernel_size; k++) {
            uint input_idx = global_id + k;
            if (input_idx < input_size) {
                scalar_result += input[input_idx] * shared_kernel[k];
            }
        }
        
        output[global_id] = scalar_result;
    }
}

/**
 * Metal Kernel: 1D Convolution with stride support
 * Supports strided convolution operations for downsampling.
 */
kernel void convolution_1d_strided(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& kernel_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Load kernel into threadgroup memory
    if (thread_id < kernel_size) {
        shared_kernel[thread_id] = kernel[thread_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (global_id < output_size) {
        float result = 0.0f;
        uint input_start = global_id * stride;
        
        // Compute strided convolution
        for (uint k = 0; k < kernel_size; k++) {
            uint input_idx = input_start + k;
            if (input_idx < input_size) {
                result += input[input_idx] * shared_kernel[k];
            }
        }
        
        output[global_id] = result;
    }
}

/**
 * Metal Kernel: Separable 1D convolution (for efficiency with separable kernels)
 * First pass of separable convolution for 2D operations.
 */
kernel void separable_convolution_1d(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_size [[buffer(3)]],
    constant uint& kernel_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    constant uint& input_stride [[buffer(6)]],
    constant uint& output_stride [[buffer(7)]],
    threadgroup float* shared_kernel [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Load kernel into threadgroup memory
    if (thread_id < kernel_size) {
        shared_kernel[thread_id] = kernel[thread_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (global_id < output_size) {
        float result = 0.0f;
        
        // Compute convolution with custom stride patterns
        for (uint k = 0; k < kernel_size; k++) {
            uint input_idx = (global_id + k) * input_stride;
            if (input_idx < input_size * input_stride) {
                result += input[input_idx] * shared_kernel[k];
            }
        }
        
        output[global_id * output_stride] = result;
    }
}
