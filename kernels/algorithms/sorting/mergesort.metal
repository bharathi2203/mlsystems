#include <metal_stdlib>
using namespace metal;

/*
 * Merge Sort - Metal Implementation
 * 
 * GPU-accelerated implementation of the merge sort algorithm
 * using parallel divide-and-conquer approach optimized for Apple Silicon.
 * 
 * Algorithm overview:
 * - Divide-and-conquer sorting algorithm
 * - Recursively divides array into halves
 * - Merges sorted subarrays in parallel
 * - Stable sorting (preserves relative order of equal elements)
 * - O(n log n) time complexity, O(n) space complexity
 * 
 * Memory patterns:
 * - Double buffering to avoid in-place merge complexity
 * - Coalesced reads and writes across thread boundaries
 * - Threadgroup memory for thread local operations
 * - Register-based merge for very small arrays
 * 
 * Performance considerations:
 * - Optimal for large arrays (>1K elements)
 * - Handles arbitrary array sizes with padding
 * - Memory bandwidth bound for large arrays
 * - Compute bound for merge operations
 */

// Configuration constants
constant uint MAX_THREADGROUP_SIZE = 1024;
constant uint SHARED_MEM_SIZE = 16384; // 16KB in floats
constant uint MERGE_TILE_SIZE = 256;

/**
 * Metal Kernel: Small array sorting using bitonic sort
 * For arrays small enough to fit in threadgroup memory
 */
kernel void bitonic_sort_small(
    device float* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    threadgroup float* shared_data [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_start = threadgroup_id * threads_per_threadgroup * 2;
    uint global_id = global_start + thread_id;
    
    // Load data into threadgroup memory
    shared_data[thread_id] = (global_id < n) ? data[global_id] : MAXFLOAT;
    shared_data[thread_id + threads_per_threadgroup] = (global_id + threads_per_threadgroup < n) ? 
        data[global_id + threads_per_threadgroup] : MAXFLOAT;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint array_size = threads_per_threadgroup * 2;
    
    // Bitonic sort
    for (uint stage = 2; stage <= array_size; stage <<= 1) {
        for (uint step = stage >> 1; step > 0; step >>= 1) {
            uint partner = thread_id ^ step;
            
            if (partner > thread_id) {
                bool ascending = ((thread_id & stage) == 0);
                
                float val1 = shared_data[thread_id];
                float val2 = shared_data[partner];
                
                if ((val1 > val2) == ascending) {
                    shared_data[thread_id] = val2;
                    shared_data[partner] = val1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // Store results back to global memory
    if (global_id < n) {
        data[global_id] = shared_data[thread_id];
    }
    if (global_id + threads_per_threadgroup < n) {
        data[global_id + threads_per_threadgroup] = shared_data[thread_id + threads_per_threadgroup];
    }
}

/**
 * Metal Kernel: Parallel merge operation
 * Merges two sorted subarrays using threadgroup cooperation
 */
kernel void parallel_merge(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& left_start [[buffer(2)]],
    constant uint& left_end [[buffer(3)]],
    constant uint& right_start [[buffer(4)]],
    constant uint& right_end [[buffer(5)]],
    constant uint& dst_start [[buffer(6)]],
    threadgroup float* shared_left [[threadgroup(0)]],
    threadgroup float* shared_right [[threadgroup(1)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint left_size = left_end - left_start;
    uint right_size = right_end - right_start;
    uint total_size = left_size + right_size;
    
    // Cooperatively load left subarray
    for (uint i = thread_id; i < left_size; i += threads_per_threadgroup) {
        shared_left[i] = src[left_start + i];
    }
    
    // Cooperatively load right subarray
    for (uint i = thread_id; i < right_size; i += threads_per_threadgroup) {
        shared_right[i] = src[right_start + i];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel merge using binary search
    for (uint i = thread_id; i < total_size; i += threads_per_threadgroup) {
        uint left_idx = 0;
        uint right_idx = 0;
        
        // Binary search to find position
        uint left_low = 0, left_high = left_size;
        uint right_low = 0, right_high = right_size;
        
        while (left_low < left_high || right_low < right_high) {
            uint left_mid = (left_low + left_high) / 2;
            uint right_mid = (right_low + right_high) / 2;
            
            if (left_mid + right_mid <= i) {
                if (left_mid == left_size || 
                    (right_mid < right_size && shared_left[left_mid] > shared_right[right_mid])) {
                    right_low = right_mid + 1;
                } else {
                    left_low = left_mid + 1;
                }
            } else {
                if (left_mid == 0 || 
                    (right_mid > 0 && shared_left[left_mid - 1] <= shared_right[right_mid - 1])) {
                    right_high = right_mid;
                } else {
                    left_high = left_mid;
                }
            }
        }
        
        left_idx = left_low;
        right_idx = i - left_idx;
        
        // Store merged element
        if (left_idx < left_size && (right_idx >= right_size || shared_left[left_idx] <= shared_right[right_idx])) {
            dst[dst_start + i] = shared_left[left_idx];
        } else {
            dst[dst_start + i] = shared_right[right_idx];
        }
    }
}

/**
 * Metal Kernel: Bottom-up merge sort for medium arrays
 * Iteratively merges sorted subarrays of increasing size
 */
kernel void mergesort_bottom_up(
    device float* data [[buffer(0)]],
    device float* temp [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& subarray_size [[buffer(3)]],
    threadgroup float* shared_data [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint merge_start = threadgroup_id * subarray_size * 2;
    
    if (merge_start >= n) return;
    
    uint left_start = merge_start;
    uint left_end = min(left_start + subarray_size, n);
    uint right_start = left_end;
    uint right_end = min(right_start + subarray_size, n);
    
    // If no right subarray, just copy
    if (right_start >= n) {
        for (uint i = thread_id; i < left_end - left_start; i += threads_per_threadgroup) {
            temp[left_start + i] = data[left_start + i];
        }
        return;
    }
    
    uint left_size = left_end - left_start;
    uint right_size = right_end - right_start;
    uint total_size = left_size + right_size;
    
    // Load data into threadgroup memory if it fits
    uint max_shared = threads_per_threadgroup;
    if (total_size <= max_shared) {
        // Use shared memory merge
        for (uint i = thread_id; i < left_size; i += threads_per_threadgroup) {
            shared_data[i] = data[left_start + i];
        }
        for (uint i = thread_id; i < right_size; i += threads_per_threadgroup) {
            shared_data[left_size + i] = data[right_start + i];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Perform merge
        for (uint i = thread_id; i < total_size; i += threads_per_threadgroup) {
            uint left_idx = 0;
            uint right_idx = 0;
            
            // Simple merge logic
            while (left_idx < left_size && right_idx < right_size && left_idx + right_idx < i) {
                if (shared_data[left_idx] <= shared_data[left_size + right_idx]) {
                    left_idx++;
                } else {
                    right_idx++;
                }
            }
            
            while (left_idx + right_idx < i && left_idx < left_size) left_idx++;
            while (left_idx + right_idx < i && right_idx < right_size) right_idx++;
            
            if (left_idx < left_size && (right_idx >= right_size || 
                shared_data[left_idx] <= shared_data[left_size + right_idx])) {
                temp[left_start + i] = shared_data[left_idx];
            } else if (right_idx < right_size) {
                temp[left_start + i] = shared_data[left_size + right_idx];
            }
        }
    } else {
        // Direct global memory merge for large subarrays
        for (uint i = thread_id; i < total_size; i += threads_per_threadgroup) {
            uint left_idx = 0;
            uint right_idx = 0;
            
            // Binary search approach for position
            uint low = 0, high = min(i + 1, left_size);
            while (low < high) {
                uint mid = (low + high) / 2;
                uint right_pos = i - mid;
                
                if (right_pos >= right_size) {
                    high = mid;
                } else if (mid >= left_size) {
                    low = mid + 1;
                } else if (data[left_start + mid] <= data[right_start + right_pos]) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            
            left_idx = low;
            right_idx = i - left_idx;
            
            if (left_idx < left_size && (right_idx >= right_size || 
                data[left_start + left_idx] <= data[right_start + right_idx])) {
                temp[left_start + i] = data[left_start + left_idx];
            } else if (right_idx < right_size) {
                temp[left_start + i] = data[right_start + right_idx];
            }
        }
    }
}

/**
 * Metal Kernel: Copy data between arrays
 * Utility kernel for swapping buffers
 */
kernel void copy_data(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id < n) {
        dst[global_id] = src[global_id];
    }
}

/**
 * Metal Kernel: Verify array is sorted
 * Validation kernel for debugging and testing
 */
kernel void verify_sorted(
    device const float* data [[buffer(0)]],
    device atomic_uint* error_count [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id < n - 1) {
        if (data[global_id] > data[global_id + 1]) {
            atomic_fetch_add_explicit(error_count, 1, memory_order_relaxed);
        }
    }
}

/**
 * Metal Kernel: Initialize array with random data
 * Utility for testing and benchmarking
 */
kernel void generate_random_data(
    device float* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant uint& seed [[buffer(2)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id < n) {
        // Simple linear congruential generator
        uint rng_state = seed + global_id;
        rng_state = rng_state * 1664525u + 1013904223u;
        
        // Convert to float [0, 1000)
        float rand_val = float(rng_state) / float(0xFFFFFFFFu) * 1000.0f;
        data[global_id] = rand_val;
    }
}

/**
 * Metal Kernel: Compute performance metrics
 * Measures sorting performance and validates correctness
 */
kernel void compute_metrics(
    device const float* original [[buffer(0)]],
    device const float* sorted [[buffer(1)]],
    device float* metrics [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    threadgroup float* shared_metrics [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    float local_sum_original = 0.0f;
    float local_sum_sorted = 0.0f;
    float local_errors = 0.0f;
    
    // Process multiple elements per thread
    for (uint i = global_id; i < n; i += threads_per_threadgroup * threadgroups_per_grid) {
        local_sum_original += original[i];
        local_sum_sorted += sorted[i];
        
        // Check for sorting errors (except last element)
        if (i < n - 1 && sorted[i] > sorted[i + 1]) {
            local_errors += 1.0f;
        }
    }
    
    shared_metrics[thread_id * 3] = local_sum_original;
    shared_metrics[thread_id * 3 + 1] = local_sum_sorted;
    shared_metrics[thread_id * 3 + 2] = local_errors;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_metrics[thread_id * 3] += shared_metrics[(thread_id + stride) * 3];
            shared_metrics[thread_id * 3 + 1] += shared_metrics[(thread_id + stride) * 3 + 1];
            shared_metrics[thread_id * 3 + 2] += shared_metrics[(thread_id + stride) * 3 + 2];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store results
    if (thread_id == 0) {
        device atomic<float>* sum_orig_ptr = (device atomic<float>*)&metrics[0];
        device atomic<float>* sum_sorted_ptr = (device atomic<float>*)&metrics[1];
        device atomic<float>* errors_ptr = (device atomic<float>*)&metrics[2];
        
        atomic_fetch_add_explicit(sum_orig_ptr, shared_metrics[0], memory_order_relaxed);
        atomic_fetch_add_explicit(sum_sorted_ptr, shared_metrics[1], memory_order_relaxed);
        atomic_fetch_add_explicit(errors_ptr, shared_metrics[2], memory_order_relaxed);
    }
}
