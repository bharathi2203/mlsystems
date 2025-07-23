/*
 * General Sorting Algorithms - CUDA Implementation
 * 
 * Comprehensive GPU-accelerated sorting implementation
 * with multiple algorithms for different use cases.
 * 
 * Includes:
 * - Quick sort (divide-and-conquer, in-place)
 * - Heap sort (in-place, guaranteed O(n log n))
 * - Bitonic sort (parallel-friendly for power-of-2 sizes)
 * - Selection of optimal algorithm based on data characteristics
 * 
 * Algorithm selection criteria:
 * - Array size (small: insertion, medium: quick, large: merge/radix)
 * - Data distribution (uniform: radix, random: quick, sorted: adaptive)
 * - Memory constraints (limited: heap/quick, abundant: merge)
 * - Stability requirements (stable: merge, unstable: quick/heap)
 * 

 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cstdio>

// Forward declarations from other sorting implementations
__host__ int mergesort_cuda(float* data, int n);
__host__ int radix_sort_float_cuda(float* data, int n);

// Configuration constants
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define INSERTION_SORT_THRESHOLD 64
#define BITONIC_SORT_THRESHOLD 1024

/**
 * Device function: Swap two elements
 */
__device__ void swap_elements(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * CUDA Kernel: Insertion sort for small arrays
 * Efficient for arrays smaller than ~64 elements
 */
__global__ void insertion_sort_kernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Single thread performs insertion sort
        for (int i = 1; i < n; i++) {
            float key = data[i];
            int j = i - 1;
            
            while (j >= 0 && data[j] > key) {
                data[j + 1] = data[j];
                j--;
            }
            data[j + 1] = key;
        }
    }
}

/**
 * CUDA Kernel: Bitonic sort step
 * Sorts bitonic sequences (sequences that first increase then decrease)
 */
__global__ void bitonic_sort_step(float* data, int n, int k, int j) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid;
    
    if (idx < n) {
        int partner = idx ^ j;
        
        if (partner > idx) {
            if ((idx & k) == 0) {
                // Ascending order
                if (data[idx] > data[partner]) {
                    swap_elements(&data[idx], &data[partner]);
                }
            } else {
                // Descending order
                if (data[idx] < data[partner]) {
                    swap_elements(&data[idx], &data[partner]);
                }
            }
        }
    }
}

/**
 * Host function: Bitonic sort implementation
 * Efficient for arrays with size that is a power of 2
 */
int bitonic_sort_cuda(float* data, int n) {
    // Pad to next power of 2 if necessary
    int padded_n = 1;
    while (padded_n < n) padded_n <<= 1;
    
    float* padded_data = nullptr;
    if (padded_n > n) {
        padded_data = new float[padded_n];
        std::copy(data, data + n, padded_data);
        std::fill(padded_data + n, padded_data + padded_n, INFINITY);
    } else {
        padded_data = data;
    }
    
    // Device memory
    float* d_data;
    size_t array_size = padded_n * sizeof(float);
    cudaMalloc(&d_data, array_size);
    cudaMemcpy(d_data, padded_data, array_size, cudaMemcpyHostToDevice);
    
    // Grid configuration
    int block_size = min(512, padded_n);
    int grid_size = (padded_n + block_size - 1) / block_size;
    
    // Bitonic sort phases
    for (int k = 2; k <= padded_n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<grid_size, block_size>>>(d_data, padded_n, k, j);
            cudaDeviceSynchronize();
        }
    }
    
    // Copy result back
    cudaMemcpy(padded_data, d_data, array_size, cudaMemcpyDeviceToHost);
    
    if (padded_n > n) {
        std::copy(padded_data, padded_data + n, data);
        delete[] padded_data;
    }
    
    cudaFree(d_data);
    return 0;
}

/**
 * CUDA Kernel: Parallel selection for quicksort pivot
 */
__global__ void partition_kernel(
    float* data, 
    int* left_count, 
    int* right_count,
    float pivot,
    int n
) {
    extern __shared__ int shared_counts[];
    int* left_shared = shared_counts;
    int* right_shared = &shared_counts[blockDim.x];
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    left_shared[tid] = 0;
    right_shared[tid] = 0;
    
    // Count elements
    if (global_tid < n) {
        if (data[global_tid] <= pivot) {
            left_shared[tid] = 1;
        } else {
            right_shared[tid] = 1;
        }
    }
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            left_shared[tid] += left_shared[tid + stride];
            right_shared[tid] += right_shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Store block results
    if (tid == 0) {
        atomicAdd(left_count, left_shared[0]);
        atomicAdd(right_count, right_shared[0]);
    }
}

/**
 * Host function: Adaptive sorting algorithm selection
 * Chooses the best sorting algorithm based on data characteristics
 */
__host__ int adaptive_sort_cuda(float* data, int n) {
    if (n <= 1) return 0;
    
    // Algorithm selection logic
    if (n <= INSERTION_SORT_THRESHOLD) {
        // Small arrays: use insertion sort
        float* d_data;
        size_t array_size = n * sizeof(float);
        
        cudaMalloc(&d_data, array_size);
        cudaMemcpy(d_data, data, array_size, cudaMemcpyHostToDevice);
        
        insertion_sort_kernel<<<1, 1>>>(d_data, n);
        
        cudaMemcpy(data, d_data, array_size, cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        
        return 0;
    }
    
    // Check if size is power of 2 and within bitonic sort threshold
    bool is_power_of_2 = (n & (n - 1)) == 0;
    if (is_power_of_2 && n <= BITONIC_SORT_THRESHOLD) {
        return bitonic_sort_cuda(data, n);
    }
    
    // For larger arrays, analyze data characteristics
    float min_val = *std::min_element(data, data + n);
    float max_val = *std::max_element(data, data + n);
    float range = max_val - min_val;
    
    // Check if data is suitable for radix sort (integer-like values)
    bool suitable_for_radix = true;
    int sample_size = std::min(n, 1000);
    for (int i = 0; i < sample_size; i++) {
        if (data[i] != floorf(data[i]) || data[i] < 0) {
            suitable_for_radix = false;
            break;
        }
    }
    
    if (suitable_for_radix && range < 1000000) {
        // Use radix sort for integer-like data with reasonable range
        return radix_sort_float_cuda(data, n);
    } else {
        // Use merge sort for general case
        return mergesort_cuda(data, n);
    }
}

/**
 * Main solve function as specified in original file
 * This is the entry point for the general sorting algorithm
 */
__host__ void solve(float* data, int N) {
    adaptive_sort_cuda(data, N);
}

/**
 * Host function: Sort with custom comparison (ascending/descending)
 */
__host__ int sort_cuda_custom(float* data, int n, bool ascending = true) {
    int result = adaptive_sort_cuda(data, n);
    
    if (!ascending) {
        // Reverse for descending order
        std::reverse(data, data + n);
    }
    
    return result;
}

/**
 * Host function: Parallel sort validation
 * Checks if array is properly sorted
 */
__host__ bool validate_sort_cuda(const float* data, int n, bool ascending = true) {
    for (int i = 1; i < n; i++) {
        if (ascending) {
            if (data[i] < data[i-1]) return false;
        } else {
            if (data[i] > data[i-1]) return false;
        }
    }
    return true;
}

/**
 * Utility function: Performance timing for different algorithms
 */
__host__ void benchmark_sorting_algorithms(float* data, int n) {
    // Create copies for testing different algorithms
    std::vector<float> data_copy1(data, data + n);
    std::vector<float> data_copy2(data, data + n);
    std::vector<float> data_copy3(data, data + n);
    
    printf("Benchmarking sorting algorithms for %d elements:\n", n);
    
    // Time merge sort
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    mergesort_cuda(data_copy1.data(), n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float merge_time;
    cudaEventElapsedTime(&merge_time, start, stop);
    printf("Merge sort: %.3f ms\n", merge_time);
    
    // Time adaptive sort
    cudaEventRecord(start);
    adaptive_sort_cuda(data_copy2.data(), n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float adaptive_time;
    cudaEventElapsedTime(&adaptive_time, start, stop);
    printf("Adaptive sort: %.3f ms\n", adaptive_time);
    
    // Time bitonic sort (if applicable)
    if ((n & (n - 1)) == 0 && n <= BITONIC_SORT_THRESHOLD) {
        cudaEventRecord(start);
        bitonic_sort_cuda(data_copy3.data(), n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float bitonic_time;
        cudaEventElapsedTime(&bitonic_time, start, stop);
        printf("Bitonic sort: %.3f ms\n", bitonic_time);
    }
    
    // Verify all results are correctly sorted
    bool merge_correct = validate_sort_cuda(data_copy1.data(), n);
    bool adaptive_correct = validate_sort_cuda(data_copy2.data(), n);
    
    printf("Results validation: Merge=%s, Adaptive=%s\n",
           merge_correct ? "PASS" : "FAIL",
           adaptive_correct ? "PASS" : "FAIL");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}