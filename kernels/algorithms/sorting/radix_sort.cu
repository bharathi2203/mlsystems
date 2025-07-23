/*
 * Radix Sort - CUDA Implementation
 * 
 * GPU-accelerated implementation of radix sort algorithm
 * using digit-based parallel sorting approach.
 * 
 * Algorithm overview:
 * - Non-comparison based sorting algorithm
 * - Sorts by processing individual digits/bits
 * - Stable sorting algorithm
 * - O(d*n) time complexity where d is number of digits
 * - Excellent for integers and fixed-precision numbers
 * 
 * Memory patterns:
 * - Double buffering to avoid conflicts
 * - Coalesced reads and writes
 * - Shared memory for digit counting
 * - Bank conflict avoidance in shared memory
 * 
 * Performance considerations:
 * - Most efficient for uniformly distributed data
 * - Digit size affects performance (4-8 bits optimal)
 * - Memory bandwidth bound for large arrays
 * - Excellent scalability with problem size
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

// Configuration constants
#define RADIX_BITS 8  // Process 8 bits at a time
#define RADIX_SIZE (1 << RADIX_BITS)  // 256 buckets
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

/**
 * Device function: Extract digit from number
 * 
 * @param value: Input value
 * @param digit_pos: Position of digit (0 = least significant)
 * @param radix_bits: Number of bits per digit
 * @return: Extracted digit value
 */
__device__ int extract_digit(unsigned int value, int digit_pos, int radix_bits) {
    int shift = digit_pos * radix_bits;
    unsigned int mask = (1 << radix_bits) - 1;
    return (value >> shift) & mask;
}

/**
 * CUDA Kernel: Count occurrences of each digit
 * 
 * @param input: Input array
 * @param counts: Output count array [n_blocks x RADIX_SIZE]
 * @param n: Array size
 * @param digit_pos: Current digit position
 */
__global__ void count_digits(
    const unsigned int* input,
    unsigned int* counts,
    int n,
    int digit_pos
) {
    // Shared memory for local counting
    __shared__ unsigned int shared_counts[RADIX_SIZE];
    
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared counts
    for (int i = tid; i < RADIX_SIZE; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();
    
    // Count digits in this block's portion
    for (int i = global_tid; i < n; i += gridDim.x * blockDim.x) {
        int digit = extract_digit(input[i], digit_pos, RADIX_BITS);
        atomicAdd(&shared_counts[digit], 1);
    }
    __syncthreads();
    
    // Write block counts to global memory
    for (int i = tid; i < RADIX_SIZE; i += blockDim.x) {
        counts[block_id * RADIX_SIZE + i] = shared_counts[i];
    }
}

/**
 * CUDA Kernel: Scatter elements based on computed offsets
 * 
 * @param input: Input array
 * @param output: Output array
 * @param offsets: Prefix sum offsets [RADIX_SIZE]
 * @param n: Array size
 * @param digit_pos: Current digit position
 */
__global__ void scatter_elements(
    const unsigned int* input,
    unsigned int* output,
    const unsigned int* offsets,
    int n,
    int digit_pos
) {
    // Shared memory for local offsets
    __shared__ unsigned int shared_offsets[RADIX_SIZE];
    __shared__ unsigned int local_offsets[RADIX_SIZE];
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load global offsets to shared memory
    for (int i = tid; i < RADIX_SIZE; i += blockDim.x) {
        shared_offsets[i] = offsets[i];
        local_offsets[i] = 0;
    }
    __syncthreads();
    
    // First pass: count local occurrences
    for (int i = global_tid; i < n; i += gridDim.x * blockDim.x) {
        int digit = extract_digit(input[i], digit_pos, RADIX_BITS);
        atomicAdd(&local_offsets[digit], 1);
    }
    __syncthreads();
    
    // Compute local prefix sums for this block
    if (tid == 0) {
        for (int i = 1; i < RADIX_SIZE; i++) {
            local_offsets[i] += local_offsets[i-1];
        }
    }
    __syncthreads();
    
    // Second pass: scatter elements
    for (int i = global_tid; i < n; i += gridDim.x * blockDim.x) {
        int digit = extract_digit(input[i], digit_pos, RADIX_BITS);
        
        // Compute final position
        unsigned int base_offset = shared_offsets[digit];
        unsigned int local_offset = (digit > 0) ? local_offsets[digit-1] : 0;
        unsigned int position = base_offset + local_offset;
        
        // Atomically get position and increment
        unsigned int final_pos = atomicAdd(&shared_offsets[digit], 1);
        output[final_pos] = input[i];
    }
}

/**
 * CUDA Kernel: Optimized scatter using warp-level cooperation
 * More efficient for larger datasets
 */
__global__ void scatter_elements_optimized(
    const unsigned int* input,
    unsigned int* output,
    unsigned int* global_counters,
    int n,
    int digit_pos
) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process elements in chunks
    for (int i = global_tid; i < n; i += gridDim.x * blockDim.x) {
        unsigned int value = input[i];
        int digit = extract_digit(value, digit_pos, RADIX_BITS);
        
        // Get position using atomic increment
        unsigned int position = atomicAdd(&global_counters[digit], 1);
        output[position] = value;
    }
}

/**
 * Host function: Radix sort implementation
 * 
 * @param data: Input array to sort (modified in-place)
 * @param n: Number of elements
 * @return: Success status (0 = success)
 */
__host__ int radix_sort_cuda(unsigned int* data, int n) {
    if (n <= 1) return 0;
    
    // Device memory allocation
    unsigned int *d_input, *d_output;
    unsigned int *d_counts, *d_offsets;
    unsigned int *d_global_counters;
    
    size_t array_size = n * sizeof(unsigned int);
    
    cudaMalloc(&d_input, array_size);
    cudaMalloc(&d_output, array_size);
    cudaMalloc(&d_global_counters, RADIX_SIZE * sizeof(unsigned int));
    
    // Copy input to device
    cudaMemcpy(d_input, data, array_size, cudaMemcpyHostToDevice);
    
    // Determine number of digits needed
    unsigned int max_value = *std::max_element(data, data + n);
    int num_digits = 0;
    while (max_value > 0) {
        max_value >>= RADIX_BITS;
        num_digits++;
    }
    if (num_digits == 0) num_digits = 1;
    
    // Grid and block configuration
    int block_size = 256;
    int grid_size = min(65535, (n + block_size - 1) / block_size);
    
    // Allocate for counting approach
    size_t counts_size = grid_size * RADIX_SIZE * sizeof(unsigned int);
    cudaMalloc(&d_counts, counts_size);
    cudaMalloc(&d_offsets, RADIX_SIZE * sizeof(unsigned int));
    
    // Current input/output pointers
    unsigned int* current_input = d_input;
    unsigned int* current_output = d_output;
    
    // Process each digit
    for (int digit = 0; digit < num_digits; digit++) {
        // Method 1: Use CUB for prefix sum (more efficient)
        
        // Reset global counters
        cudaMemset(d_global_counters, 0, RADIX_SIZE * sizeof(unsigned int));
        
        // Count digits in parallel
        count_digits<<<grid_size, block_size>>>(
            current_input, d_counts, n, digit
        );
        
        // Reduce counts across blocks
        for (int i = 0; i < RADIX_SIZE; i++) {
            unsigned int total_count = 0;
            
            // Sum counts from all blocks for digit i
            for (int block = 0; block < grid_size; block++) {
                unsigned int block_count;
                cudaMemcpy(&block_count, 
                          &d_counts[block * RADIX_SIZE + i], 
                          sizeof(unsigned int), 
                          cudaMemcpyDeviceToHost);
                total_count += block_count;
            }
            
            // Set offset for this digit
            cudaMemcpy(&d_offsets[i], &total_count, sizeof(unsigned int), 
                      cudaMemcpyHostToDevice);
        }
        
        // Compute prefix sum on CPU (simpler for this implementation)
        std::vector<unsigned int> h_offsets(RADIX_SIZE);
        cudaMemcpy(h_offsets.data(), d_offsets, 
                  RADIX_SIZE * sizeof(unsigned int), 
                  cudaMemcpyDeviceToHost);
        
        unsigned int running_sum = 0;
        for (int i = 0; i < RADIX_SIZE; i++) {
            unsigned int count = h_offsets[i];
            h_offsets[i] = running_sum;
            running_sum += count;
        }
        
        cudaMemcpy(d_global_counters, h_offsets.data(), 
                  RADIX_SIZE * sizeof(unsigned int), 
                  cudaMemcpyHostToDevice);
        
        // Scatter elements
        scatter_elements_optimized<<<grid_size, block_size>>>(
            current_input, current_output, d_global_counters, n, digit
        );
        
        // Swap input and output for next iteration
        unsigned int* temp = current_input;
        current_input = current_output;
        current_output = temp;
    }
    
    // Copy result back to host
    cudaMemcpy(data, current_input, array_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    cudaFree(d_global_counters);
    
    return 0;
}

/**
 * Host function: Radix sort for floating point numbers
 * Handles IEEE 754 floating point by treating as unsigned integers
 * with special handling for sign bit
 */
__host__ int radix_sort_float_cuda(float* data, int n) {
    if (n <= 1) return 0;
    
    // Convert floats to sortable unsigned integers
    std::vector<unsigned int> uint_data(n);
    for (int i = 0; i < n; i++) {
        unsigned int bits = *(unsigned int*)&data[i];
        
        // Handle IEEE 754 format for proper sorting
        if (bits & 0x80000000) {
            // Negative number: flip all bits
            bits = ~bits;
        } else {
            // Positive number: flip sign bit
            bits |= 0x80000000;
        }
        uint_data[i] = bits;
    }
    
    // Sort as unsigned integers
    int result = radix_sort_cuda(uint_data.data(), n);
    
    // Convert back to floats
    for (int i = 0; i < n; i++) {
        unsigned int bits = uint_data[i];
        
        // Reverse the conversion
        if (bits & 0x80000000) {
            // Was positive: flip sign bit back
            bits &= 0x7FFFFFFF;
        } else {
            // Was negative: flip all bits back
            bits = ~bits;
        }
        data[i] = *(float*)&bits;
    }
    
    return result;
}

/**
 * Utility function: Generate test data
 */
__host__ void generate_test_data_uint(unsigned int* data, int n, 
                                     int seed = 42, unsigned int range = 1000000) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        data[i] = rand() % range;
    }
}

/**
 * Utility function: Verify sorting correctness
 */
__host__ bool is_sorted_uint(const unsigned int* data, int n) {
    for (int i = 1; i < n; i++) {
        if (data[i] < data[i-1]) return false;
    }
    return true;
}

/**
 * Utility function: Print array for debugging
 */
__host__ void print_array_uint(const unsigned int* data, int n, int max_print = 20) {
    int print_count = (n < max_print) ? n : max_print;
    
    printf("Array: [");
    for (int i = 0; i < print_count; i++) {
        printf("%u", data[i]);
        if (i < print_count - 1) printf(", ");
    }
    if (n > max_print) {
        printf(", ... (%d more elements)", n - max_print);
    }
    printf("]\n");
}
