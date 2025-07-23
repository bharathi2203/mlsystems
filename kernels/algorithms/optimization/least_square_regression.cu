/*
 * Least Squares Regression - CUDA Implementation
 * 
 * GPU-accelerated implementation of linear least squares regression
 * for supervised learning and predictive modeling.
 * 
 * Mathematical foundation:
 * - Solves: min ||Ax - b||² where A is feature matrix, x is parameters, b is targets
 * - Normal equation: x = (A^T A)^(-1) A^T b
 * - Gradient descent: x_new = x_old - α * ∇f(x)
 * - Regularized version: min ||Ax - b||² + λ||x||² (Ridge regression)
 * 
 * Algorithm approaches:
 * 1. Normal equations (direct matrix inversion)
 * 2. Gradient descent (iterative optimization)
 * 3. Conjugate gradient (efficient for large sparse systems)
 * 4. QR decomposition (numerically stable)
 * 5. SVD decomposition (most robust but expensive)
 * 
 * Memory patterns:
 * - Tiled matrix operations for cache efficiency
 * - Coalesced access to feature and target data
 * - Shared memory for intermediate computations
 * - Register blocking for better arithmetic intensity
 * 
 * Numerical stability:
 * - Pivoting strategies for matrix inversion
 * - Iterative refinement for improved accuracy
 * - Regularization to handle ill-conditioned systems
 * - Mixed precision arithmetic for performance
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <float.h>
#include <math.h>

// Configuration constants
#define TILE_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

/**
 * CUDA Kernel: Matrix-matrix multiplication using shared memory tiling
 * Computes C = A^T * A for normal equations
 * 
 * @param A: Input feature matrix [n_samples x n_features]
 * @param ATA: Output A^T*A matrix [n_features x n_features]
 * @param n_samples: Number of training samples
 * @param n_features: Number of features
 */
__global__ void compute_ata_matrix(
    const float* A,
    float* ATA,
    int n_samples,
    int n_features
) {
    // Shared memory for tiling
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_AT[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Tile across the common dimension (n_samples)
    for (int tile = 0; tile < (n_samples + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A and A^T
        int sample_idx = tile * TILE_SIZE + threadIdx.x;
        int feature_idx_row = tile * TILE_SIZE + threadIdx.y;
        
        // Load A tile (features x samples)
        if (row < n_features && sample_idx < n_samples) {
            tile_A[threadIdx.y][threadIdx.x] = A[sample_idx * n_features + row];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load A^T tile (samples x features)  
        if (col < n_features && feature_idx_row < n_samples) {
            tile_AT[threadIdx.y][threadIdx.x] = A[feature_idx_row * n_features + col];
        } else {
            tile_AT[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_AT[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < n_features && col < n_features) {
        ATA[row * n_features + col] = sum;
    }
}

/**
 * CUDA Kernel: Compute A^T * b for normal equations
 * 
 * @param A: Input feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param ATb: Output A^T*b vector [n_features]
 * @param n_samples: Number of training samples
 * @param n_features: Number of features
 */
__global__ void compute_atb_vector(
    const float* A,
    const float* b,
    float* ATb,
    int n_samples,
    int n_features
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feature_idx < n_features) {
        float sum = 0.0f;
        
        // Compute dot product of feature column with target vector
        for (int sample = 0; sample < n_samples; sample++) {
            sum += A[sample * n_features + feature_idx] * b[sample];
        }
        
        ATb[feature_idx] = sum;
    }
}

/**
 * CUDA Kernel: Gradient descent step for least squares
 * Updates parameters: x_new = x_old - learning_rate * gradient
 * 
 * @param A: Feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param x: Current parameters [n_features]
 * @param x_new: Updated parameters [n_features]
 * @param learning_rate: Step size
 * @param lambda: Regularization parameter
 * @param n_samples: Number of training samples
 * @param n_features: Number of features
 */
__global__ void gradient_descent_step(
    const float* A,
    const float* b,
    const float* x,
    float* x_new,
    float learning_rate,
    float lambda,
    int n_samples,
    int n_features
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feature_idx < n_features) {
        float gradient = 0.0f;
        
        // Compute gradient: A^T * (A*x - b) + lambda*x
        for (int sample = 0; sample < n_samples; sample++) {
            // Compute prediction error for this sample
            float prediction = 0.0f;
            for (int f = 0; f < n_features; f++) {
                prediction += A[sample * n_features + f] * x[f];
            }
            float error = prediction - b[sample];
            
            // Accumulate gradient contribution
            gradient += A[sample * n_features + feature_idx] * error;
        }
        
        // Add regularization term
        gradient = (gradient / n_samples) + lambda * x[feature_idx];
        
        // Update parameter
        x_new[feature_idx] = x[feature_idx] - learning_rate * gradient;
    }
}

/**
 * CUDA Kernel: Optimized gradient computation using shared memory
 * More efficient version for larger datasets
 */
__global__ void compute_gradient_optimized(
    const float* A,
    const float* b,
    const float* x,
    float* gradient,
    int n_samples,
    int n_features
) {
    extern __shared__ float shared_data[];
    float* shared_predictions = shared_data;
    
    int tid = threadIdx.x;
    int feature_idx = blockIdx.x;
    
    // Compute predictions in shared memory
    for (int sample_batch = 0; sample_batch < n_samples; sample_batch += blockDim.x) {
        int sample_idx = sample_batch + tid;
        
        if (sample_idx < n_samples) {
            // Compute prediction for this sample
            float prediction = 0.0f;
            for (int f = 0; f < n_features; f++) {
                prediction += A[sample_idx * n_features + f] * x[f];
            }
            shared_predictions[tid] = prediction - b[sample_idx];
        } else {
            shared_predictions[tid] = 0.0f;
        }
        __syncthreads();
        
        // Accumulate gradient contributions
        if (feature_idx < n_features) {
            float grad_sum = 0.0f;
            for (int i = 0; i < blockDim.x && (sample_batch + i) < n_samples; i++) {
                int sample = sample_batch + i;
                grad_sum += A[sample * n_features + feature_idx] * shared_predictions[i];
            }
            
            // Use atomic add to accumulate across thread blocks
            atomicAdd(&gradient[feature_idx], grad_sum);
        }
        __syncthreads();
    }
}

/**
 * CUDA Kernel: Compute loss function (Mean Squared Error + Regularization)
 * 
 * @param A: Feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param x: Current parameters [n_features]
 * @param loss: Output loss value [1]
 * @param lambda: Regularization parameter
 * @param n_samples: Number of training samples
 * @param n_features: Number of features
 */
__global__ void compute_loss(
    const float* A,
    const float* b,
    const float* x,
    float* loss,
    float lambda,
    int n_samples,
    int n_features
) {
    extern __shared__ float shared_loss[];
    
    int tid = threadIdx.x;
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_loss = 0.0f;
    
    if (sample_idx < n_samples) {
        // Compute prediction
        float prediction = 0.0f;
        for (int f = 0; f < n_features; f++) {
            prediction += A[sample_idx * n_features + f] * x[f];
        }
        
        // Compute squared error
        float error = prediction - b[sample_idx];
        local_loss = error * error;
    }
    
    shared_loss[tid] = local_loss;
    __syncthreads();
    
    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_loss[tid] += shared_loss[tid + stride];
        }
        __syncthreads();
    }
    
    // Add regularization term (only for thread 0 of first block)
    if (blockIdx.x == 0 && tid == 0) {
        float reg_term = 0.0f;
        for (int f = 0; f < n_features; f++) {
            reg_term += x[f] * x[f];
        }
        shared_loss[0] += lambda * reg_term;
    }
    
    // Store block result
    if (tid == 0) {
        atomicAdd(loss, shared_loss[0]);
    }
}

/**
 * Host function: Solve least squares using normal equations
 * 
 * @param A: Feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param x: Output solution [n_features]
 * @param n_samples: Number of training samples
 * @param n_features: Number of features
 * @param lambda: Regularization parameter
 * @return: Success status
 */
__host__ int solve_normal_equations_cuda(
    const float* A,
    const float* b,
    float* x,
    int n_samples,
    int n_features,
    float lambda = 0.0f
) {
    // Device memory allocation
    float *d_A, *d_b, *d_x, *d_ATA, *d_ATb;
    
    size_t A_size = n_samples * n_features * sizeof(float);
    size_t b_size = n_samples * sizeof(float);
    size_t x_size = n_features * sizeof(float);
    size_t ATA_size = n_features * n_features * sizeof(float);
    
    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_x, x_size);
    cudaMalloc(&d_ATA, ATA_size);
    cudaMalloc(&d_ATb, x_size);
    
    // Copy input data
    cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice);
    
    // Grid and block configuration
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size_ata((n_features + TILE_SIZE - 1) / TILE_SIZE,
                       (n_features + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_size_atb((n_features + 255) / 256);
    
    // Step 1: Compute A^T * A
    compute_ata_matrix<<<grid_size_ata, block_size>>>(
        d_A, d_ATA, n_samples, n_features
    );
    
    // Add regularization to diagonal if lambda > 0
    if (lambda > 0.0f) {
        // Simple kernel to add lambda*I to diagonal
        int block_size_reg = min(256, n_features);
        int grid_size_reg = (n_features + block_size_reg - 1) / block_size_reg;
        
        auto add_regularization = [=] __global__ (float* ATA, int n_features, float lambda) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n_features) {
                ATA[idx * n_features + idx] += lambda;
            }
        };
        
        add_regularization<<<grid_size_reg, block_size_reg>>>(d_ATA, n_features, lambda);
    }
    
    // Step 2: Compute A^T * b
    compute_atb_vector<<<grid_size_atb, 256>>>(
        d_A, d_b, d_ATb, n_samples, n_features
    );
    
    // Step 3: Solve ATA * x = ATb using CUSOLVER
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);
    
    // LU decomposition workspace
    int* d_info;
    float* d_workspace;
    int workspace_size;
    
    cudaMalloc(&d_info, sizeof(int));
    
    // Query workspace size
    cusolverDnSgetrf_bufferSize(cusolver_handle, n_features, n_features, 
                               d_ATA, n_features, &workspace_size);
    
    cudaMalloc(&d_workspace, workspace_size * sizeof(float));
    
    // LU factorization
    int* d_pivot;
    cudaMalloc(&d_pivot, n_features * sizeof(int));
    
    cusolverDnSgetrf(cusolver_handle, n_features, n_features,
                    d_ATA, n_features, d_workspace, d_pivot, d_info);
    
    // Solve linear system
    cudaMemcpy(d_x, d_ATb, x_size, cudaMemcpyDeviceToDevice);
    
    cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N, n_features, 1,
                    d_ATA, n_features, d_pivot, d_x, n_features, d_info);
    
    // Copy result back
    cudaMemcpy(x, d_x, x_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cusolverDnDestroy(cusolver_handle);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_ATA);
    cudaFree(d_ATb);
    cudaFree(d_info);
    cudaFree(d_workspace);
    cudaFree(d_pivot);
    
    return 0;
}

/**
 * Host function: Solve least squares using gradient descent
 * 
 * @param A: Feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param x: Initial and output solution [n_features]
 * @param n_samples: Number of training samples
 * @param n_features: Number of features
 * @param learning_rate: Step size for gradient descent
 * @param lambda: Regularization parameter
 * @param max_iterations: Maximum number of iterations
 * @param tolerance: Convergence tolerance
 * @return: Number of iterations performed
 */
__host__ int solve_gradient_descent_cuda(
    const float* A,
    const float* b,
    float* x,
    int n_samples,
    int n_features,
    float learning_rate = 0.01f,
    float lambda = 0.0f,
    int max_iterations = 1000,
    float tolerance = 1e-6f
) {
    // Device memory allocation
    float *d_A, *d_b, *d_x, *d_x_new, *d_loss;
    
    size_t A_size = n_samples * n_features * sizeof(float);
    size_t b_size = n_samples * sizeof(float);
    size_t x_size = n_features * sizeof(float);
    
    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_x, x_size);
    cudaMalloc(&d_x_new, x_size);
    cudaMalloc(&d_loss, sizeof(float));
    
    // Copy input data
    cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);
    
    // Grid and block configuration
    int block_size = min(256, n_features);
    int grid_size = (n_features + block_size - 1) / block_size;
    
    int block_size_loss = 256;
    int grid_size_loss = (n_samples + block_size_loss - 1) / block_size_loss;
    size_t shared_mem_loss = block_size_loss * sizeof(float);
    
    float prev_loss = FLT_MAX;
    int iteration = 0;
    
    // Gradient descent loop
    for (iteration = 0; iteration < max_iterations; iteration++) {
        // Gradient descent step
        gradient_descent_step<<<grid_size, block_size>>>(
            d_A, d_b, d_x, d_x_new, learning_rate, lambda,
            n_samples, n_features
        );
        
        // Update parameters
        cudaMemcpy(d_x, d_x_new, x_size, cudaMemcpyDeviceToDevice);
        
        // Check convergence every 10 iterations
        if (iteration % 10 == 0) {
            // Reset loss
            cudaMemset(d_loss, 0, sizeof(float));
            
            // Compute current loss
            compute_loss<<<grid_size_loss, block_size_loss, shared_mem_loss>>>(
                d_A, d_b, d_x, d_loss, lambda, n_samples, n_features
            );
            
            float current_loss;
            cudaMemcpy(&current_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            current_loss /= n_samples; // Average loss
            
            // Check convergence
            if (abs(current_loss - prev_loss) < tolerance) {
                break;
            }
            prev_loss = current_loss;
        }
    }
    
    // Copy result back
    cudaMemcpy(x, d_x, x_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_x_new);
    cudaFree(d_loss);
    
    return iteration;
}
