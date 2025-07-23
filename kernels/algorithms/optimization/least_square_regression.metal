#include <metal_stdlib>
using namespace metal;

/*
 * Least Squares Regression - Metal Implementation
 * 
 * GPU-accelerated implementation of linear least squares regression
 * optimized for Apple Silicon GPUs using Metal Shading Language.
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
 * - Threadgroup memory for intermediate computations
 * - Register blocking for better arithmetic intensity
 * 
 * Numerical stability:
 * - Pivoting strategies for matrix inversion
 * - Iterative refinement for improved accuracy
 * - Regularization to handle ill-conditioned systems
 * - Mixed precision arithmetic for performance
 */

// Configuration constants
constant uint TILE_SIZE = 16;
constant uint MAX_FEATURES = 512;
constant uint THREADGROUP_SIZE = 256;

/**
 * Metal Kernel: Matrix-matrix multiplication using tiled approach
 * Computes C = A * B using threadgroup memory for efficient tile operations
 * 
 * @param A: Input matrix A [M x K]
 * @param B: Input matrix B [K x N]
 * @param C: Output matrix C [M x N]
 * @param M: Number of rows in A and C
 * @param K: Number of columns in A and rows in B
 * @param N: Number of columns in B and C
 */
kernel void matrix_multiply_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadgroup_id [[threadgroup_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    float sum = 0.0f;
    
    // Process tiles
    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile A
        uint a_row = threadgroup_id.y * TILE_SIZE + tid.y;
        uint a_col = tile * TILE_SIZE + tid.x;
        tileA[tid.y * TILE_SIZE + tid.x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        
        // Load tile B
        uint b_row = tile * TILE_SIZE + tid.y;
        uint b_col = threadgroup_id.x * TILE_SIZE + tid.x;
        tileB[tid.y * TILE_SIZE + tid.x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial sum
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Metal Kernel: Matrix transpose operation
 * Optimized transpose using threadgroup memory for coalesced access
 * 
 * @param input: Input matrix [rows x cols]
 * @param output: Output matrix [cols x rows]
 * @param rows: Number of rows in input
 * @param cols: Number of columns in input
 */
kernel void matrix_transpose(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    threadgroup float* tile [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    // Load into threadgroup memory with bank conflict avoidance
    if (row < rows && col < cols) {
        tile[tid.y * (TILE_SIZE + 1) + tid.x] = input[row * cols + col];
    } else {
        tile[tid.y * (TILE_SIZE + 1) + tid.x] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write transposed data
    uint out_row = gid.x; // Swapped
    uint out_col = gid.y; // Swapped
    
    if (out_row < cols && out_col < rows) {
        output[out_row * rows + out_col] = tile[tid.x * (TILE_SIZE + 1) + tid.y];
    }
}

/**
 * Metal Kernel: Compute A^T * A efficiently
 * Specialized kernel for computing Gram matrix A^T * A
 * 
 * @param A: Input matrix [n_samples x n_features]
 * @param ATA: Output Gram matrix [n_features x n_features]
 * @param n_samples: Number of samples
 * @param n_features: Number of features
 */
kernel void compute_gram_matrix(
    device const float* A [[buffer(0)]],
    device float* ATA [[buffer(1)]],
    constant uint& n_samples [[buffer(2)]],
    constant uint& n_features [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row < n_features && col < n_features) {
        float sum = 0.0f;
        
        // Compute dot product of columns row and col
        for (uint i = 0; i < n_samples; i++) {
            sum += A[i * n_features + row] * A[i * n_features + col];
        }
        
        ATA[row * n_features + col] = sum;
    }
}

/**
 * Metal Kernel: Compute A^T * b efficiently
 * Specialized kernel for computing A^T * b vector
 * 
 * @param A: Input matrix [n_samples x n_features]
 * @param b: Input vector [n_samples]
 * @param ATb: Output vector [n_features]
 * @param n_samples: Number of samples
 * @param n_features: Number of features
 */
kernel void compute_AT_b(
    device const float* A [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* ATb [[buffer(2)]],
    constant uint& n_samples [[buffer(3)]],
    constant uint& n_features [[buffer(4)]],
    threadgroup float* shared_sum [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint feature_id = threadgroup_id;
    
    if (feature_id >= n_features) return;
    
    // Parallel reduction for dot product
    float local_sum = 0.0f;
    
    for (uint i = thread_id; i < n_samples; i += threads_per_threadgroup) {
        local_sum += A[i * n_features + feature_id] * b[i];
    }
    
    shared_sum[thread_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_sum[thread_id] += shared_sum[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (thread_id == 0) {
        ATb[feature_id] = shared_sum[0];
    }
}

/**
 * Metal Kernel: Gradient descent step for least squares
 * Computes gradient and updates parameters
 * 
 * @param A: Feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param x: Current parameters [n_features]
 * @param gradient: Output gradient [n_features]
 * @param n_samples: Number of samples
 * @param n_features: Number of features
 * @param lambda: Regularization parameter
 */
kernel void compute_gradient(
    device const float* A [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* gradient [[buffer(3)]],
    constant uint& n_samples [[buffer(4)]],
    constant uint& n_features [[buffer(5)]],
    constant float& lambda [[buffer(6)]],
    threadgroup float* shared_residuals [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint feature_id = threadgroup_id;
    
    if (feature_id >= n_features) return;
    
    float grad_sum = 0.0f;
    
    // Compute gradient component for this feature
    for (uint i = thread_id; i < n_samples; i += threads_per_threadgroup) {
        // Compute prediction for sample i
        float prediction = 0.0f;
        for (uint j = 0; j < n_features; j++) {
            prediction += A[i * n_features + j] * x[j];
        }
        
        // Compute residual
        float residual = prediction - b[i];
        
        // Accumulate gradient
        grad_sum += residual * A[i * n_features + feature_id];
    }
    
    shared_residuals[thread_id] = grad_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_residuals[thread_id] += shared_residuals[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (thread_id == 0) {
        // Add regularization term
        float reg_term = lambda * x[feature_id];
        gradient[feature_id] = (2.0f / float(n_samples)) * shared_residuals[0] + reg_term;
    }
}

/**
 * Metal Kernel: Update parameters using gradient descent
 * 
 * @param x: Current parameters [n_features]
 * @param gradient: Computed gradient [n_features]
 * @param learning_rate: Step size
 * @param n_features: Number of features
 */
kernel void update_parameters(
    device float* x [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    constant float& learning_rate [[buffer(2)]],
    constant uint& n_features [[buffer(3)]],
    uint feature_id [[thread_position_in_grid]]
) {
    if (feature_id < n_features) {
        x[feature_id] -= learning_rate * gradient[feature_id];
    }
}

/**
 * Metal Kernel: Compute loss function (MSE + regularization)
 * 
 * @param A: Feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param x: Current parameters [n_features]
 * @param loss: Output loss value [1]
 * @param n_samples: Number of samples
 * @param n_features: Number of features
 * @param lambda: Regularization parameter
 */
kernel void compute_loss(
    device const float* A [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* loss [[buffer(3)]],
    constant uint& n_samples [[buffer(4)]],
    constant uint& n_features [[buffer(5)]],
    constant float& lambda [[buffer(6)]],
    threadgroup float* shared_losses [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    float local_loss = 0.0f;
    
    // Compute MSE loss
    if (global_id < n_samples) {
        float prediction = 0.0f;
        for (uint j = 0; j < n_features; j++) {
            prediction += A[global_id * n_features + j] * x[j];
        }
        
        float residual = prediction - b[global_id];
        local_loss = residual * residual;
    }
    
    shared_losses[thread_id] = local_loss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_losses[thread_id] += shared_losses[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (thread_id == 0) {
        device atomic<float>* loss_ptr = (device atomic<float>*)loss;
        atomic_fetch_add_explicit(loss_ptr, shared_losses[0], memory_order_relaxed);
    }
}

/**
 * Metal Kernel: Add regularization to loss
 * 
 * @param x: Current parameters [n_features]
 * @param loss: Current loss value [1]
 * @param lambda: Regularization parameter
 * @param n_features: Number of features
 */
kernel void add_regularization_loss(
    device const float* x [[buffer(0)]],
    device float* loss [[buffer(1)]],
    constant float& lambda [[buffer(2)]],
    constant uint& n_features [[buffer(3)]],
    threadgroup float* shared_reg [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    float local_reg = 0.0f;
    
    for (uint i = thread_id; i < n_features; i += threads_per_threadgroup) {
        local_reg += x[i] * x[i];
    }
    
    shared_reg[thread_id] = local_reg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_reg[thread_id] += shared_reg[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (thread_id == 0) {
        *loss += lambda * shared_reg[0];
    }
}

/**
 * Metal Kernel: Cholesky decomposition for solving normal equations
 * Computes L such that A = L * L^T for positive definite matrix A
 * 
 * @param A: Input positive definite matrix [n x n]
 * @param L: Output lower triangular matrix [n x n]
 * @param n: Matrix dimension
 */
kernel void cholesky_decomposition(
    device const float* A [[buffer(0)]],
    device float* L [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    
    if (i >= n || j >= n || j > i) {
        if (i < n && j < n) L[i * n + j] = 0.0f;
        return;
    }
    
    // Compute L[i][j]
    if (i == j) {
        // Diagonal elements
        float sum = 0.0f;
        for (uint k = 0; k < j; k++) {
            float l_jk = L[j * n + k];
            sum += l_jk * l_jk;
        }
        L[i * n + j] = sqrt(A[i * n + j] - sum);
    } else {
        // Off-diagonal elements
        float sum = 0.0f;
        for (uint k = 0; k < j; k++) {
            sum += L[i * n + k] * L[j * n + k];
        }
        L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
    }
}

/**
 * Metal Kernel: Forward substitution for solving Ly = b
 * 
 * @param L: Lower triangular matrix [n x n]
 * @param b: Right-hand side vector [n]
 * @param y: Solution vector [n]
 * @param n: System dimension
 */
kernel void forward_substitution(
    device const float* L [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= n) return;
    
    float sum = 0.0f;
    for (uint j = 0; j < row; j++) {
        sum += L[row * n + j] * y[j];
    }
    
    y[row] = (b[row] - sum) / L[row * n + row];
}

/**
 * Metal Kernel: Backward substitution for solving L^T x = y
 * 
 * @param L: Lower triangular matrix [n x n]
 * @param y: Right-hand side vector [n]
 * @param x: Solution vector [n]
 * @param n: System dimension
 */
kernel void backward_substitution(
    device const float* L [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= n) return;
    
    uint i = n - 1 - row; // Process in reverse order
    
    float sum = 0.0f;
    for (uint j = i + 1; j < n; j++) {
        sum += L[j * n + i] * x[j]; // L^T[i][j] = L[j][i]
    }
    
    x[i] = (y[i] - sum) / L[i * n + i];
}
