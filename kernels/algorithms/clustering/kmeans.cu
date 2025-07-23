/*
 * K-Means Clustering Algorithm - CUDA Implementation
 * 
 * GPU-accelerated implementation of the k-means clustering algorithm
 * for unsupervised learning and data analysis.
 * 
 * Key concepts:
 * - Iterative centroid optimization
 * - Distance computation parallelization
 * - Cluster assignment updates
 * - Convergence detection
 * 
 * Algorithm:
 * 1. Initialize k centroids randomly or with k-means++
 * 2. Assign each point to nearest centroid (parallel distance computation)
 * 3. Update centroids as mean of assigned points (parallel reduction)
 * 4. Repeat until convergence or max iterations
 * 
 * Distance metrics:
 * - Euclidean distance (L2) - primary implementation
 * - Manhattan distance (L1) - alternative
 * - Cosine similarity - for normalized data
 * 
 * Memory access patterns:
 * - Points: coalesced reads across thread blocks
 * - Centroids: cached in shared memory
 * - Assignments: coalesced writes
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

// Configuration constants
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_SHARED_MEMORY_PER_BLOCK 48000
#define WARP_SIZE 32

/**
 * CUDA Kernel: Compute distances and assign points to nearest centroids
 * 
 * Each thread processes one data point and computes distances to all centroids.
 * Uses shared memory to cache centroids for efficient access.
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Current centroids [n_clusters x n_dims] 
 * @param assignments: Output cluster assignments [n_points]
 * @param distances: Output minimum distances [n_points]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 */
__global__ void kmeans_assign_points(
    const float* points,
    const float* centroids, 
    int* assignments,
    float* distances,
    int n_points,
    int n_dims,
    int n_clusters
) {
    // Shared memory for centroids (limited by available shared memory)
    extern __shared__ float shared_centroids[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Load centroids into shared memory cooperatively
    int centroids_size = n_clusters * n_dims;
    for (int i = local_tid; i < centroids_size; i += blockDim.x) {
        if (i < centroids_size) {
            shared_centroids[i] = centroids[i];
        }
    }
    __syncthreads();
    
    if (tid < n_points) {
        float min_distance = FLT_MAX;
        int best_cluster = 0;
        
        // Compute distance to each centroid
        for (int c = 0; c < n_clusters; c++) {
            float distance = 0.0f;
            
            // Euclidean distance computation
            for (int d = 0; d < n_dims; d++) {
                float diff = points[tid * n_dims + d] - shared_centroids[c * n_dims + d];
                distance += diff * diff;
            }
            
            // Update minimum distance and best cluster
            if (distance < min_distance) {
                min_distance = distance;
                best_cluster = c;
            }
        }
        
        assignments[tid] = best_cluster;
        distances[tid] = sqrtf(min_distance);
    }
}

/**
 * CUDA Kernel: Update centroids based on point assignments
 * 
 * Uses atomic operations to safely accumulate points assigned to each centroid.
 * Each thread processes one data point and contributes to its assigned centroid.
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param assignments: Cluster assignments [n_points]
 * @param new_centroids: Output updated centroids [n_clusters x n_dims]
 * @param cluster_counts: Output points per cluster [n_clusters]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions  
 * @param n_clusters: Number of clusters
 */
__global__ void kmeans_update_centroids(
    const float* points,
    const int* assignments,
    float* new_centroids,
    int* cluster_counts,
    int n_points,
    int n_dims,
    int n_clusters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n_points) {
        int cluster_id = assignments[tid];
        
        // Atomically increment cluster count
        atomicAdd(&cluster_counts[cluster_id], 1);
        
        // Atomically add point coordinates to centroid sum
        for (int d = 0; d < n_dims; d++) {
            atomicAdd(&new_centroids[cluster_id * n_dims + d], 
                     points[tid * n_dims + d]);
        }
    }
}

/**
 * CUDA Kernel: Finalize centroid computation by dividing by cluster counts
 * 
 * @param centroids: Centroid sums to be normalized [n_clusters x n_dims]
 * @param cluster_counts: Number of points per cluster [n_clusters]
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 */
__global__ void kmeans_finalize_centroids(
    float* centroids,
    const int* cluster_counts,
    int n_dims,
    int n_clusters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cluster_id = tid / n_dims;
    int dim_id = tid % n_dims;
    
    if (cluster_id < n_clusters && dim_id < n_dims) {
        int count = cluster_counts[cluster_id];
        if (count > 0) {
            centroids[tid] /= count;
        }
    }
}

/**
 * Host function: K-means clustering with CUDA acceleration
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Initial centroids [n_clusters x n_dims], updated in-place
 * @param assignments: Output cluster assignments [n_points]  
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 * @param max_iterations: Maximum number of iterations
 * @param tolerance: Convergence tolerance for centroid movement
 * @return: Number of iterations performed
 */
__host__ int kmeans_cuda(
    const float* points,
    float* centroids,
    int* assignments,
    int n_points,
    int n_dims, 
    int n_clusters,
    int max_iterations = 100,
    float tolerance = 1e-4f
) {
    // Device memory allocation
    float *d_points, *d_centroids, *d_new_centroids, *d_distances;
    int *d_assignments, *d_cluster_counts;
    
    size_t points_size = n_points * n_dims * sizeof(float);
    size_t centroids_size = n_clusters * n_dims * sizeof(float);
    size_t assignments_size = n_points * sizeof(int);
    size_t distances_size = n_points * sizeof(float);
    size_t counts_size = n_clusters * sizeof(int);
    
    cudaMalloc(&d_points, points_size);
    cudaMalloc(&d_centroids, centroids_size);
    cudaMalloc(&d_new_centroids, centroids_size);
    cudaMalloc(&d_assignments, assignments_size);
    cudaMalloc(&d_distances, distances_size);
    cudaMalloc(&d_cluster_counts, counts_size);
    
    // Copy input data to device
    cudaMemcpy(d_points, points, points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, centroids_size, cudaMemcpyHostToDevice);
    
    // Grid and block configuration
    dim3 block_size(256);
    dim3 grid_size_points((n_points + block_size.x - 1) / block_size.x);
    dim3 grid_size_centroids((n_clusters * n_dims + block_size.x - 1) / block_size.x);
    
    // Shared memory size for centroids
    size_t shared_mem_size = min(centroids_size, (size_t)MAX_SHARED_MEMORY_PER_BLOCK);
    
    int iteration = 0;
    float convergence_error = tolerance + 1.0f;
    
    // Allocate host memory for convergence checking
    float* h_old_centroids = new float[n_clusters * n_dims];
    
    // Main k-means iteration loop
    while (iteration < max_iterations && convergence_error > tolerance) {
        // Save current centroids for convergence checking
        cudaMemcpy(h_old_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost);
        
        // Step 1: Assign points to nearest centroids
        kmeans_assign_points<<<grid_size_points, block_size, shared_mem_size>>>(
            d_points, d_centroids, d_assignments, d_distances,
            n_points, n_dims, n_clusters
        );
        
        // Step 2: Reset centroid accumulators
        cudaMemset(d_new_centroids, 0, centroids_size);
        cudaMemset(d_cluster_counts, 0, counts_size);
        
        // Step 3: Accumulate new centroids
        kmeans_update_centroids<<<grid_size_points, block_size>>>(
            d_points, d_assignments, d_new_centroids, d_cluster_counts,
            n_points, n_dims, n_clusters
        );
        
        // Step 4: Finalize centroid computation
        kmeans_finalize_centroids<<<grid_size_centroids, block_size>>>(
            d_new_centroids, d_cluster_counts, n_dims, n_clusters
        );
        
        // Copy new centroids back
        cudaMemcpy(d_centroids, d_new_centroids, centroids_size, cudaMemcpyDeviceToDevice);
        
        // Check convergence
        if (iteration % 5 == 0) { // Check every 5 iterations for efficiency
            float* h_new_centroids = new float[n_clusters * n_dims];
            cudaMemcpy(h_new_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost);
            
            convergence_error = 0.0f;
            for (int i = 0; i < n_clusters * n_dims; i++) {
                float diff = h_new_centroids[i] - h_old_centroids[i];
                convergence_error += diff * diff;
            }
            convergence_error = sqrtf(convergence_error);
            
            delete[] h_new_centroids;
        }
        
        iteration++;
    }
    
    // Copy results back to host
    cudaMemcpy(centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(assignments, d_assignments, assignments_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    delete[] h_old_centroids;
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_new_centroids);
    cudaFree(d_assignments);
    cudaFree(d_distances);
    cudaFree(d_cluster_counts);
    
    return iteration;
} 