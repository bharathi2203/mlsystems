#include <metal_stdlib>
using namespace metal;

/*
 * K-Means Clustering Algorithm - Metal Implementation
 * 
 * GPU-accelerated implementation of the k-means clustering algorithm
 * optimized for Apple Silicon GPUs using Metal Shading Language.
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
 * - Centroids: cached in threadgroup memory
 * - Assignments: coalesced writes
 */

// Configuration constants
constant uint MAX_CLUSTERS = 256;
constant uint THREADGROUP_SIZE = 256;

/**
 * Metal Kernel: Compute distances and assign points to nearest centroids
 * Each thread processes one data point and computes distances to all centroids.
 * Uses threadgroup memory to cache centroids for efficient access.
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Current centroids [n_clusters x n_dims] 
 * @param assignments: Output cluster assignments [n_points]
 * @param distances: Output minimum distances [n_points]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 */
kernel void kmeans_assign_points(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device int* assignments [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant uint& n_points [[buffer(4)]],
    constant uint& n_dims [[buffer(5)]],
    constant uint& n_clusters [[buffer(6)]],
    threadgroup float* shared_centroids [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Load centroids into threadgroup memory cooperatively
    uint centroids_size = n_clusters * n_dims;
    for (uint i = thread_id; i < centroids_size; i += threads_per_threadgroup) {
        if (i < centroids_size) {
            shared_centroids[i] = centroids[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (global_id < n_points) {
        float min_distance = MAXFLOAT;
        int best_cluster = 0;
        
        // Compute distance to each centroid
        for (uint c = 0; c < n_clusters; c++) {
            float distance = 0.0f;
            
            // Euclidean distance computation
            for (uint d = 0; d < n_dims; d++) {
                float diff = points[global_id * n_dims + d] - shared_centroids[c * n_dims + d];
                distance += diff * diff;
            }
            
            // Update minimum distance and best cluster
            if (distance < min_distance) {
                min_distance = distance;
                best_cluster = c;
            }
        }
        
        assignments[global_id] = best_cluster;
        distances[global_id] = sqrt(min_distance);
    }
}

/**
 * Metal Kernel: Update centroids based on point assignments
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
kernel void kmeans_update_centroids(
    device const float* points [[buffer(0)]],
    device const int* assignments [[buffer(1)]],
    device float* new_centroids [[buffer(2)]],
    device atomic_uint* cluster_counts [[buffer(3)]],
    constant uint& n_points [[buffer(4)]],
    constant uint& n_dims [[buffer(5)]],
    constant uint& n_clusters [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id < n_points) {
        int cluster_id = assignments[global_id];
        
        // Atomically increment cluster count
        atomic_fetch_add_explicit(&cluster_counts[cluster_id], 1, memory_order_relaxed);
        
        // Atomically add point coordinates to centroid sum
        for (uint d = 0; d < n_dims; d++) {
            device atomic<float>* centroid_ptr = (device atomic<float>*)&new_centroids[cluster_id * n_dims + d];
            atomic_fetch_add_explicit(centroid_ptr, points[global_id * n_dims + d], memory_order_relaxed);
        }
    }
}

/**
 * Metal Kernel: Finalize centroid computation by dividing by cluster counts
 * 
 * @param centroids: Centroid sums to be normalized [n_clusters x n_dims]
 * @param cluster_counts: Number of points per cluster [n_clusters]
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 */
kernel void kmeans_finalize_centroids(
    device float* centroids [[buffer(0)]],
    device const atomic_uint* cluster_counts [[buffer(1)]],
    constant uint& n_dims [[buffer(2)]],
    constant uint& n_clusters [[buffer(3)]],
    uint global_id [[thread_position_in_grid]]
) {
    uint cluster_id = global_id / n_dims;
    uint dim_id = global_id % n_dims;
    
    if (cluster_id < n_clusters && dim_id < n_dims) {
        uint count = atomic_load_explicit(&cluster_counts[cluster_id], memory_order_relaxed);
        if (count > 0) {
            centroids[global_id] /= float(count);
        }
    }
}

/**
 * Metal Kernel: Compute convergence metrics
 * Computes the change in centroids between iterations
 * 
 * @param old_centroids: Previous centroids [n_clusters x n_dims]
 * @param new_centroids: Current centroids [n_clusters x n_dims]
 * @param convergence_deltas: Output convergence metrics [n_clusters]
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 */
kernel void compute_convergence(
    device const float* old_centroids [[buffer(0)]],
    device const float* new_centroids [[buffer(1)]],
    device float* convergence_deltas [[buffer(2)]],
    constant uint& n_dims [[buffer(3)]],
    constant uint& n_clusters [[buffer(4)]],
    uint cluster_id [[thread_position_in_grid]]
) {
    if (cluster_id < n_clusters) {
        float centroid_delta = 0.0f;
        float centroid_norm = 0.0f;
        
        for (uint d = 0; d < n_dims; d++) {
            float old_val = old_centroids[cluster_id * n_dims + d];
            float new_val = new_centroids[cluster_id * n_dims + d];
            float delta = new_val - old_val;
            
            centroid_delta += delta * delta;
            centroid_norm += new_val * new_val;
        }
        
        // Relative change metric
        convergence_deltas[cluster_id] = (centroid_norm > 1e-8f) ? 
            sqrt(centroid_delta / centroid_norm) : sqrt(centroid_delta);
    }
}

/**
 * Metal Kernel: Initialize centroids randomly within data bounds
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Output initial centroids [n_clusters x n_dims]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 * @param seed: Random seed
 */
kernel void initialize_centroids_random(
    device const float* points [[buffer(0)]],
    device float* centroids [[buffer(1)]],
    constant uint& n_points [[buffer(2)]],
    constant uint& n_dims [[buffer(3)]],
    constant uint& n_clusters [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    threadgroup float* shared_min_max [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Find data bounds cooperatively
    threadgroup float* min_vals = shared_min_max;
    threadgroup float* max_vals = &shared_min_max[n_dims];
    
    // Initialize bounds
    if (thread_id < n_dims) {
        min_vals[thread_id] = MAXFLOAT;
        max_vals[thread_id] = -MAXFLOAT;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Find min/max values
    for (uint i = global_id; i < n_points; i += threads_per_threadgroup * threadgroups_per_grid) {
        for (uint d = 0; d < n_dims; d++) {
            float val = points[i * n_dims + d];
            min_vals[d] = min(min_vals[d], val);
            max_vals[d] = max(max_vals[d], val);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Generate random centroids
    if (global_id < n_clusters * n_dims) {
        uint cluster_id = global_id / n_dims;
        uint dim_id = global_id % n_dims;
        
        // Simple linear congruential generator
        uint rng_state = seed + global_id;
        rng_state = rng_state * 1664525u + 1013904223u;
        float rand_val = float(rng_state) / float(0xFFFFFFFFu);
        
        centroids[global_id] = min_vals[dim_id] + rand_val * (max_vals[dim_id] - min_vals[dim_id]);
    }
}

/**
 * Metal Kernel: Compute total within-cluster sum of squares (WCSS)
 * Quality metric for clustering result
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Final centroids [n_clusters x n_dims]
 * @param assignments: Cluster assignments [n_points]
 * @param wcss_partial: Partial WCSS values [threadgroups]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 */
kernel void compute_wcss(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device const int* assignments [[buffer(2)]],
    device float* wcss_partial [[buffer(3)]],
    constant uint& n_points [[buffer(4)]],
    constant uint& n_dims [[buffer(5)]],
    constant uint& n_clusters [[buffer(6)]],
    threadgroup float* shared_wcss [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    float local_wcss = 0.0f;
    
    if (global_id < n_points) {
        int cluster_id = assignments[global_id];
        
        // Compute squared distance to assigned centroid
        for (uint d = 0; d < n_dims; d++) {
            float diff = points[global_id * n_dims + d] - centroids[cluster_id * n_dims + d];
            local_wcss += diff * diff;
        }
    }
    
    shared_wcss[thread_id] = local_wcss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_wcss[thread_id] += shared_wcss[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store threadgroup result
    if (thread_id == 0) {
        wcss_partial[threadgroup_id] = shared_wcss[0];
    }
}
