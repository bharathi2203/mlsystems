#include <metal_stdlib>
using namespace metal;

/*
 * K-Means Clustering Algorithm - Advanced Metal Implementation
 * 
 * Optimized GPU implementation with focus on memory efficiency
 * and scalability for large datasets using Metal Shading Language.
 * 
 * Key differences from basic kmeans.metal:
 * - Tiled memory access patterns for better cache utilization
 * - Optimized for very large datasets that don't fit in GPU memory
 * - Multiple distance metrics (L1, L2, cosine)
 * - Advanced convergence detection strategies
 * - Better numerical stability for high-dimensional data
 * 
 */

// Configuration constants for optimization
constant uint WARP_SIZE = 32;
constant uint MAX_CLUSTERS_SHARED = 64;
constant uint MAX_DIMS_SHARED = 32;
constant uint TILE_SIZE = 256;

// Distance metric enumeration
constant uint EUCLIDEAN = 0;
constant uint MANHATTAN = 1;
constant uint COSINE = 2;

/**
 * Device function: Compute distance between point and centroid
 * Template-like specialization for different distance metrics
 */
float compute_distance_euclidean(const device float* point, const device float* centroid, uint n_dims) {
    float sum = 0.0f;
    for (uint d = 0; d < n_dims; d++) {
        float diff = point[d] - centroid[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

float compute_distance_manhattan(const device float* point, const device float* centroid, uint n_dims) {
    float sum = 0.0f;
    for (uint d = 0; d < n_dims; d++) {
        sum += abs(point[d] - centroid[d]);
    }
    return sum;
}

float compute_distance_cosine(const device float* point, const device float* centroid, uint n_dims) {
    float dot_product = 0.0f;
    float norm_point = 0.0f;
    float norm_centroid = 0.0f;
    
    for (uint d = 0; d < n_dims; d++) {
        dot_product += point[d] * centroid[d];
        norm_point += point[d] * point[d];
        norm_centroid += centroid[d] * centroid[d];
    }
    
    float norm_product = sqrt(norm_point * norm_centroid);
    return (norm_product > 1e-8f) ? (1.0f - dot_product / norm_product) : 0.0f;
}

/**
 * Metal Kernel: Tiled point assignment with multiple distance metrics
 * Processes points in tiles to maximize cache reuse and memory efficiency.
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Current centroids [n_clusters x n_dims]
 * @param assignments: Output cluster assignments [n_points]
 * @param distances: Output minimum distances [n_points]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 * @param distance_metric: Distance metric to use (0=Euclidean, 1=Manhattan, 2=Cosine)
 */
kernel void kmeans_assign_points_tiled(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device int* assignments [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant uint& n_points [[buffer(4)]],
    constant uint& n_dims [[buffer(5)]],
    constant uint& n_clusters [[buffer(6)]],
    constant uint& distance_metric [[buffer(7)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    // Shared memory layout
    threadgroup float* shared_centroids = shared_mem;
    threadgroup float* shared_points = &shared_mem[n_clusters * n_dims];
    
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Cooperatively load centroids into threadgroup memory
    for (uint i = thread_id; i < n_clusters * n_dims; i += threads_per_threadgroup) {
        shared_centroids[i] = centroids[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process points in tiles
    for (uint tile_start = threadgroup_id * threads_per_threadgroup; 
         tile_start < n_points; 
         tile_start += threadgroups_per_grid * threads_per_threadgroup) {
        
        uint point_idx = tile_start + thread_id;
        
        // Load point tile into threadgroup memory
        if (point_idx < n_points) {
            for (uint d = 0; d < n_dims; d++) {
                shared_points[thread_id * n_dims + d] = points[point_idx * n_dims + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute distances for this tile
        if (point_idx < n_points) {
            float min_distance = MAXFLOAT;
            int best_cluster = 0;
            
            // Compare with each centroid
            for (uint c = 0; c < n_clusters; c++) {
                float distance;
                
                // Select distance metric
                if (distance_metric == EUCLIDEAN) {
                    distance = compute_distance_euclidean(
                        &shared_points[thread_id * n_dims],
                        &shared_centroids[c * n_dims],
                        n_dims
                    );
                } else if (distance_metric == MANHATTAN) {
                    distance = compute_distance_manhattan(
                        &shared_points[thread_id * n_dims],
                        &shared_centroids[c * n_dims],
                        n_dims
                    );
                } else { // COSINE
                    distance = compute_distance_cosine(
                        &shared_points[thread_id * n_dims],
                        &shared_centroids[c * n_dims],
                        n_dims
                    );
                }
                
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = c;
                }
            }
            
            assignments[point_idx] = best_cluster;
            distances[point_idx] = min_distance;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/**
 * Metal Kernel: Warp-optimized centroid updates
 * Uses threadgroup-level cooperation for efficient parallel reduction.
 */
kernel void kmeans_update_centroids_warp_optimized(
    device const float* points [[buffer(0)]],
    device const int* assignments [[buffer(1)]],
    device float* new_centroids [[buffer(2)]],
    device atomic_uint* cluster_counts [[buffer(3)]],
    constant uint& n_points [[buffer(4)]],
    constant uint& n_dims [[buffer(5)]],
    constant uint& n_clusters [[buffer(6)]],
    threadgroup float* warp_centroids [[threadgroup(0)]],
    threadgroup atomic_uint* warp_counts [[threadgroup(1)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    uint warp_id = thread_id / WARP_SIZE;
    uint lane_id = thread_id % WARP_SIZE;
    
    // Initialize threadgroup memory
    if (lane_id < n_clusters && lane_id < MAX_CLUSTERS_SHARED) {
        for (uint w = 0; w < 32; w++) {
            warp_centroids[w * MAX_CLUSTERS_SHARED + lane_id] = 0.0f;
            atomic_store_explicit(&warp_counts[w * MAX_CLUSTERS_SHARED + lane_id], 0, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Process points with threadgroup-level cooperation
    for (uint point_idx = global_id; point_idx < n_points; point_idx += threads_per_threadgroup * threadgroups_per_grid) {
        int cluster_id = assignments[point_idx];
        
        // Accumulate within warp first
        if (cluster_id < MAX_CLUSTERS_SHARED) {
            atomic_fetch_add_explicit(&warp_counts[warp_id * MAX_CLUSTERS_SHARED + cluster_id], 1, memory_order_relaxed);
            
            for (uint d = 0; d < n_dims; d++) {
                device atomic<float>* centroid_ptr = (device atomic<float>*)&warp_centroids[warp_id * MAX_CLUSTERS_SHARED + cluster_id];
                atomic_fetch_add_explicit(centroid_ptr, points[point_idx * n_dims + d], memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce across warps
    if (thread_id < n_clusters) {
        float centroid_sum = 0.0f;
        uint total_count = 0;
        
        for (uint w = 0; w < 32; w++) {
            centroid_sum += warp_centroids[w * MAX_CLUSTERS_SHARED + thread_id];
            total_count += atomic_load_explicit(&warp_counts[w * MAX_CLUSTERS_SHARED + thread_id], memory_order_relaxed);
        }
        
        // Update global centroids
        atomic_fetch_add_explicit(&cluster_counts[thread_id], total_count, memory_order_relaxed);
        for (uint d = 0; d < n_dims; d++) {
            device atomic<float>* global_centroid_ptr = (device atomic<float>*)&new_centroids[thread_id * n_dims + d];
            atomic_fetch_add_explicit(global_centroid_ptr, centroid_sum, memory_order_relaxed);
        }
    }
}

/**
 * Metal Kernel: Adaptive convergence detection
 * Computes convergence metrics adaptively based on data characteristics
 * and iteration history.
 */
kernel void compute_convergence_metrics(
    device const float* old_centroids [[buffer(0)]],
    device const float* new_centroids [[buffer(1)]],
    device float* convergence_deltas [[buffer(2)]],
    constant uint& n_clusters [[buffer(3)]],
    constant uint& n_dims [[buffer(4)]],
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
 * Metal Kernel: K-means++ initialization
 * Implements the k-means++ algorithm for better initial centroid selection
 */
kernel void kmeans_plus_plus_init(
    device const float* points [[buffer(0)]],
    device float* centroids [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device float* probabilities [[buffer(3)]],
    constant uint& n_points [[buffer(4)]],
    constant uint& n_dims [[buffer(5)]],
    constant uint& n_clusters [[buffer(6)]],
    constant uint& current_cluster [[buffer(7)]],
    constant uint& distance_metric [[buffer(8)]],
    threadgroup float* shared_centroid [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Load current centroid into threadgroup memory
    if (current_cluster > 0) {
        for (uint i = thread_id; i < n_dims; i += threads_per_threadgroup) {
            shared_centroid[i] = centroids[(current_cluster - 1) * n_dims + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (global_id < n_points) {
        if (current_cluster == 0) {
            // First centroid: initialize distances to infinity
            distances[global_id] = MAXFLOAT;
        } else {
            // Compute distance to latest centroid
            float distance;
            
            if (distance_metric == EUCLIDEAN) {
                distance = compute_distance_euclidean(
                    &points[global_id * n_dims],
                    shared_centroid,
                    n_dims
                );
            } else if (distance_metric == MANHATTAN) {
                distance = compute_distance_manhattan(
                    &points[global_id * n_dims],
                    shared_centroid,
                    n_dims
                );
            } else { // COSINE
                distance = compute_distance_cosine(
                    &points[global_id * n_dims],
                    shared_centroid,
                    n_dims
                );
            }
            
            // Update minimum distance
            distances[global_id] = min(distances[global_id], distance);
        }
        
        // Compute probability (squared distance)
        probabilities[global_id] = distances[global_id] * distances[global_id];
    }
}

/**
 * Metal Kernel: Compute silhouette score for clustering quality assessment
 */
kernel void compute_silhouette_score(
    device const float* points [[buffer(0)]],
    device const int* assignments [[buffer(1)]],
    device float* silhouette_scores [[buffer(2)]],
    constant uint& n_points [[buffer(3)]],
    constant uint& n_dims [[buffer(4)]],
    constant uint& n_clusters [[buffer(5)]],
    constant uint& distance_metric [[buffer(6)]],
    uint point_id [[thread_position_in_grid]]
) {
    if (point_id >= n_points) return;
    
    int own_cluster = assignments[point_id];
    float a = 0.0f; // Average distance to points in same cluster
    float b = MAXFLOAT; // Minimum average distance to points in other clusters
    
    // Count points in same cluster
    uint same_cluster_count = 0;
    for (uint i = 0; i < n_points; i++) {
        if (assignments[i] == own_cluster && i != point_id) {
            same_cluster_count++;
        }
    }
    
    // Compute average distances
    for (uint cluster = 0; cluster < n_clusters; cluster++) {
        float total_distance = 0.0f;
        uint count = 0;
        
        for (uint i = 0; i < n_points; i++) {
            if (assignments[i] == cluster && i != point_id) {
                float distance;
                
                if (distance_metric == EUCLIDEAN) {
                    distance = compute_distance_euclidean(
                        &points[point_id * n_dims],
                        &points[i * n_dims],
                        n_dims
                    );
                } else if (distance_metric == MANHATTAN) {
                    distance = compute_distance_manhattan(
                        &points[point_id * n_dims],
                        &points[i * n_dims],
                        n_dims
                    );
                } else { // COSINE
                    distance = compute_distance_cosine(
                        &points[point_id * n_dims],
                        &points[i * n_dims],
                        n_dims
                    );
                }
                
                total_distance += distance;
                count++;
            }
        }
        
        if (count > 0) {
            float avg_distance = total_distance / count;
            
            if (cluster == own_cluster) {
                a = avg_distance;
            } else {
                b = min(b, avg_distance);
            }
        }
    }
    
    // Compute silhouette score
    float silhouette = (same_cluster_count > 0) ? (b - a) / max(a, b) : 0.0f;
    silhouette_scores[point_id] = silhouette;
}

/**
 * Metal Kernel: Parallel reduction for computing total silhouette score
 */
kernel void reduce_silhouette_scores(
    device const float* silhouette_scores [[buffer(0)]],
    device float* total_score [[buffer(1)]],
    constant uint& n_points [[buffer(2)]],
    threadgroup float* shared_scores [[threadgroup(0)]],
    uint thread_id [[thread_position_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint global_id = threadgroup_id * threads_per_threadgroup + thread_id;
    
    // Load scores into threadgroup memory
    shared_scores[thread_id] = (global_id < n_points) ? silhouette_scores[global_id] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within threadgroup
    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_scores[thread_id] += shared_scores[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store threadgroup result
    if (thread_id == 0) {
        device atomic<float>* total_ptr = (device atomic<float>*)total_score;
        atomic_fetch_add_explicit(total_ptr, shared_scores[0], memory_order_relaxed);
    }
}
