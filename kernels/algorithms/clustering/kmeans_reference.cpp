/*
 * K-Means Clustering Algorithm - C++ Reference Implementation
 * 
 * Simple, single-threaded CPU implementation of k-means clustering 
 * for educational purposes and correctness validation.
 * 
 * This reference implementation serves as:
 * - Educational baseline for understanding the algorithm
 * - Correctness validation for GPU implementations
 * - Performance comparison baseline
 * - Debugging reference when GPU results seem incorrect
 * 
 * Algorithm:
 * 1. Initialize k centroids randomly or with provided initial values
 * 2. Iteratively:
 *    a. Assign each point to nearest centroid (Euclidean distance)
 *    b. Update centroids as mean of assigned points
 *    c. Check for convergence or maximum iterations
 * 
 * Advantages:
 * - Simple and easy to understand
 * - Straightforward debugging
 * - No GPU memory management complexity
 * 
 * Disadvantages:
 * - Single-threaded, slow for large datasets
 * - No GPU acceleration
 * - Limited scalability
 * 
 * Applications:
 * - Small to medium datasets
 * - Educational purposes
 * - Algorithm validation
 * - Prototyping before GPU implementation
 */

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <iostream>

/**
 * Compute Euclidean distance between two points
 * 
 * @param point1: First point coordinates
 * @param point2: Second point coordinates  
 * @param n_dims: Number of dimensions
 * @return: Euclidean distance
 */
float euclidean_distance(const float* point1, const float* point2, int n_dims) {
    float distance = 0.0f;
    for (int d = 0; d < n_dims; d++) {
        float diff = point1[d] - point2[d];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

/**
 * Assign each point to the nearest centroid
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Current centroids [n_clusters x n_dims]
 * @param assignments: Output cluster assignments [n_points]
 * @param distances: Output minimum distances [n_points]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 */
void assign_points_to_centroids(
    const float* points,
    const float* centroids,
    int* assignments,
    float* distances,
    int n_points,
    int n_dims,
    int n_clusters
) {
    for (int i = 0; i < n_points; i++) {
        float min_distance = std::numeric_limits<float>::max();
        int best_cluster = 0;
        
        // Find nearest centroid
        for (int c = 0; c < n_clusters; c++) {
            float distance = euclidean_distance(
                &points[i * n_dims],
                &centroids[c * n_dims],
                n_dims
            );
            
            if (distance < min_distance) {
                min_distance = distance;
                best_cluster = c;
            }
        }
        
        assignments[i] = best_cluster;
        distances[i] = min_distance;
    }
}

/**
 * Update centroids based on current point assignments
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param assignments: Current cluster assignments [n_points]
 * @param centroids: Output updated centroids [n_clusters x n_dims]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 * @return: True if centroids changed significantly, false otherwise
 */
bool update_centroids(
    const float* points,
    const int* assignments,
    float* centroids,
    int n_points,
    int n_dims,
    int n_clusters
) {
    // Initialize centroid sums and counts
    std::vector<float> centroid_sums(n_clusters * n_dims, 0.0f);
    std::vector<int> cluster_counts(n_clusters, 0);
    
    // Accumulate points for each cluster
    for (int i = 0; i < n_points; i++) {
        int cluster_id = assignments[i];
        cluster_counts[cluster_id]++;
        
        for (int d = 0; d < n_dims; d++) {
            centroid_sums[cluster_id * n_dims + d] += points[i * n_dims + d];
        }
    }
    
    // Compute new centroids and check for changes
    bool changed = false;
    for (int c = 0; c < n_clusters; c++) {
        if (cluster_counts[c] > 0) {
            for (int d = 0; d < n_dims; d++) {
                float new_value = centroid_sums[c * n_dims + d] / cluster_counts[c];
                if (std::abs(new_value - centroids[c * n_dims + d]) > 1e-6f) {
                    changed = true;
                }
                centroids[c * n_dims + d] = new_value;
            }
        }
    }
    
    return changed;
}

/**
 * Initialize centroids randomly within data bounds
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Output initial centroids [n_clusters x n_dims]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 * @param seed: Random seed for reproducibility
 */
void initialize_centroids_random(
    const float* points,
    float* centroids,
    int n_points,
    int n_dims,
    int n_clusters,
    int seed = 42
) {
    std::mt19937 gen(seed);
    
    // Find data bounds
    std::vector<float> min_vals(n_dims, std::numeric_limits<float>::max());
    std::vector<float> max_vals(n_dims, std::numeric_limits<float>::lowest());
    
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < n_dims; d++) {
            float val = points[i * n_dims + d];
            min_vals[d] = std::min(min_vals[d], val);
            max_vals[d] = std::max(max_vals[d], val);
        }
    }
    
    // Generate random centroids within bounds
    for (int c = 0; c < n_clusters; c++) {
        for (int d = 0; d < n_dims; d++) {
            std::uniform_real_distribution<float> dist(min_vals[d], max_vals[d]);
            centroids[c * n_dims + d] = dist(gen);
        }
    }
}

/**
 * K-means clustering reference implementation
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Initial centroids [n_clusters x n_dims], updated in-place
 * @param assignments: Output cluster assignments [n_points]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 * @param max_iterations: Maximum number of iterations
 * @param tolerance: Convergence tolerance
 * @param initialize_centroids: Whether to randomly initialize centroids
 * @return: Number of iterations performed
 */
int kmeans_reference(
    const float* points,
    float* centroids,
    int* assignments,
    int n_points,
    int n_dims,
    int n_clusters,
    int max_iterations = 100,
    float tolerance = 1e-4f,
    bool initialize_centroids = false
) {
    // Initialize centroids if requested
    if (initialize_centroids) {
        initialize_centroids_random(points, centroids, n_points, n_dims, n_clusters);
    }
    
    std::vector<float> distances(n_points);
    
    int iteration = 0;
    bool converged = false;
    
    // Main k-means iteration loop
    while (iteration < max_iterations && !converged) {
        // Step 1: Assign points to nearest centroids
        assign_points_to_centroids(
            points, centroids, assignments, distances.data(),
            n_points, n_dims, n_clusters
        );
        
        // Step 2: Update centroids
        bool centroids_changed = update_centroids(
            points, assignments, centroids,
            n_points, n_dims, n_clusters
        );
        
        // Check convergence
        converged = !centroids_changed;
        iteration++;
        
        // Optional: Print iteration info
        if (iteration % 10 == 0) {
            float total_distance = 0.0f;
            for (int i = 0; i < n_points; i++) {
                total_distance += distances[i];
            }
            std::cout << "Iteration " << iteration 
                      << ", Total distance: " << total_distance << std::endl;
        }
    }
    
    return iteration;
}

/**
 * Compute total within-cluster sum of squares (WCSS)
 * Quality metric for clustering result
 * 
 * @param points: Input data points [n_points x n_dims]
 * @param centroids: Final centroids [n_clusters x n_dims]
 * @param assignments: Cluster assignments [n_points]
 * @param n_points: Number of data points
 * @param n_dims: Number of dimensions
 * @param n_clusters: Number of clusters
 * @return: Total WCSS
 */
float compute_wcss(
    const float* points,
    const float* centroids,
    const int* assignments,
    int n_points,
    int n_dims,
    int n_clusters
) {
    float total_wcss = 0.0f;
    
    for (int i = 0; i < n_points; i++) {
        int cluster_id = assignments[i];
        float distance = euclidean_distance(
            &points[i * n_dims],
            &centroids[cluster_id * n_dims],
            n_dims
        );
        total_wcss += distance * distance;
    }
    
    return total_wcss;
} 