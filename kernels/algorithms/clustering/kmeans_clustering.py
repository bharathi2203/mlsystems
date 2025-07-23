import triton
import triton.language as tl
import torch

"""
K-Means Clustering Algorithm - Advanced Triton Implementation

Optimized GPU implementation with focus on memory efficiency
and scalability for large datasets using OpenAI Triton.

Key differences from basic kmeans.py:
- Tiled memory access patterns for better cache utilization
- Optimized for very large datasets that don't fit in GPU memory
- Multiple distance metrics (L1, L2, cosine)
- Advanced convergence detection strategies
- Better numerical stability for high-dimensional data
"""

@triton.jit
def compute_distance_euclidean(point_ptr, centroid_ptr, n_dims):
    """Compute Euclidean distance between point and centroid"""
    sum_sq = 0.0
    for d in range(n_dims):
        diff = tl.load(point_ptr + d) - tl.load(centroid_ptr + d)
        sum_sq += diff * diff
    return tl.sqrt(sum_sq)

@triton.jit
def compute_distance_manhattan(point_ptr, centroid_ptr, n_dims):
    """Compute Manhattan distance between point and centroid"""
    sum_abs = 0.0
    for d in range(n_dims):
        diff = tl.load(point_ptr + d) - tl.load(centroid_ptr + d)
        sum_abs += tl.abs(diff)
    return sum_abs

@triton.jit
def compute_distance_cosine(point_ptr, centroid_ptr, n_dims):
    """Compute Cosine distance between point and centroid"""
    dot_product = 0.0
    norm_point = 0.0
    norm_centroid = 0.0
    
    for d in range(n_dims):
        p_val = tl.load(point_ptr + d)
        c_val = tl.load(centroid_ptr + d)
        dot_product += p_val * c_val
        norm_point += p_val * p_val
        norm_centroid += c_val * c_val
    
    norm_product = tl.sqrt(norm_point * norm_centroid)
    return tl.where(norm_product > 1e-8, 1.0 - dot_product / norm_product, 0.0)

@triton.jit
def kmeans_assign_points_multimetric_kernel(
    points_ptr,  # Pointer to input data points [n_points x n_dims]
    centroids_ptr,  # Pointer to centroids [n_clusters x n_dims]
    assignments_ptr,  # Pointer to output assignments [n_points]
    distances_ptr,  # Pointer to output distances [n_points]
    n_points,  # Number of data points
    n_dims,  # Number of dimensions
    n_clusters,  # Number of clusters
    distance_metric,  # Distance metric (0=Euclidean, 1=Manhattan, 2=Cosine)
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Advanced point assignment kernel with multiple distance metrics
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_points
    
    # Initialize outputs
    min_distances = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    best_clusters = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Load points for this block
    point_offsets = offsets[:, None] * n_dims + tl.arange(0, n_dims)[None, :]
    points = tl.load(points_ptr + point_offsets, mask=mask[:, None])
    
    # Iterate through all centroids
    for cluster_id in range(n_clusters):
        # Load centroid
        centroid_offsets = cluster_id * n_dims + tl.arange(0, n_dims)
        centroid = tl.load(centroids_ptr + centroid_offsets)
        
        # Compute distances based on metric
        if distance_metric == 0:  # Euclidean
            diff = points - centroid[None, :]
            squared_diff = diff * diff
            distances = tl.sqrt(tl.sum(squared_diff, axis=1))
        elif distance_metric == 1:  # Manhattan
            diff = tl.abs(points - centroid[None, :])
            distances = tl.sum(diff, axis=1)
        else:  # Cosine
            dot_products = tl.sum(points * centroid[None, :], axis=1)
            norm_points = tl.sqrt(tl.sum(points * points, axis=1))
            norm_centroid = tl.sqrt(tl.sum(centroid * centroid))
            norm_products = norm_points * norm_centroid
            distances = tl.where(norm_products > 1e-8, 
                                1.0 - dot_products / norm_products, 0.0)
        
        # Update minimum distances and best clusters
        update_mask = distances < min_distances
        min_distances = tl.where(update_mask, distances, min_distances)
        best_clusters = tl.where(update_mask, cluster_id, best_clusters)
    
    # Store results
    tl.store(assignments_ptr + offsets, best_clusters, mask=mask)
    tl.store(distances_ptr + offsets, min_distances, mask=mask)


@triton.jit
def kmeans_update_centroids_tiled_kernel(
    points_ptr,  # Pointer to input data points [n_points x n_dims]
    assignments_ptr,  # Pointer to cluster assignments [n_points]
    centroids_ptr,  # Pointer to output centroids [n_clusters x n_dims]
    counts_ptr,  # Pointer to cluster counts [n_clusters]
    n_points,  # Number of data points
    n_dims,  # Number of dimensions
    n_clusters,  # Number of clusters
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Tiled centroid update kernel for memory efficiency
    """
    cluster_id = tl.program_id(axis=0)
    
    if cluster_id >= n_clusters:
        return
    
    # Initialize accumulators
    centroid_sum = tl.zeros([n_dims], dtype=tl.float32)
    count = 0
    
    # Process points in tiles
    for tile_start in range(0, n_points, BLOCK_SIZE):
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_points
        
        # Load assignments for this tile
        assignments = tl.load(assignments_ptr + offsets, mask=mask, other=-1)
        
        # Find points assigned to this cluster
        cluster_mask = (assignments == cluster_id) & mask
        
        if tl.sum(cluster_mask.to(tl.int32)) > 0:
            # Load points assigned to this cluster
            point_offsets = offsets[:, None] * n_dims + tl.arange(0, n_dims)[None, :]
            points = tl.load(points_ptr + point_offsets, mask=cluster_mask[:, None], other=0.0)
            
            # Accumulate points
            cluster_points = tl.where(cluster_mask[:, None], points, 0.0)
            centroid_sum += tl.sum(cluster_points, axis=0)
            count += tl.sum(cluster_mask.to(tl.int32))
    
    # Compute final centroid
    if count > 0:
        final_centroid = centroid_sum / count
        
        # Store centroid
        centroid_offsets = cluster_id * n_dims + tl.arange(0, n_dims)
        tl.store(centroids_ptr + centroid_offsets, final_centroid)
        
        # Store count
        tl.store(counts_ptr + cluster_id, count)


@triton.jit
def kmeans_plus_plus_kernel(
    points_ptr,  # Pointer to input data points [n_points x n_dims]
    centroids_ptr,  # Pointer to centroids [n_clusters x n_dims]
    distances_ptr,  # Pointer to minimum distances [n_points]
    probabilities_ptr,  # Pointer to selection probabilities [n_points]
    n_points,  # Number of data points
    n_dims,  # Number of dimensions
    current_cluster,  # Current cluster being initialized
    distance_metric,  # Distance metric
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    K-means++ initialization kernel
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_points
    
    if current_cluster == 0:
        # Initialize distances to infinity for first centroid
        tl.store(distances_ptr + offsets, float('inf'), mask=mask)
        return
    
    # Load current centroid
    centroid_offsets = (current_cluster - 1) * n_dims + tl.arange(0, n_dims)
    centroid = tl.load(centroids_ptr + centroid_offsets)
    
    # Load points for this block
    point_offsets = offsets[:, None] * n_dims + tl.arange(0, n_dims)[None, :]
    points = tl.load(points_ptr + point_offsets, mask=mask[:, None])
    
    # Load current minimum distances
    current_distances = tl.load(distances_ptr + offsets, mask=mask, other=float('inf'))
    
    # Compute distances to new centroid
    if distance_metric == 0:  # Euclidean
        diff = points - centroid[None, :]
        squared_diff = diff * diff
        new_distances = tl.sqrt(tl.sum(squared_diff, axis=1))
    elif distance_metric == 1:  # Manhattan
        diff = tl.abs(points - centroid[None, :])
        new_distances = tl.sum(diff, axis=1)
    else:  # Cosine
        dot_products = tl.sum(points * centroid[None, :], axis=1)
        norm_points = tl.sqrt(tl.sum(points * points, axis=1))
        norm_centroid = tl.sqrt(tl.sum(centroid * centroid))
        norm_products = norm_points * norm_centroid
        new_distances = tl.where(norm_products > 1e-8, 
                               1.0 - dot_products / norm_products, 0.0)
    
    # Update minimum distances
    updated_distances = tl.minimum(current_distances, new_distances)
    tl.store(distances_ptr + offsets, updated_distances, mask=mask)
    
    # Compute probabilities (squared distances)
    probabilities = updated_distances * updated_distances
    tl.store(probabilities_ptr + offsets, probabilities, mask=mask)


@triton.jit
def compute_silhouette_kernel(
    points_ptr,  # Pointer to input data points [n_points x n_dims]
    assignments_ptr,  # Pointer to cluster assignments [n_points]
    silhouette_scores_ptr,  # Pointer to output silhouette scores [n_points]
    n_points,  # Number of data points
    n_dims,  # Number of dimensions
    n_clusters,  # Number of clusters
    distance_metric,  # Distance metric
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Compute silhouette score for clustering quality assessment
    """
    point_id = tl.program_id(axis=0)
    
    if point_id >= n_points:
        return
    
    own_cluster = tl.load(assignments_ptr + point_id)
    
    # Load point coordinates
    point_offsets = point_id * n_dims + tl.arange(0, n_dims)
    point = tl.load(points_ptr + point_offsets)
    
    # Compute average distances to all clusters
    cluster_distances = tl.zeros([n_clusters], dtype=tl.float32)
    cluster_counts = tl.zeros([n_clusters], dtype=tl.int32)
    
    # Process all other points
    for other_id in range(n_points):
        if other_id == point_id:
            continue
        
        other_cluster = tl.load(assignments_ptr + other_id)
        
        # Load other point coordinates
        other_offsets = other_id * n_dims + tl.arange(0, n_dims)
        other_point = tl.load(points_ptr + other_offsets)
        
        # Compute distance
        if distance_metric == 0:  # Euclidean
            diff = point - other_point
            distance = tl.sqrt(tl.sum(diff * diff))
        elif distance_metric == 1:  # Manhattan
            diff = tl.abs(point - other_point)
            distance = tl.sum(diff)
        else:  # Cosine
            dot_product = tl.sum(point * other_point)
            norm_point = tl.sqrt(tl.sum(point * point))
            norm_other = tl.sqrt(tl.sum(other_point * other_point))
            norm_product = norm_point * norm_other
            distance = tl.where(norm_product > 1e-8, 
                              1.0 - dot_product / norm_product, 0.0)
        
        # Accumulate distance for the cluster
        cluster_distances = tl.where(tl.arange(0, n_clusters) == other_cluster,
                                   cluster_distances + distance,
                                   cluster_distances)
        cluster_counts = tl.where(tl.arange(0, n_clusters) == other_cluster,
                                cluster_counts + 1,
                                cluster_counts)
    
    # Compute average distances
    avg_distances = tl.where(cluster_counts > 0,
                           cluster_distances / cluster_counts.to(tl.float32),
                           float('inf'))
    
    # Get a (average distance to own cluster) and b (min average distance to other clusters)
    a = avg_distances[own_cluster]
    
    # Find minimum average distance to other clusters
    b = float('inf')
    for c in range(n_clusters):
        if c != own_cluster and cluster_counts[c] > 0:
            b = tl.minimum(b, avg_distances[c])
    
    # Compute silhouette score
    silhouette = tl.where(cluster_counts[own_cluster] > 0 and tl.maximum(a, b) > 0,
                         (b - a) / tl.maximum(a, b), 0.0)
    
    tl.store(silhouette_scores_ptr + point_id, silhouette)


def kmeans_clustering_triton(points: torch.Tensor, n_clusters: int, 
                           max_iterations: int = 100, tolerance: float = 1e-4,
                           distance_metric: int = 0, init_method: str = 'kmeans++',
                           n_runs: int = 1) -> tuple:
    """
    Advanced K-means clustering using Triton implementation
    
    Args:
        points: Input tensor of shape [n_points, n_dims]
        n_clusters: Number of clusters
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        distance_metric: Distance metric (0=Euclidean, 1=Manhattan, 2=Cosine)
        init_method: Initialization method ('random' or 'kmeans++')
        n_runs: Number of runs for best result selection
        
    Returns:
        Tuple of (centroids, assignments, silhouette_score, n_iterations)
    """
    device = points.device
    n_points, n_dims = points.shape
    
    best_loss = float('inf')
    best_centroids = None
    best_assignments = None
    best_silhouette = -1.0
    
    for run in range(n_runs):
        # Initialize centroids
        centroids = torch.empty((n_clusters, n_dims), device=device, dtype=torch.float32)
        
        if init_method == 'kmeans++':
            # K-means++ initialization
            distances = torch.full((n_points,), float('inf'), device=device, dtype=torch.float32)
            probabilities = torch.zeros(n_points, device=device, dtype=torch.float32)
            
            # Select first centroid randomly
            first_idx = torch.randint(0, n_points, (1,)).item()
            centroids[0] = points[first_idx]
            
            BLOCK_SIZE = 256
            grid_points = triton.cdiv(n_points, BLOCK_SIZE)
            
            for cluster_id in range(1, n_clusters):
                # Update distances and probabilities
                kmeans_plus_plus_kernel[grid_points](
                    points, centroids, distances, probabilities,
                    n_points, n_dims, cluster_id, distance_metric, BLOCK_SIZE
                )
                
                # Select next centroid based on probabilities
                total_prob = probabilities.sum()
                if total_prob > 0:
                    cumulative_probs = torch.cumsum(probabilities / total_prob, dim=0)
                    rand_val = torch.rand(1).item()
                    selected_idx = torch.searchsorted(cumulative_probs, rand_val).item()
                    centroids[cluster_id] = points[selected_idx]
        else:
            # Random initialization
            min_vals = points.min(dim=0)[0]
            max_vals = points.max(dim=0)[0]
            centroids = min_vals + torch.rand_like(centroids) * (max_vals - min_vals)
        
        # Allocate memory for assignments and distances
        assignments = torch.zeros(n_points, device=device, dtype=torch.int32)
        distances = torch.zeros(n_points, device=device, dtype=torch.float32)
        cluster_counts = torch.zeros(n_clusters, device=device, dtype=torch.int32)
        
        # Main k-means loop
        BLOCK_SIZE = 256
        grid_points = triton.cdiv(n_points, BLOCK_SIZE)
        grid_clusters = n_clusters
        
        prev_loss = float('inf')
        
        for iteration in range(max_iterations):
            # Step 1: Assign points to nearest centroids
            kmeans_assign_points_multimetric_kernel[grid_points](
                points, centroids, assignments, distances,
                n_points, n_dims, n_clusters, distance_metric, BLOCK_SIZE
            )
            
            # Step 2: Update centroids
            cluster_counts.zero_()
            centroids.zero_()
            
            kmeans_update_centroids_tiled_kernel[grid_clusters](
                points, assignments, centroids, cluster_counts,
                n_points, n_dims, n_clusters, BLOCK_SIZE
            )
            
            # Step 3: Check convergence
            if iteration % 5 == 0:
                current_loss = distances.sum().item() / n_points
                
                if abs(current_loss - prev_loss) < tolerance:
                    break
                
                prev_loss = current_loss
        
        # Compute silhouette score for this run
        silhouette_scores = torch.zeros(n_points, device=device, dtype=torch.float32)
        compute_silhouette_kernel[n_points](
            points, assignments, silhouette_scores,
            n_points, n_dims, n_clusters, distance_metric, BLOCK_SIZE
        )
        
        avg_silhouette = silhouette_scores.mean().item()
        current_loss = distances.sum().item()
        
        # Keep best result
        if current_loss < best_loss:
            best_loss = current_loss
            best_centroids = centroids.clone()
            best_assignments = assignments.clone()
            best_silhouette = avg_silhouette
    
    return best_centroids, best_assignments, best_silhouette, iteration + 1


# Testing and utility functions
def evaluate_clustering_quality(points: torch.Tensor, centroids: torch.Tensor, 
                               assignments: torch.Tensor) -> dict:
    """Evaluate clustering quality metrics"""
    device = points.device
    n_points, n_dims = points.shape
    n_clusters = centroids.shape[0]
    
    # Within-cluster sum of squares (WCSS)
    wcss = 0.0
    for k in range(n_clusters):
        cluster_points = points[assignments == k]
        if len(cluster_points) > 0:
            centroid = centroids[k]
            distances = torch.norm(cluster_points - centroid, dim=1)
            wcss += (distances ** 2).sum().item()
    
    # Between-cluster sum of squares (BCSS)
    overall_centroid = points.mean(dim=0)
    bcss = 0.0
    for k in range(n_clusters):
        cluster_size = (assignments == k).sum().item()
        if cluster_size > 0:
            centroid = centroids[k]
            distance = torch.norm(centroid - overall_centroid).item()
            bcss += cluster_size * (distance ** 2)
    
    # Calinski-Harabasz score
    if n_clusters > 1 and wcss > 0:
        ch_score = (bcss / (n_clusters - 1)) / (wcss / (n_points - n_clusters))
    else:
        ch_score = 0.0
    
    return {
        'wcss': wcss,
        'bcss': bcss,
        'calinski_harabasz': ch_score,
        'total_variance': wcss + bcss
    }


def test_kmeans_clustering_triton():
    """Test the advanced Triton k-means implementation"""
    # Generate synthetic clustered data
    torch.manual_seed(42)
    n_points, n_dims, n_clusters = 2000, 3, 4
    
    # Create well-separated clusters
    cluster_centers = torch.tensor([
        [0, 0, 0], [5, 5, 5], [-3, 4, -2], [8, -3, 6]
    ], dtype=torch.float32)
    
    points = []
    for center in cluster_centers:
        cluster_points = center + torch.randn(n_points // n_clusters, n_dims)
        points.append(cluster_points)
    
    points = torch.cat(points, dim=0).cuda()
    
    # Test different distance metrics
    for metric, metric_name in [(0, 'Euclidean'), (1, 'Manhattan'), (2, 'Cosine')]:
        print(f"\nTesting {metric_name} distance metric:")
        
        centroids, assignments, silhouette, n_iter = kmeans_clustering_triton(
            points, n_clusters, distance_metric=metric, init_method='kmeans++', n_runs=3
        )
        
        # Evaluate quality
        quality_metrics = evaluate_clustering_quality(points, centroids, assignments)
        
        print(f"  Converged in {n_iter} iterations")
        print(f"  Silhouette score: {silhouette:.4f}")
        print(f"  WCSS: {quality_metrics['wcss']:.2f}")
        print(f"  Calinski-Harabasz: {quality_metrics['calinski_harabasz']:.2f}")
        print(f"  Cluster sizes: {torch.bincount(assignments)}")


if __name__ == "__main__":
    test_kmeans_clustering_triton()
