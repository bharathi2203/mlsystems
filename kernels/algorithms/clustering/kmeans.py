import triton
import triton.language as tl
import torch

"""
K-Means Clustering Algorithm - Triton Implementation

GPU-accelerated implementation of the k-means clustering algorithm
using OpenAI Triton for optimal GPU performance.

Key concepts:
- Iterative centroid optimization
- Distance computation parallelization
- Cluster assignment updates
- Convergence detection

Algorithm:
1. Initialize k centroids randomly or with k-means++
2. Assign each point to nearest centroid (parallel distance computation)
3. Update centroids as mean of assigned points (parallel reduction)
4. Repeat until convergence or max iterations

Distance metrics:
- Euclidean distance (L2) - primary implementation
- Manhattan distance (L1) - alternative
- Cosine similarity - for normalized data

Memory access patterns:
- Points: coalesced reads across thread blocks
- Centroids: cached in block memory
- Assignments: coalesced writes
"""

@triton.jit
def kmeans_assign_points_kernel(
    points_ptr,  # Pointer to input data points [n_points x n_dims]
    centroids_ptr,  # Pointer to centroids [n_clusters x n_dims]
    assignments_ptr,  # Pointer to output assignments [n_points]
    distances_ptr,  # Pointer to output distances [n_points]
    n_points,  # Number of data points
    n_dims,  # Number of dimensions
    n_clusters,  # Number of clusters
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    """
    Triton kernel for assigning points to nearest centroids
    
    Each program instance processes BLOCK_SIZE points and computes
    distances to all centroids using block-level operations.
    """
    # Get program ID and compute point indices
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
        
        # Compute squared distances
        diff = points - centroid[None, :]
        squared_diff = diff * diff
        distances = tl.sum(squared_diff, axis=1)
        
        # Update minimum distances and best clusters
        update_mask = distances < min_distances
        min_distances = tl.where(update_mask, distances, min_distances)
        best_clusters = tl.where(update_mask, cluster_id, best_clusters)
    
    # Take square root for final distances
    final_distances = tl.sqrt(min_distances)
    
    # Store results
    tl.store(assignments_ptr + offsets, best_clusters, mask=mask)
    tl.store(distances_ptr + offsets, final_distances, mask=mask)


@triton.jit
def kmeans_update_centroids_kernel(
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
    Triton kernel for updating centroids based on point assignments
    
    Each program instance processes points for one cluster and accumulates
    the centroid coordinates using atomic operations.
    """
    # Get cluster ID from program ID
    cluster_id = tl.program_id(axis=0)
    
    if cluster_id >= n_clusters:
        return
    
    # Initialize accumulators
    centroid_sum = tl.zeros([n_dims], dtype=tl.float32)
    count = 0
    
    # Process points in blocks
    for block_start in range(0, n_points, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_points
        
        # Load assignments for this block
        assignments = tl.load(assignments_ptr + offsets, mask=mask, other=0)
        
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
def kmeans_compute_loss_kernel(
    points_ptr,  # Pointer to input data points [n_points x n_dims]
    centroids_ptr,  # Pointer to centroids [n_clusters x n_dims]
    assignments_ptr,  # Pointer to cluster assignments [n_points]
    loss_ptr,  # Pointer to output loss values [n_blocks]
    n_points,  # Number of data points
    n_dims,  # Number of dimensions
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Triton kernel for computing within-cluster sum of squares (WCSS)
    """
    # Get program ID and compute point indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_points
    
    # Load assignments
    assignments = tl.load(assignments_ptr + offsets, mask=mask, other=0)
    
    # Load points
    point_offsets = offsets[:, None] * n_dims + tl.arange(0, n_dims)[None, :]
    points = tl.load(points_ptr + point_offsets, mask=mask[:, None], other=0.0)
    
    # Compute distances to assigned centroids
    block_loss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(BLOCK_SIZE):
        if mask[i]:
            cluster_id = assignments[i]
            
            # Load centroid
            centroid_offsets = cluster_id * n_dims + tl.arange(0, n_dims)
            centroid = tl.load(centroids_ptr + centroid_offsets)
            
            # Compute squared distance
            diff = points[i, :] - centroid
            squared_distance = tl.sum(diff * diff)
            block_loss = tl.where(i == tl.arange(0, BLOCK_SIZE), squared_distance, block_loss)
    
    # Sum losses in this block
    total_loss = tl.sum(block_loss)
    
    # Store block result
    tl.store(loss_ptr + pid, total_loss)


@triton.jit
def initialize_centroids_kernel(
    points_ptr,  # Pointer to input data points [n_points x n_dims]
    centroids_ptr,  # Pointer to output centroids [n_clusters x n_dims]
    n_points,  # Number of data points
    n_dims,  # Number of dimensions
    n_clusters,  # Number of clusters
    seed,  # Random seed
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Triton kernel for random centroid initialization
    """
    # Get cluster and dimension indices
    cluster_id = tl.program_id(axis=0)
    dim_id = tl.program_id(axis=1)
    
    if cluster_id >= n_clusters or dim_id >= n_dims:
        return
    
    # Find min and max values for this dimension
    min_val = float('inf')
    max_val = float('-inf')
    
    for block_start in range(0, n_points, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_points
        
        # Load dimension values
        dim_offsets = offsets * n_dims + dim_id
        values = tl.load(points_ptr + dim_offsets, mask=mask, other=0.0)
        
        # Update min/max
        block_min = tl.min(tl.where(mask, values, float('inf')))
        block_max = tl.max(tl.where(mask, values, float('-inf')))
        
        min_val = tl.minimum(min_val, block_min)
        max_val = tl.maximum(max_val, block_max)
    
    # Generate random value
    rng_state = seed + cluster_id * n_dims + dim_id
    rng_state = (rng_state * 1664525 + 1013904223) % (2**32)
    rand_val = rng_state.to(tl.float32) / (2**32)
    
    # Compute random centroid coordinate
    centroid_val = min_val + rand_val * (max_val - min_val)
    
    # Store centroid coordinate
    centroid_offset = cluster_id * n_dims + dim_id
    tl.store(centroids_ptr + centroid_offset, centroid_val)


def kmeans_triton(points: torch.Tensor, n_clusters: int, max_iterations: int = 100, 
                  tolerance: float = 1e-4, init_method: str = 'random') -> tuple:
    """
    K-means clustering using Triton implementation
    
    Args:
        points: Input tensor of shape [n_points, n_dims]
        n_clusters: Number of clusters
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        init_method: Initialization method ('random' or 'kmeans++')
        
    Returns:
        Tuple of (centroids, assignments, n_iterations)
    """
    device = points.device
    n_points, n_dims = points.shape
    
    # Initialize centroids
    centroids = torch.empty((n_clusters, n_dims), device=device, dtype=torch.float32)
    
    if init_method == 'random':
        # Use Triton kernel for initialization
        grid = (n_clusters, n_dims)
        BLOCK_SIZE = 256
        initialize_centroids_kernel[grid](
            points, centroids, n_points, n_dims, n_clusters, 
            torch.randint(0, 2**31, (1,)).item(), BLOCK_SIZE
        )
    else:
        # Simple random initialization fallback
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
        kmeans_assign_points_kernel[grid_points](
            points, centroids, assignments, distances,
            n_points, n_dims, n_clusters, BLOCK_SIZE
        )
        
        # Step 2: Update centroids
        cluster_counts.zero_()
        centroids.zero_()
        
        kmeans_update_centroids_kernel[grid_clusters](
            points, assignments, centroids, cluster_counts,
            n_points, n_dims, n_clusters, BLOCK_SIZE
        )
        
        # Step 3: Check convergence
        if iteration % 5 == 0:
            # Compute loss
            loss_blocks = torch.zeros(grid_points, device=device, dtype=torch.float32)
            kmeans_compute_loss_kernel[grid_points](
                points, centroids, assignments, loss_blocks,
                n_points, n_dims, BLOCK_SIZE
            )
            
            current_loss = loss_blocks.sum().item() / n_points
            
            if abs(current_loss - prev_loss) < tolerance:
                break
            
            prev_loss = current_loss
    
    return centroids, assignments, iteration + 1


def kmeans_triton_multiple_runs(points: torch.Tensor, n_clusters: int, n_runs: int = 10, 
                               max_iterations: int = 100, tolerance: float = 1e-4) -> tuple:
    """
    Run k-means multiple times and return the best result
    
    Args:
        points: Input tensor of shape [n_points, n_dims]
        n_clusters: Number of clusters
        n_runs: Number of runs
        max_iterations: Maximum number of iterations per run
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (best_centroids, best_assignments, best_loss)
    """
    best_loss = float('inf')
    best_centroids = None
    best_assignments = None
    
    for run in range(n_runs):
        centroids, assignments, _ = kmeans_triton(
            points, n_clusters, max_iterations, tolerance
        )
        
        # Compute final loss
        BLOCK_SIZE = 256
        grid_points = triton.cdiv(points.shape[0], BLOCK_SIZE)
        loss_blocks = torch.zeros(grid_points, device=points.device, dtype=torch.float32)
        
        kmeans_compute_loss_kernel[grid_points](
            points, centroids, assignments, loss_blocks,
            points.shape[0], points.shape[1], BLOCK_SIZE
        )
        
        total_loss = loss_blocks.sum().item()
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_centroids = centroids.clone()
            best_assignments = assignments.clone()
    
    return best_centroids, best_assignments, best_loss


# Example usage and testing functions
def test_kmeans_triton():
    """Test the Triton k-means implementation"""
    # Generate synthetic data
    torch.manual_seed(42)
    n_points, n_dims, n_clusters = 1000, 2, 3
    
    # Create clustered data
    cluster_centers = torch.tensor([[0, 0], [3, 3], [-2, 4]], dtype=torch.float32)
    points = []
    
    for center in cluster_centers:
        cluster_points = center + 0.5 * torch.randn(n_points // n_clusters, n_dims)
        points.append(cluster_points)
    
    points = torch.cat(points, dim=0).cuda()
    
    # Run k-means
    centroids, assignments, n_iter = kmeans_triton(points, n_clusters)
    
    print(f"Converged in {n_iter} iterations")
    print(f"Final centroids:\n{centroids}")
    print(f"Cluster sizes: {torch.bincount(assignments)}")
    
    return centroids, assignments


if __name__ == "__main__":
    test_kmeans_triton()
