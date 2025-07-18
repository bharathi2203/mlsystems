# GPU Kernels Collection

This directory contains hand-optimized GPU kernels organized by functionality rather than programming language. Each algorithm is implemented across multiple platforms (CUDA, Metal, Triton) to demonstrate cross-platform optimization techniques.

## Organization

### `fundamentals/` - Core GPU Programming Patterns
Basic operations that form the building blocks of more complex algorithms:
- **vector_ops/**: Element-wise operations, dot products
- **memory_patterns/**: Matrix transpose, copy operations, memory coalescing
- **reductions/**: Parallel reduction patterns, prefix sums, histograms

### `linear_algebra/` - Dense Linear Algebra
Core mathematical operations for ML and scientific computing:
- **matmul/**: Matrix multiplication variants (naive, tiled, tensor cores)
- **gemm/**: General matrix multiply with different precisions

### `convolutions/` - Convolution Operations
Essential for computer vision and signal processing:
- 1D, 2D, and 3D convolution implementations
- Different optimization strategies (im2col, Winograd, FFT)

### `ml_primitives/` - Machine Learning Building Blocks
Common operations in neural networks:
- **activations/**: ReLU, Softmax, Leaky ReLU implementations
- **attention/**: Multi-head attention, Flash attention variants
- **losses/**: MSE, Cross-entropy loss functions

### `algorithms/` - Complex Algorithms
Higher-level algorithmic implementations:
- **sorting/**: Merge sort, radix sort, bitonic sort
- **clustering/**: K-means, hierarchical clustering
- **optimization/**: Gradient descent variants, regression algorithms

### `specialized/` - Advanced and Experimental
Cutting-edge or domain-specific implementations:
- Quantized operations
- Sparse matrix operations
- Domain-specific algorithms

## Platform Support

Each algorithm directory may contain:
- `*.cu` - CUDA implementations
- `*.metal` - Metal Performance Shaders
- `*.py` - Triton implementations
- `README.md` - Algorithm explanation and performance notes
- `bench/` - Benchmark scripts and results

## Learning Path

1. **Start with fundamentals/** - Understand basic GPU programming patterns
2. **Progress to linear_algebra/** - Learn optimization techniques for dense math
3. **Explore ml_primitives/** - See how ML operations are implemented
4. **Study algorithms/** - Understand complex algorithmic implementations
5. **Experiment with specialized/** - Explore advanced topics

## Building and Running

See the platform-specific guides in `/platforms/` for setup instructions. 