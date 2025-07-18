# Systems Programming Portfolio

A comprehensive collection of high-performance GPU kernels, distributed training implementations, and systems programming projects. This repository demonstrates deep understanding of parallel computing, memory optimization, and cross-platform development across CUDA, Metal, and Triton.

## Repository Structure

### [`kernels/`](./kernels/) - GPU Kernel Implementations
Hand-optimized GPU kernels organized by functionality:
- **fundamentals/**: Core patterns (reductions, memory operations, quantization)
- **linear_algebra/**: Matrix operations, GEMM variants
- **convolutions/**: 1D/2D/3D convolution implementations
- **ml_primitives/**: Activations, attention, normalization, optimizers, loss functions
- **algorithms/**: Sorting, clustering, optimization algorithms
- **specialized/**: Flash Attention, sparse matrices, advanced techniques

### [`platforms/`](./platforms/) - Build Systems & Utilities
Platform-specific build configurations and common utilities:
- **cuda/**: NVIDIA CUDA development environment
- **metal/**: Apple Metal Performance Shaders
- **triton/**: OpenAI Triton kernel development

### [`projects/`](./projects/) - Complete Applications
End-to-end implementations showcasing kernel integration:
- **ray_tracer/**: Multi-platform ray tracing engine
- **inference_engines/**: ML model optimization and inference

### [`training/`](./training/) - Distributed Training & Optimization
Advanced training techniques and optimization strategies:
- **distributed/**: Multi-GPU and multi-node training
- **optimization/**: Knowledge distillation, quantization
- **profiling/**: Performance analysis and GPU utilization

### [`reading/`](./reading/) - Learning Resources
Summaries and notes from various learning materials:
- **2025/**: Monthly reading summaries and notes
- **topics/**: Organized by subject area
- **references/**: Quick reference materials and cheat sheets

### [`benchmarks/`](./benchmarks/) - Performance Analysis
Comprehensive benchmarking and performance data:
- **data/**: Raw benchmark results
- **plots/**: Performance visualization
- **results/**: Analysis and conclusions

### [`docs/`](./docs/) - Documentation
Setup guides, tutorials, and reference materials

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 11.0+ (for CUDA kernels)
- macOS with Metal support (for Metal kernels)
- Python 3.8+ with PyTorch (for Triton kernels)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd systems-programming-portfolio

# Set up CUDA environment
cd platforms/cuda && mkdir build && cd build
cmake .. && make -j$(nproc)

# Set up Python environment for Triton
cd ../../platforms/triton
pip install -r requirements.txt
```

### Run Examples
```bash
# Test CUDA kernels
./platforms/cuda/build/test_kernels

# Test kernel implementations
cd kernels/fundamentals/vector_ops && nvcc dot_product.cu -o dot_product && ./dot_product

# Build and test projects
cd projects/ray_tracer && mkdir build && cd build
cmake .. && make -j$(nproc) && ./ray_tracer
```

## Highlights

### Performance Optimizations
- **Memory Coalescing**: Optimized memory access patterns
- **Shared Memory**: Efficient use of on-chip memory
- **Tensor Cores**: Mixed-precision GEMM implementations
- **Occupancy Optimization**: Maximized GPU utilization

### Cross-Platform Development
- **Algorithm-First Design**: Same algorithms across CUDA, Metal, Triton
- **Performance Comparison**: Benchmarks across different platforms
- **Platform Abstractions**: Common interfaces for different backends

### Production-Ready Code
- **Error Handling**: Comprehensive error checking and validation
- **Testing**: Unit tests and integration tests for all kernels
- **Documentation**: Detailed explanations and performance analysis
- **Benchmarking**: Quantified performance metrics

## Learning Path

1. **GPU Fundamentals** → Start with `/kernels/fundamentals/vector_ops/`
2. **Memory Optimization** → Explore `/kernels/fundamentals/memory_patterns/`
3. **Linear Algebra** → Study `/kernels/linear_algebra/matmul/`
4. **ML Primitives** → Implement `/kernels/ml_primitives/attention/`
5. **Complete Projects** → Build `/projects/ray_tracer/`

## Performance Results

| Kernel | CUDA (ms) | Metal (ms) | Speedup | GPU |
|--------|-----------|------------|---------|-----|
| Matrix Multiply (4096x4096) | 2.3 | 3.1 | 1.35x | RTX 4090 |
| Convolution (1024x1024) | 0.8 | 1.2 | 1.5x | RTX 4090 |
| Attention (seq=2048) | 1.5 | 2.1 | 1.4x | RTX 4090 |

*See `/benchmarks/` for complete performance analysis*

## Contributing

This is a personal learning and portfolio repository. The code demonstrates:
- **Systems Programming Expertise**: Low-level optimization and platform knowledge
- **Performance Engineering**: Quantified optimization results
- **Cross-Platform Development**: Multi-vendor GPU programming
- **Research Application**: Implementation of cutting-edge algorithms

## Contact

This repository serves as a technical portfolio demonstrating expertise in:
- High-performance computing and GPU programming
- Machine learning systems and optimization
- Cross-platform development and performance analysis
- Research implementation and benchmarking

Built with passion for systems programming and performance optimization.