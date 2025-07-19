# Systems / ML Programming Portfolio/Journal

Stuff I'm currently exploring.

## Repository Structure

### [`kernels/`](./kernels/) - GPU Kernel Implementations
GPU kernels organized (roughly) by functionality:
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
- **ray_tracer/**: Multi-platform ray tracing engine

### [`training/`](./training/) - Distributed Training & Optimization
Advanced training techniques and optimization strategies:
- **distributed/**: Multi-GPU and multi-node training
- **optimization/**: Knowledge distillation, quantization
- **profiling/**: Performance analysis and GPU utilization

### [`reading/`](./reading/) - Things I've read 
Summaries and notes from various learning materials:
- **2025/**: Monthly reading summaries and notes
- **topics/**: Organized by subject area
- **references/**: Quick reference materials and cheat sheets
