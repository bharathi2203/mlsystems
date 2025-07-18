# Platform-Specific Build Systems and Utilities

This directory contains platform-specific build systems, common headers, and utilities for building and testing GPU kernels across different platforms.

## Directory Structure

### `cuda/`
- **common/**: Common CUDA headers, error checking utilities
- **CMakeLists.txt**: CMake build configuration for CUDA kernels
- **Makefile**: Alternative make-based build system

### `metal/`
- **common/**: Metal Performance Shaders utilities and headers
- **build_shaders.py**: Python script to compile Metal shaders
- **metal_utils.h**: Common Metal C++ wrapper utilities

### `triton/`
- **common/**: Triton kernel utilities and helpers
- **requirements.txt**: Python dependencies for Triton development
- **triton_utils.py**: Common Triton kernel patterns and utilities

## Quick Start

### CUDA Setup
```bash
cd platforms/cuda
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Metal Setup (macOS)
```bash
cd platforms/metal
python build_shaders.py --input ../../kernels --output ./compiled
```

### Triton Setup
```bash
cd platforms/triton
pip install -r requirements.txt
python -c "import triton; print('Triton ready')"
```

## Common Utilities

Each platform directory provides:
- Error checking and validation utilities
- Performance timing and benchmarking helpers
- Memory management wrappers
- Common kernel launch configurations
- Testing frameworks

## Dependencies

### CUDA
- NVIDIA CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler

### Metal
- macOS 10.15+
- Xcode 12.0+
- Metal Performance Shaders framework

### Triton
- Python 3.8+
- PyTorch 1.12+
- Triton 2.0+ 