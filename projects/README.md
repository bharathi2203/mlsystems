# Complete Projects and Applications

This directory contains complete, end-to-end implementations that showcase how individual kernels and techniques come together to build full applications and systems.

## Current Projects

### `ray_tracer/`
A multi-platform ray tracing implementation demonstrating:
- Cross-platform compute (CUDA, Metal, CPU)
- Memory management and optimization
- Performance comparison across platforms
- Real-time rendering techniques

**Features:**
- Sphere and triangle primitive support
- Basic lighting and shading models
- Configurable scene descriptions
- Performance benchmarking

### `inference_engines/`
Machine learning inference optimization projects:
- Model pruning and compression techniques
- Quantization strategies
- Custom inference kernels
- Performance profiling and optimization

**Features:**
- PyTorch model optimization
- Custom CUDA kernels for inference
- Memory usage analysis
- Latency optimization

## Project Structure

Each project follows a consistent structure:
```
project_name/
├── README.md           # Project-specific documentation
├── src/               # Source code
├── include/           # Header files
├── tests/             # Unit and integration tests
├── benchmarks/        # Performance benchmarks
├── examples/          # Usage examples
├── docs/              # Additional documentation
└── CMakeLists.txt     # Build configuration
```

## Building Projects

### Prerequisites
- Follow platform setup in `/platforms/`
- Install project-specific dependencies (see individual READMEs)

### Build All Projects
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Individual Project
```bash
cd projects/ray_tracer
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Contributing New Projects

When adding a new project:
1. Create descriptive README.md
2. Include comprehensive tests
3. Add performance benchmarks
4. Document build requirements
5. Provide usage examples
6. Consider cross-platform compatibility

## Project Goals

These projects demonstrate:
- **Integration**: How individual kernels work together
- **Optimization**: Real-world performance tuning
- **Cross-platform**: Multi-GPU vendor support
- **Best Practices**: Production-ready code patterns
- **Documentation**: Clear explanation of design decisions 