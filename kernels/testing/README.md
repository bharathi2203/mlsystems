# Kernel Testing Framework

Comprehensive testing suite for validating correctness and performance of CUDA, Metal, and Triton kernels.

## Quick Start

```bash
# Install dependencies
pip install torch triton pytest numpy

# Run all tests (quick)
cd kernels/testing
python run_tests.py --quick

# Test specific kernel
python run_tests.py --kernel relu_activation --verbose

# Benchmark performance
python run_tests.py --benchmark --output results.json

# Test specific category
python run_tests.py --category fundamentals
```

## Testing Features

### ✅ Correctness Validation
- Compare kernel outputs against PyTorch/NumPy reference implementations
- Configurable error tolerance for numerical precision
- Handles different data types (FP32, FP16, INT8)
- Validates edge cases and boundary conditions

### 🚀 Performance Benchmarking
- CUDA event timing for precise measurements
- Memory throughput calculations (GB/s)
- Warmup iterations to eliminate cold start effects
- Statistical analysis across multiple runs

### 🔄 Cross-Platform Testing
- **Triton**: Python-based GPU kernels (currently supported)
- **CUDA**: Native CUDA kernels (framework ready)
- **Metal**: Apple GPU shaders (framework ready)

### 📊 Automated Reporting
- JSON reports with detailed metrics
- Performance comparisons across platforms
- Error analysis and debugging information
- Test result visualization

## Usage Examples

### Basic Testing

```bash
# Test all kernels with default settings
python run_tests.py

# Quick correctness check (minimal iterations)
python run_tests.py --quick

# Verbose output with per-test details
python run_tests.py --verbose
```

### Targeted Testing

```bash
# Test specific kernel
python run_tests.py --kernel matmul

# Test kernel category
python run_tests.py --category ml_primitives

# Test specific platform
python run_tests.py --platform triton

# Test with custom sizes
python run_tests.py --sizes "1024" "64,64" "32,32,32"
```

### Performance Analysis

```bash
# Comprehensive benchmarking
python run_tests.py --benchmark --output benchmark_results.json

# High precision validation
python run_tests.py --tolerance 1e-8

# Test large tensors
python run_tests.py --sizes "1048576" "2048,2048" --benchmark
```

## Test Configuration

### TestConfig Parameters

```python
config = TestConfig(
    test_sizes=[(1024,), (32, 32), (64, 64, 64)],  # Test tensor sizes
    data_types=[torch.float32, torch.float16],      # Data types to test
    tolerance=1e-5,                                 # Error tolerance
    num_warmup=5,                                   # Warmup iterations
    num_iterations=20,                              # Benchmark iterations
    validate_correctness=True,                      # Enable correctness checks
    benchmark_performance=True                      # Enable performance measurement
)
```

### Supported Test Sizes

- **1D**: Vector operations (1K to 1M elements)
- **2D**: Matrix operations (32x32 to 4Kx4K)
- **3D**: Tensor operations (64³ to 256³)
- **Custom**: Specify via `--sizes` parameter

## Adding New Tests

### 1. Reference Implementation

Add to `reference_implementations.py`:

```python
@staticmethod
def my_new_kernel(input_tensor: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch"""
    return torch.some_operation(input_tensor)

# Add to REFERENCE_IMPLEMENTATIONS mapping
REFERENCE_IMPLEMENTATIONS['my_new_kernel'] = ReferenceImplementations.my_new_kernel
```

### 2. Kernel Implementation

Ensure your kernel follows the naming convention:

```python
# In kernels/category/my_new_kernel.py
def my_new_kernel_triton(input_tensor: torch.Tensor) -> torch.Tensor:
    """Triton implementation"""
    # Launch kernel and return result
    pass
```

### 3. Test Discovery

The framework automatically discovers kernels by:
- Scanning `kernels/` directory for `.cu` files
- Finding corresponding `.metal` and `.py` files
- Matching reference implementations by name

## Platform-Specific Testing

### Triton Kernels (Currently Supported)

```python
# Framework automatically imports and tests Triton modules
# Looks for function named: {kernel_name}_triton()
```

### CUDA Kernels (Framework Ready)

```python
# TODO: Implement CUDA compilation and testing
# Would use nvcc to compile .cu files and ctypes to call kernels
```

### Metal Kernels (Framework Ready)

```python
# TODO: Implement Metal compilation and testing  
# Would use Metal API to compile .metal shaders and execute
```

## Performance Metrics

### Throughput Calculation

```
Throughput (GB/s) = (Data Size in Bytes) / (Execution Time in Seconds) / 1e9
```

Includes both read and write operations for memory-bound kernels.

### Efficiency Targets

- **Memory-bound kernels**: >80% of theoretical memory bandwidth
- **Compute-bound kernels**: >90% of theoretical compute performance
- **Mixed workloads**: Balanced efficiency based on arithmetic intensity

## Error Analysis

### Tolerance Settings

- **Default**: 1e-5 (suitable for most FP32 operations)
- **FP16**: 1e-3 (accounts for reduced precision)
- **Strict**: 1e-8 (for high-precision requirements)

### Error Types

1. **Absolute Error**: `|output - reference|`
2. **Relative Error**: `|output - reference| / (|reference| + epsilon)`
3. **Max Error**: Maximum absolute difference across all elements

## Continuous Integration

### GitHub Actions Example

```yaml
name: Kernel Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install torch triton pytest
         - name: Run tests
       run: |
         cd kernels/testing
         python run_tests.py --quick --output ci_results.json
    - name: Upload results
      uses: actions/upload-artifact@v2
             with:
         name: test-results
         path: kernels/testing/ci_results.json
```

## Debugging Failed Tests

### Common Issues

1. **Import Errors**: Check Triton/CUDA installation
2. **Shape Mismatches**: Verify kernel output dimensions
3. **Numerical Precision**: Adjust tolerance for data type
4. **Memory Issues**: Reduce test sizes for large tensors

### Debug Commands

```bash
# Test single kernel with high verbosity
python run_tests.py --kernel problematic_kernel --verbose

# Test with relaxed tolerance
python run_tests.py --kernel problematic_kernel --tolerance 1e-3

# Test smaller sizes
python run_tests.py --kernel problematic_kernel --sizes "32" "8,8"
```

## Contributing

1. Add reference implementations for new kernels
2. Ensure kernel naming conventions are followed
3. Test across different tensor sizes and data types
4. Document any platform-specific requirements
5. Add performance benchmarks for new kernel categories

## Platform Requirements

- **Python**: 3.8+
- **PyTorch**: 1.12+
- **Triton**: 2.0+ (for Triton kernel testing)
- **CUDA**: 11.0+ (for CUDA kernel testing)
- **Metal**: macOS with Metal support (for Metal kernel testing) 