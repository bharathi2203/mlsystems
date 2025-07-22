# CUDA Performance Checklist

*Based on Mark Sim's CUDA Performance Lecture*

## Overview

This document outlines the key performance optimization techniques for CUDA kernels, following a profiling-first approach. The goal is to achieve optimal GPU utilization through systematic optimization strategies.

## Core Performance Principles

### 1. Memory Hierarchy Understanding

**SRAM vs DRAM:**
- **SRAM (Shared Memory)**: On-chip, ~KB range, fastest access
- **DRAM (Global Memory)**: Off-chip, ~GB range, slower access
- **L1 Cache**: ~10x faster than global memory
- **L2 Cache**: ~1.5x faster than global memory

**Key Insight**: Latency is fundamentally hard to reduce - GPUs hide latency through parallelism rather than reducing it.

### 2. Memory Coalescing

**Principle**: Access contiguous memory locations to maximize memory bandwidth utilization.

**Implementation**:
```cuda
// Good: Coalesced access
int idx = blockIdx.x * blockDim.x + threadIdx.x;
output[idx] = input[idx];

// Bad: Non-coalesced access (strided)
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2 % n;
output[idx] = input[idx];
```

**Impact**: Can provide 20-30% performance improvement.

### 3. Occupancy Optimization

**Definition**: Ratio of active warps to maximum possible warps on a streaming multiprocessor.

**Optimization Techniques**:
- Use CUDA Occupancy Calculator
- Adjust block size and grid size
- Consider tile quantization and wave quantization

**Tools**:
```cuda
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize,
    kernelFunction, sharedMemSize, maxBlockSize
);
```

### 4. Control Divergence Minimization

**Problem**: Warps execute in lockstep - if threads diverge, performance drops significantly.

**Solutions**:
- Avoid nested if-conditions
- Use branchless programming techniques
- Rewrite conditions using algebraic tricks

**Example**:
```cuda
// Instead of if-else
if (data[idx] % 2 == 0) {
    result[idx] = data[idx] * 2;
} else {
    result[idx] = data[idx] + 1;
}

// Use branchless approach
int is_even = (data[idx] % 2 == 0);
result[idx] = data[idx] * (2 * is_even) + (1 * (1 - is_even));
```

**Impact**: Can provide 3x speedup in extreme cases.

### 5. Thread Coarsening

**Principle**: For memory-bound kernels, do more work per thread to reduce memory access overhead.

**Implementation**:
```cuda
// Regular vector addition
int idx = blockIdx.x * blockDim.x + threadIdx.x;
c[idx] = a[idx] + b[idx];

// Coarsened (factor of 2)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
c[idx] = a[idx] + b[idx];
c[idx + 1] = a[idx + 1] + b[idx + 1];
```

**Impact**: Can provide 10-30x speedup for memory-bound operations.

### 6. Privatization

**Principle**: Use local copies of data to avoid repeated global memory accesses.

**Applications**:
- Sliding window algorithms
- Partial updates to private copies
- Shared memory utilization

### 7. Algorithmic Rewriting

**Principle**: Sometimes mathematical reformulation can dramatically improve performance.

**Example**: Online Softmax
- Traditional softmax requires two passes over data
- Online softmax uses progressive normalization
- Eliminates overflow issues while maintaining performance

## Performance Analysis Framework

### Roof Line Model

**Operational Intensity**: FLOPS / Memory Bytes
- **Memory-bound**: < 1 FLOPS/byte
- **Compute-bound**: > 1 FLOPS/byte

**Optimization Strategy**:
- **Memory-bound**: Focus on fusions, quantization, compilation
- **Compute-bound**: Focus on better algorithms

### Profiling Tools

#### Nsight Compute (ncu)

**Primary Use**: Detailed kernel-level profiling and analysis

**Basic Usage**:
```bash
# Profile a single kernel launch
ncu --kernel-name myKernel ./my_app

# Profile with specific metrics
ncu --metrics dram__bytes.sum,l1tex__t_bytes.sum ./my_app

# Export to CSV for analysis
ncu --csv --log-file profile.csv ./my_app

# Profile with sampling (faster)
ncu --sampling-interval 1 ./my_app
```

**Key Metrics to Monitor**:
- **Memory Metrics**:
  - `dram__bytes.sum`: Global memory throughput
  - `l1tex__t_bytes.sum`: L1 cache throughput
  - `l2tex__t_bytes.sum`: L2 cache throughput
  - `shared__bytes.sum`: Shared memory usage
- **Compute Metrics**:
  - `sm__cycles_elapsed.avg`: SM cycles
  - `sm__warps_active.avg`: Active warps
  - `sm__sass_thread_inst_executed_op_*`: Instruction counts
- **Occupancy**:
  - `sm__maximum_warps_per_active_cycle_pct`: Peak occupancy
  - `sm__warps_active.avg`: Average active warps

**Advanced Features**:
```bash
# Source-level analysis
ncu --source-level-analysis ./my_app

# Kernel replay for detailed analysis
ncu --kernel-replay ./my_app

# Custom metrics
ncu --metrics sm__cycles_elapsed.avg,sm__warps_active.avg ./my_app
```

#### Nsight Systems (nsys)

**Primary Use**: System-level profiling and timeline analysis

**Basic Usage**:
```bash
# Basic system profile
nsys profile ./my_app

# Profile with specific options
nsys profile --stats=true --force-overwrite ./my_app

# Profile specific time range
nsys profile --capture-range=cudaProfilerApi ./my_app

# Export to various formats
nsys export --type sqlite profile.qdrep
nsys export --type chrome-tracing profile.qdrep
```

**Key Features**:
- **Timeline View**: Visualize CPU/GPU timeline
- **Kernel Analysis**: Identify kernel launch patterns
- **Memory Operations**: Track memory allocations/transfers
- **API Calls**: Monitor CUDA API usage

#### nvprof (Legacy)

**Note**: Deprecated in favor of Nsight Compute, but still useful for older systems

**Basic Usage**:
```bash
# Basic profiling
nvprof ./my_app

# Profile specific metrics
nvprof --metrics dram_read_throughput,dram_write_throughput ./my_app

# Timeline analysis
nvprof --print-gpu-trace ./my_app

# Export to CSV
nvprof --csv --log-file profile.csv ./my_app
```

#### Nsight Graphics (for Graphics Workloads)

**Primary Use**: Graphics API profiling (OpenGL, Vulkan, DirectX)

**Usage**:
```bash
# Profile graphics application
nsight-graphics ./graphics_app

# Command line profiling
nsight-graphics --profile ./graphics_app
```

#### Visual Profiler (GUI Tool)

**Features**:
- Interactive timeline analysis
- Kernel analysis with source correlation
- Memory access pattern visualization
- Occupancy analysis

**Launch**:
```bash
nvvp  # Launches Visual Profiler GUI
```

### Profiling Best Practices

#### 1. Profiling Strategy

**Three-Level Approach**:
1. **System Level** (nsys): Identify bottlenecks across the entire application
2. **Kernel Level** (ncu): Detailed analysis of specific kernels
3. **Source Level** (ncu with source analysis): Line-by-line optimization

#### 2. Metric Selection

**Memory-Bound Kernels**:
```bash
ncu --metrics dram__bytes.sum,l1tex__t_bytes.sum,shared__bytes.sum ./app
```

**Compute-Bound Kernels**:
```bash
ncu --metrics sm__cycles_elapsed.avg,sm__sass_thread_inst_executed_op_* ./app
```

**Occupancy Analysis**:
```bash
ncu --metrics sm__warps_active.avg,sm__maximum_warps_per_active_cycle_pct ./app
```

#### 3. Performance Bottleneck Identification

**Memory Bottlenecks**:
- Low DRAM throughput
- High L1/L2 cache miss rates
- Poor memory coalescing patterns

**Compute Bottlenecks**:
- Low instruction throughput
- High control divergence
- Poor occupancy

**Latency Bottlenecks**:
- Kernel launch overhead
- Memory transfer delays
- Synchronization points

#### 4. Profiling Workflow

1. **Baseline Measurement**:
   ```bash
   ncu --metrics all --csv --log-file baseline.csv ./app
   ```

2. **Bottleneck Identification**:
   ```bash
   nsys profile --stats=true ./app
   ```

3. **Kernel Optimization**:
   ```bash
   ncu --kernel-name targetKernel --metrics dram__bytes.sum ./app
   ```

4. **Validation**:
   ```bash
   ncu --metrics all --csv --log-file optimized.csv ./app
   ```

#### 5. Common Profiling Commands

**Quick Performance Check**:
```bash
ncu --set full ./app
```

**Memory Analysis**:
```bash
ncu --metrics dram__bytes.sum,l1tex__t_bytes.sum,shared__bytes.sum ./app
```

**Occupancy Analysis**:
```bash
ncu --metrics sm__warps_active.avg,sm__maximum_warps_per_active_cycle_pct ./app
```

**Instruction Analysis**:
```bash
ncu --metrics sm__sass_thread_inst_executed_op_* ./app
```

### Profiling Tools Comparison

| Tool | Use Case | Detail Level | Performance Impact |
|------|----------|--------------|-------------------|
| **nsys** | System overview | Low | Minimal |
| **ncu** | Kernel analysis | High | Low |
| **nvvp** | Interactive analysis | High | Low |
| **nvprof** | Legacy support | Medium | Medium |

### Key Metrics Reference

**Memory Metrics**:
- `dram__bytes.sum`: Global memory throughput
- `l1tex__t_bytes.sum`: L1 cache throughput
- `l2tex__t_bytes.sum`: L2 cache throughput
- `shared__bytes.sum`: Shared memory usage

**Compute Metrics**:
- `sm__cycles_elapsed.avg`: SM cycles
- `sm__warps_active.avg`: Active warps
- `sm__sass_thread_inst_executed_op_*`: Instruction counts

**Occupancy Metrics**:
- `sm__maximum_warps_per_active_cycle_pct`: Peak occupancy
- `sm__warps_active.avg`: Average active warps

## Optimization Checklist

### Memory Optimizations
- [ ] Coalesce global memory accesses
- [ ] Use shared memory for frequently accessed data
- [ ] Minimize memory transactions
- [ ] Consider privatization for repeated access patterns

### Occupancy Optimizations
- [ ] Use CUDA Occupancy Calculator
- [ ] Adjust block size for target architecture
- [ ] Consider tile dimensions (powers of 2, 8, 16, 32, 128)
- [ ] Account for data type size (fp16, fp32, int8)

### Control Flow Optimizations
- [ ] Minimize branch divergence
- [ ] Use branchless programming where possible
- [ ] Avoid nested conditionals
- [ ] Consider loop unrolling for small loops

### Algorithmic Optimizations
- [ ] Implement tiling for matrix operations
- [ ] Consider thread coarsening for memory-bound kernels
- [ ] Explore mathematical reformulations
- [ ] Use specialized kernels (tensor cores, etc.)

### Profiling and Validation
- [ ] Profile with ncu to identify bottlenecks
- [ ] Measure memory vs compute bound behavior
- [ ] Validate optimizations with benchmarks
- [ ] Consider workload-specific optimizations

## Architecture-Specific Considerations

### Data Type Optimizations
- **INT8**: Multiples of 16 on A100
- **FP16**: Multiples of 64 on A100  
- **TF32**: Multiples of 32 on A100
- **FP64**: Multiples of 16 on A100

### Tensor Core Usage
- Look for operations with "ish" suffix
- Ensure proper data type alignment
- Consider mixed precision training

## Common Pitfalls

1. **Premature Optimization**: Profile first, optimize based on data
2. **Ignoring Memory Hierarchy**: Always consider SRAM vs DRAM tradeoffs
3. **Over-optimization**: Simple kernels can be faster than complex ones
4. **Architecture Assumptions**: Test on target hardware
5. **Compiler Optimizations**: Modern compilers are very good at optimization

## Resources

- **Programming Massively Parallel Processors** (PMPP) - Chapter 6.1
- **NVIDIA Tensor Core Performance Guide**
- **CUDA Best Practices Guide**
- **Citadel's GPU Architecture Papers**

## Key Takeaways

1. **Profile First**: Always measure before optimizing
2. **Memory is King**: Most optimizations focus on memory access patterns
3. **Occupancy Matters**: Higher occupancy generally means better performance
4. **Divergence is Expensive**: Avoid control flow divergence when possible
5. **Coarsening Works**: For memory-bound kernels, do more work per thread
6. **Math Matters**: Algorithmic improvements can dwarf implementation optimizations

---
