# Vector Operations

Element-wise operations that form the foundation of GPU computing. These kernels demonstrate basic thread organization, memory access patterns, and performance optimization techniques.

## Implemented Operations

### `dot_product.cu`
Parallel dot product using reduction patterns:
- Block-level reduction with shared memory
- Warp-level primitives for final reduction
- Memory coalescing optimization

**Performance**: ~800 GFLOPS on RTX 4090 (FP32)

### `elemwise_operations.cu` / `elemwise_operations.metal`
Element-wise arithmetic operations:
- Vector addition, subtraction, multiplication
- Broadcasting support for different tensor shapes
- Fused operations to minimize memory bandwidth

**Performance**: Memory bandwidth limited (~900 GB/s on RTX 4090)

## Key Concepts Demonstrated

1. **Thread Organization**
   - Grid and block dimensions
   - Thread indexing patterns
   - Boundary condition handling

2. **Memory Access**
   - Coalesced memory access patterns
   - Memory bandwidth optimization
   - Cache utilization

3. **Cross-Platform Implementation**
   - CUDA thread hierarchy
   - Metal threadgroup organization
   - Triton block programming model

## Usage Examples

```cuda
// CUDA example
dim3 block(256);
dim3 grid((n + block.x - 1) / block.x);
vector_add<<<grid, block>>>(a, b, c, n);
```

```metal
// Metal example
[encoder dispatchThreads:MTLSizeMake(n, 1, 1) 
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
```

## Performance Notes

- Vector operations are typically memory bandwidth limited
- Optimal performance achieved with large vectors (>1M elements)
- Consider tensor fusion for multiple operations
- Platform-specific optimizations available in each implementation

## Next Steps

After mastering vector operations:
1. Study memory patterns in `../memory_patterns/`
2. Learn reduction techniques in `../reductions/`
3. Implement matrix operations in `../../linear_algebra/`
4. Review learning materials in `/reading/topics/gpu_programming/` 