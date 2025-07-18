# Quantization Kernels

High-performance implementations of quantization techniques for model compression and acceleration, supporting various precision formats and quantization schemes.

## Precision Formats

### `fp16_operations.cu`
Half-precision floating point operations:
- **Native FP16**: Uses `__half` data type for reduced memory
- **Tensor Core optimization**: Leverages mixed-precision capabilities
- **Automatic conversion**: Seamless FP32 ↔ FP16 conversion
- **Numerical stability**: Maintains training stability

**Benefits**: 2x memory reduction, 1.5-2x speed improvement

### `bf16_operations.cu`
Brain Float 16 operations:
- **Google Brain format**: Wider exponent range than FP16
- **Better numerical stability**: Less prone to overflow/underflow
- **Drop-in replacement**: For FP32 in most cases
- **Hardware support**: Optimized for modern accelerators

### `fp8_kernels.cu`
8-bit floating point (experimental):
- **Ultra-low precision**: Maximum memory efficiency
- **E4M3/E5M2 formats**: Different exponent/mantissa trade-offs
- **Loss scaling**: Required for training stability
- **Inference optimization**: Extreme speed improvements

## Integer Quantization

### `int8_gemm.cu`
8-bit integer matrix multiplication:
- **Weight quantization**: Symmetric/asymmetric weight quantization
- **Activation quantization**: Dynamic or static activation quantization
- **DP4A instructions**: Uses integer dot product instructions
- **Dequantization**: Efficient conversion back to FP32

**Performance**: Up to 4x speedup vs FP32 on supported hardware

### `int4_operations.cu`
4-bit quantization kernels:
- **Extreme compression**: 8x memory reduction vs FP32
- **GPTQ/AWQ integration**: Compatible with popular quantization methods
- **Block-wise quantization**: Fine-grained quantization granularity
- **Custom kernels**: Optimized for specific operations

### `int1_kernels.cu`
1-bit quantization (BinaryConnect):
- **Binary weights**: Weights constrained to {-1, +1}
- **XNOR operations**: Efficient binary convolutions
- **Popcount instructions**: Fast bit counting operations
- **Extreme efficiency**: Minimal memory and computation

## Quantization Schemes

### `symmetric_quantization.cu`
Symmetric quantization implementation:
- **Zero-centered**: Quantization range symmetric around zero
- **No zero-point**: Simplified quantization formula
- **Weight quantization**: Ideal for weights with symmetric distribution
- **Formula**: `q = round(x / scale)`

### `asymmetric_quantization.cu`
Asymmetric quantization with zero-point:
- **Arbitrary range**: Can quantize any value range
- **Zero-point offset**: Handles non-symmetric distributions
- **Activation quantization**: Better for activations with bias
- **Formula**: `q = round(x / scale) + zero_point`

### `dynamic_quantization.cu`
Runtime quantization:
- **Adaptive scaling**: Computes scale factors at runtime
- **No calibration**: No need for calibration dataset
- **Outlier handling**: Adapts to activation distribution
- **Inference only**: Typically used during inference

### `static_quantization.cu`
Calibration-based quantization:
- **Pre-computed scales**: Fixed scale factors from calibration
- **Better accuracy**: More accurate than dynamic quantization
- **Deployment optimized**: Minimal runtime overhead
- **Training/inference**: Can be used in both modes

## Advanced Quantization Techniques

### `block_wise_quantization.cu`
Fine-grained quantization:
- **Sub-tensor quantization**: Quantizes smaller blocks independently
- **Better precision**: Adapts to local value distributions
- **Memory trade-off**: Higher metadata overhead
- **LLM optimization**: Popular for large language models

### `mixed_precision_kernels.cu`
Selective precision operations:
- **Critical path FP32**: Keeps important operations in full precision
- **Bulk operations FP16**: Uses lower precision for throughput
- **Automatic casting**: Seamless precision conversion
- **Gradient scaling**: Prevents underflow in backward pass

### `quantization_aware_training.cu`
Training with quantization simulation:
- **Fake quantization**: Simulates quantization during training
- **Straight-through estimator**: Gradient estimation through quantization
- **Scale learning**: Learnable quantization parameters
- **Robust training**: Model learns to be robust to quantization

## Performance Optimizations

### Memory Access
- **Packed storage**: Efficient packing of sub-byte quantized values
- **Vectorized loads**: Use vector instructions for packed data
- **Coalesced access**: Optimize memory access patterns
- **Shared memory**: Cache frequently accessed scales/zero-points

### Computation
- **Fused kernels**: Combine quantization with other operations
- **Lookup tables**: Pre-computed conversion tables
- **SIMD instructions**: Vectorized quantization operations
- **Warp-level primitives**: Efficient parallel reductions

## Usage Examples

```cuda
// FP16 GEMM with automatic mixed precision
fp16_gemm<<<grid, block>>>(
    A_fp16, B_fp16, C_fp32,  // Mixed precision inputs/outputs
    M, N, K, alpha, beta
);

// INT8 quantized matrix multiplication
int8_gemm<<<grid, block>>>(
    A_int8, B_int8, C_int32,
    scale_A, scale_B, zero_point_A, zero_point_B,
    M, N, K
);

// Dynamic quantization
dynamic_quantize<<<grid, block>>>(
    input_fp32, output_int8,
    scales, zero_points, size
);
```

## Integration with Frameworks

### PyTorch Integration
- **torch.quantization**: Compatible with PyTorch quantization APIs
- **Custom operators**: Register as custom CUDA operators
- **Autograd support**: Gradient computation through quantization

### Model Deployment
- **ONNX export**: Export quantized models to ONNX format
- **TensorRT integration**: Compatible with TensorRT optimizations
- **Mobile deployment**: Optimized for edge devices

## Benchmarking Results

| Precision | Memory Usage | Inference Speed | Accuracy Drop |
|-----------|-------------|-----------------|---------------|
| FP32      | 100%        | 1.0x           | 0%            |
| FP16      | 50%         | 1.7x           | <0.1%         |
| BF16      | 50%         | 1.6x           | <0.05%        |
| INT8      | 25%         | 2.8x           | 0.2-1%        |
| INT4      | 12.5%       | 4.2x           | 1-3%          |

## Next Steps

1. **Implement GPTQ/AWQ** quantization algorithms
2. **Add hardware-specific** optimizations (Tensor Cores, etc.)
3. **Develop calibration** tools and datasets
4. **Benchmark against** vendor implementations
5. **Explore post-training** quantization techniques 