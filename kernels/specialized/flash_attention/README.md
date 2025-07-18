# Flash Attention Implementation

Memory-efficient attention mechanisms that reduce memory complexity from O(N²) to O(N) while maintaining mathematical equivalence to standard attention.

## Implementations

### `flash_attention_forward.cu`
Forward pass implementation of Flash Attention:
- **Tiled computation**: Processes attention in blocks to fit in SRAM
- **Online softmax**: Computes softmax incrementally without storing full matrices
- **Memory efficiency**: Reduces HBM access by 5-10x compared to standard attention
- **Numerical stability**: Uses online normalization to prevent overflow

**Key Features:**
- Block-wise attention computation
- Incremental softmax with running statistics
- Optimized memory access patterns
- Support for causal and non-causal attention

### `flash_attention_backward.cu`
Backward pass with gradient computation:
- **Recomputation strategy**: Recomputes attention on-the-fly during backward pass
- **Memory optimization**: Avoids storing large intermediate matrices
- **Gradient accuracy**: Maintains numerical precision for training

### `flash_attention_v2.cu`
Enhanced version with additional optimizations:
- **Improved tiling strategy**: Better work distribution across SMs
- **Warp-level optimizations**: Uses cooperative groups for efficiency
- **Mixed precision support**: FP16/BF16 compute with FP32 accumulation

## Performance Characteristics



## Mathematical Foundation

Flash Attention maintains mathematical equivalence to standard attention:

```
Attention(Q, K, V) = softmax(QK^T / √d)V
```

But computes it in blocks:
1. **Tile loading**: Load Q_i, K_j, V_j blocks into SRAM
2. **Local attention**: Compute S_ij = Q_i K_j^T
3. **Online softmax**: Update running max and sum statistics
4. **Output update**: Accumulate partial results with proper scaling

## Usage Examples

```cuda
// Launch Flash Attention kernel
dim3 grid(num_heads, batch_size);
dim3 block(BLOCK_SIZE);

flash_attention_forward<<<grid, block, smem_size>>>(
    Q, K, V, O,           // Input/output tensors
    seq_len, head_dim,    // Dimensions
    scale                 // Attention scale factor
);
```

## Integration with Transformers

Flash Attention is a drop-in replacement for standard attention:
- **Same API**: Compatible with existing transformer implementations
- **Better scaling**: Enables training on longer sequences
- **Memory efficiency**: Allows larger batch sizes or models

## Next Steps

1. **Study the tiling strategy** in the forward pass
2. **Understand online softmax** computation
3. **Implement causal attention** variants
4. **Explore Flash Attention v2** optimizations
5. **Benchmark against** standard attention on your hardware 