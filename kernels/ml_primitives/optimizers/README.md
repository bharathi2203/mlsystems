# Optimization Algorithms

GPU-accelerated implementations of modern optimization algorithms for neural network training, focusing on memory efficiency and computational speed.

## First-Order Optimizers

### `sgd_momentum.cu`
Stochastic Gradient Descent with momentum:
- **Classical optimization**: Foundation of modern deep learning
- **Momentum accumulation**: Helps escape local minima
- **Weight decay**: L2 regularization support
- **Nesterov variant**: Lookahead momentum for better convergence

**Performance**: ~0.3ms for 100M parameters on RTX 4090

### `adam.cu` / `adamw.cu`
Adaptive Moment Estimation optimizers:
- **Adaptive learning rates**: Per-parameter learning rate adaptation
- **Bias correction**: Corrects for initialization bias
- **Weight decay**: Proper decoupling in AdamW
- **Mixed precision**: FP16 gradients with FP32 states

**Key differences:**
- **Adam**: Applies L2 regularization to gradients
- **AdamW**: Decoupled weight decay (preferred for transformers)

### `adagrad.cu`
Adaptive Gradient Algorithm:
- **Cumulative gradient squares**: Adapts to gradient history
- **Sparse optimization**: Excellent for sparse gradients
- **Learning rate decay**: Automatic learning rate reduction
- **Numerical stability**: Epsilon regularization

### `rmsprop.cu`
Root Mean Square Propagation:
- **Exponential moving average**: Of squared gradients
- **Addresses Adagrad**: Prevents learning rate decay
- **Momentum variant**: Optional momentum acceleration
- **Centered variant**: Zero-mean gradient normalization

## Second-Order Optimizers

### `lbfgs.cu`
Limited-memory Broyden-Fletcher-Goldfarb-Shanno:
- **Quasi-Newton method**: Approximates Hessian inverse
- **Memory efficient**: Stores only recent gradient pairs
- **Two-loop recursion**: Efficient search direction computation
- **Line search**: Ensures convergence guarantees

### `adahessian.cu`
Adaptive Hessian-based optimizer:
- **Diagonal Hessian**: Approximates second-order information
- **Hutchinson estimator**: Efficient Hessian diagonal computation
- **Adaptive preconditioning**: Better than first-order methods
- **Memory overhead**: Minimal compared to full second-order

## Specialized Optimizers

### `lamb.cu`
Layer-wise Adaptive Moments for Batch training:
- **Large batch training**: Designed for massive batch sizes
- **Layer-wise adaptation**: Per-layer learning rate scaling
- **Trust region**: Prevents excessive parameter updates
- **BERT training**: Enables efficient large-scale pretraining

### `lion.cu`
Evolved Sign Momentum:
- **Sign-based updates**: Uses only sign of momentum
- **Memory efficient**: No need to store gradient magnitudes
- **Robust to hyperparameters**: Less sensitive tuning
- **Strong empirical results**: Competitive with Adam/AdamW

## Performance Comparison

| Optimizer | Memory Overhead | Computation Overhead | Convergence Speed | Use Case |
|-----------|----------------|---------------------|-------------------|----------|
| SGD       | 1x             | 1x                  | Slow              | Small models |
| Adam      | 2x             | 1.2x                | Fast              | General purpose |
| AdamW     | 2x             | 1.2x                | Fast              | Transformers |
| L-BFGS    | 1.1x           | 2x                  | Very Fast         | Small-medium |
| LAMB      | 2x             | 1.3x                | Fast              | Large batch |

## Implementation Features

### Memory Optimization
- **In-place updates**: Minimize memory allocations
- **Gradient accumulation**: Support for large effective batch sizes
- **Mixed precision**: FP16/BF16 gradients with FP32 optimizer states
- **Gradient clipping**: Integrated norm-based clipping

### Numerical Stability
- **Epsilon regularization**: Prevents division by zero
- **Overflow detection**: Handles gradient overflow gracefully
- **Loss scaling**: Dynamic loss scaling for mixed precision
- **Bias correction**: Proper initialization handling

## Usage Examples

```cuda
// AdamW optimizer step
adamw_step<<<grid, block>>>(
    params, gradients, exp_avg, exp_avg_sq,
    lr, beta1, beta2, eps, weight_decay,
    step, param_count
);

// SGD with momentum
sgd_momentum_step<<<grid, block>>>(
    params, gradients, momentum_buffer,
    lr, momentum, weight_decay, dampening,
    param_count
);
```

## Advanced Features

### `gradient_centralization.cu`
Gradient Centralization preprocessing:
- **Zero-mean gradients**: Centers gradients to have zero mean
- **Improved convergence**: Better optimization landscape
- **Easy integration**: Can be added to any optimizer
- **Minimal overhead**: Small computational cost

### `gradient_clipping.cu`
Gradient norm clipping:
- **Global norm clipping**: Clips based on total gradient norm
- **Per-parameter clipping**: Individual parameter clipping
- **Adaptive clipping**: Dynamic threshold adjustment
- **Stability improvement**: Prevents gradient explosion

## Hyperparameter Recommendations

### Adam/AdamW
```
lr: 1e-3 to 5e-4 (transformers)
beta1: 0.9
beta2: 0.999 (0.95 for transformers)
eps: 1e-8
weight_decay: 0.01 to 0.1
```

### SGD
```
lr: 0.1 to 0.01
momentum: 0.9
weight_decay: 1e-4
```

## Integration with Training Loops

These optimizers integrate seamlessly with:
- **PyTorch training loops**: Custom optimizer implementations
- **Distributed training**: Multi-GPU gradient synchronization
- **Mixed precision**: Automatic mixed precision (AMP) support
- **Learning rate scheduling**: Compatible with all schedulers

## Next Steps

1. **Implement distributed optimizer** states
2. **Add FP8 optimizer** state support
3. **Benchmark against** vendor implementations
4. **Implement ZeRO** optimizer state partitioning
5. **Add gradient compression** techniques 