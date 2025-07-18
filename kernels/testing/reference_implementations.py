#!/usr/bin/env python3
"""
Reference implementations for kernel validation

These implementations use PyTorch/NumPy to provide ground truth
results for testing custom kernel implementations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union

class ReferenceImplementations:
    """Collection of reference implementations for kernel validation"""
    
    @staticmethod
    def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Reference dot product implementation"""
        return torch.dot(a.flatten(), b.flatten())
    
    @staticmethod
    def elemwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Reference element-wise addition"""
        return a + b
    
    @staticmethod
    def elemwise_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Reference element-wise multiplication"""
        return a * b
    
    @staticmethod
    def matrix_transpose(x: torch.Tensor) -> torch.Tensor:
        """Reference matrix transpose"""
        return x.T
    
    @staticmethod
    def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Reference matrix multiplication"""
        return torch.matmul(a, b)
    
    @staticmethod
    def relu_activation(x: torch.Tensor) -> torch.Tensor:
        """Reference ReLU activation"""
        return torch.relu(x)
    
    @staticmethod
    def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
        """Reference Leaky ReLU activation"""
        return F.leaky_relu(x, negative_slope)
    
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        """Reference GELU activation"""
        return F.gelu(x)
    
    @staticmethod
    def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Reference ELU activation"""
        return F.elu(x, alpha)
    
    @staticmethod
    def mish(x: torch.Tensor) -> torch.Tensor:
        """Reference Mish activation"""
        return x * torch.tanh(F.softplus(x))
    
    @staticmethod
    def swiglu(x: torch.Tensor) -> torch.Tensor:
        """Reference SwiGLU activation"""
        x_chunks = x.chunk(2, dim=-1)
        return F.silu(x_chunks[0]) * x_chunks[1]
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Reference softmax"""
        return F.softmax(x, dim=dim)
    
    @staticmethod
    def layer_norm(x: torch.Tensor, normalized_shape: int, 
                   eps: float = 1e-5) -> torch.Tensor:
        """Reference layer normalization"""
        return F.layer_norm(x, (normalized_shape,), eps=eps)
    
    @staticmethod
    def rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Reference RMS normalization"""
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps)
    
    @staticmethod
    def group_norm(x: torch.Tensor, num_groups: int, 
                   eps: float = 1e-5) -> torch.Tensor:
        """Reference group normalization"""
        return F.group_norm(x, num_groups, eps=eps)
    
    @staticmethod
    def cross_entropy(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Reference cross entropy loss"""
        return F.cross_entropy(input, target)
    
    @staticmethod
    def mse_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Reference MSE loss"""
        return F.mse_loss(input, target)
    
    @staticmethod
    def kl_divergence(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Reference KL divergence"""
        return F.kl_div(F.log_softmax(input, dim=-1), 
                       F.softmax(target, dim=-1), reduction='batchmean')
    
    @staticmethod
    def focal_loss(input: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Reference focal loss"""
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def contrastive_loss(embedding1: torch.Tensor, embedding2: torch.Tensor,
                        labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """Reference contrastive loss"""
        distances = F.pairwise_distance(embedding1, embedding2)
        losses = labels * distances.pow(2) + \
                (1 - labels) * F.relu(margin - distances).pow(2)
        return losses.mean()
    
    @staticmethod
    def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor,
                    negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        """Reference triplet loss"""
        return F.triplet_margin_loss(anchor, positive, negative, margin)
    
    @staticmethod
    def conv1d(input: torch.Tensor, weight: torch.Tensor, 
               stride: int = 1, padding: int = 0) -> torch.Tensor:
        """Reference 1D convolution"""
        return F.conv1d(input, weight, stride=stride, padding=padding)
    
    @staticmethod
    def conv2d(input: torch.Tensor, weight: torch.Tensor,
               stride: int = 1, padding: int = 0) -> torch.Tensor:
        """Reference 2D convolution"""
        return F.conv2d(input, weight, stride=stride, padding=padding)
    
    @staticmethod
    def conv3d(input: torch.Tensor, weight: torch.Tensor,
               stride: int = 1, padding: int = 0) -> torch.Tensor:
        """Reference 3D convolution"""
        return F.conv3d(input, weight, stride=stride, padding=padding)
    
    @staticmethod
    def prefix_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Reference prefix sum (cumulative sum)"""
        return torch.cumsum(x, dim=dim)
    
    @staticmethod
    def reduction_sum(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Reference reduction sum"""
        return torch.sum(x, dim=dim)
    
    @staticmethod
    def histogram(x: torch.Tensor, bins: int = 10, 
                 range_vals: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """Reference histogram"""
        if range_vals is None:
            range_vals = (x.min().item(), x.max().item())
        return torch.histc(x, bins=bins, min=range_vals[0], max=range_vals[1])
    
    @staticmethod
    def frobenius_norm(x: torch.Tensor) -> torch.Tensor:
        """Reference Frobenius norm"""
        return torch.norm(x, p='fro')
    
    @staticmethod
    def bitonic_sort(x: torch.Tensor, descending: bool = False) -> torch.Tensor:
        """Reference bitonic sort (uses PyTorch sort)"""
        return torch.sort(x, descending=descending)[0]
    
    @staticmethod
    def tensor_contraction_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Reference 2D tensor contraction (matrix multiply)"""
        return torch.matmul(a, b)
    
    @staticmethod
    def quantize_fp32_to_int8(x: torch.Tensor, scale: float, 
                             zero_point: int) -> torch.Tensor:
        """Reference FP32 to INT8 quantization"""
        quantized = torch.round(x / scale + zero_point)
        return torch.clamp(quantized, -128, 127).to(torch.int8)
    
    @staticmethod
    def dequantize_int8_to_fp32(x: torch.Tensor, scale: float,
                               zero_point: int) -> torch.Tensor:
        """Reference INT8 to FP32 dequantization"""
        return scale * (x.float() - zero_point)
    
    @staticmethod
    def fp32_to_fp16(x: torch.Tensor) -> torch.Tensor:
        """Reference FP32 to FP16 conversion"""
        return x.half()
    
    @staticmethod
    def fp16_to_fp32(x: torch.Tensor) -> torch.Tensor:
        """Reference FP16 to FP32 conversion"""
        return x.float()
    
    @staticmethod
    def multi_head_attention(query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Reference multi-head self-attention"""
        return F.multi_head_attention_forward(
            query, key, value, num_heads,
            in_proj_weight=None, in_proj_bias=None,
            bias_k=None, bias_v=None, add_zero_attn=False,
            dropout_p=0.0, out_proj_weight=None, out_proj_bias=None,
            training=False, key_padding_mask=None, need_weights=False,
            attn_mask=None
        )[0]
    
    @staticmethod
    def gaussian_blur(x: torch.Tensor, kernel_size: int = 5,
                     sigma: float = 1.0) -> torch.Tensor:
        """Reference Gaussian blur"""
        channels = x.shape[1] if len(x.shape) == 4 else 1
        kernel = torch.zeros(channels, 1, kernel_size, kernel_size)
        
        # Create Gaussian kernel
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x_dist = (i - center) ** 2
                y_dist = (j - center) ** 2
                kernel[:, 0, i, j] = torch.exp(-(x_dist + y_dist) / (2 * sigma ** 2))
        
        kernel = kernel / kernel.sum()
        return F.conv2d(x, kernel, padding=kernel_size//2, groups=channels)
    
    @staticmethod
    def monte_carlo_integration(func, num_samples: int = 10000,
                               domain: Tuple[float, float] = (0, 1)) -> float:
        """Reference Monte Carlo integration"""
        samples = torch.rand(num_samples) * (domain[1] - domain[0]) + domain[0]
        return (domain[1] - domain[0]) * func(samples).mean().item()
    
    @staticmethod
    def sparse_matrix_vector_mult(sparse_matrix: torch.sparse.FloatTensor,
                                 vector: torch.Tensor) -> torch.Tensor:
        """Reference sparse matrix-vector multiplication"""
        return torch.sparse.mm(sparse_matrix, vector.unsqueeze(1)).squeeze(1)
    
    @staticmethod
    def pooling_2d(x: torch.Tensor, kernel_size: int = 2,
                  stride: int = 2, pool_type: str = 'max') -> torch.Tensor:
        """Reference 2D pooling"""
        if pool_type == 'max':
            return F.max_pool2d(x, kernel_size, stride)
        elif pool_type == 'avg':
            return F.avg_pool2d(x, kernel_size, stride)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
    
    @staticmethod
    def kmeans(x: torch.Tensor, k: int, max_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference K-means clustering"""
        # Simple K-means implementation
        n_samples, n_features = x.shape
        centroids = x[torch.randperm(n_samples)[:k]]
        
        for _ in range(max_iters):
            # Assign points to closest centroid
            distances = torch.cdist(x, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    new_centroids[i] = x[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        
        return centroids, labels
    
    @staticmethod
    def adamw_step(param: torch.Tensor, grad: torch.Tensor, 
                   exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor,
                   lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                   eps: float = 1e-8, weight_decay: float = 0.01,
                   step: int = 1) -> torch.Tensor:
        """Reference AdamW optimizer step"""
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        denom = (exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(eps)
        step_size = lr / bias_correction1
        
        # Apply weight decay
        param.mul_(1 - lr * weight_decay)
        
        # Update parameters
        param.addcdiv_(exp_avg, denom, value=-step_size)
        
        return param
    
    @staticmethod
    def sgd_momentum_step(param: torch.Tensor, grad: torch.Tensor,
                         momentum_buffer: torch.Tensor,
                         lr: float = 1e-3, momentum: float = 0.9,
                         weight_decay: float = 0.0) -> torch.Tensor:
        """Reference SGD with momentum optimizer step"""
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        
        momentum_buffer.mul_(momentum).add_(grad)
        param.add_(momentum_buffer, alpha=-lr)
        
        return param

# Create a mapping for easy lookup
REFERENCE_IMPLEMENTATIONS = {
    'dot_product': ReferenceImplementations.dot_product,
    'elemwise_operations': ReferenceImplementations.elemwise_add,
    'elemwise_add': ReferenceImplementations.elemwise_add,
    'elemwise_mul': ReferenceImplementations.elemwise_mul,
    'matrix_transpose': ReferenceImplementations.matrix_transpose,
    'matmul': ReferenceImplementations.matrix_multiply,
    'matrix_multiply': ReferenceImplementations.matrix_multiply,
    'relu_activation': ReferenceImplementations.relu_activation,
    'leaky_relu': ReferenceImplementations.leaky_relu,
    'gelu': ReferenceImplementations.gelu,
    'elu': ReferenceImplementations.elu,
    'mish': ReferenceImplementations.mish,
    'swiglu': ReferenceImplementations.swiglu,
    'softmax': ReferenceImplementations.softmax,
    'layer_norm': ReferenceImplementations.layer_norm,
    'rms_norm': ReferenceImplementations.rms_norm,
    'group_norm': ReferenceImplementations.group_norm,
    'cross_entropy': ReferenceImplementations.cross_entropy,
    'mse': ReferenceImplementations.mse_loss,
    'kl_divergence': ReferenceImplementations.kl_divergence,
    'focal_loss': ReferenceImplementations.focal_loss,
    'contrastive_loss': ReferenceImplementations.contrastive_loss,
    'triplet_loss': ReferenceImplementations.triplet_loss,
    '1d_convolution': ReferenceImplementations.conv1d,
    '2d_convolution': ReferenceImplementations.conv2d,
    '3d_convolution': ReferenceImplementations.conv3d,
    'prefix_sum': ReferenceImplementations.prefix_sum,
    'cumulative_sum': ReferenceImplementations.prefix_sum,
    'reduction': ReferenceImplementations.reduction_sum,
    'partial_sum': ReferenceImplementations.reduction_sum,
    'histogram': ReferenceImplementations.histogram,
    'frobenius_norm': ReferenceImplementations.frobenius_norm,
    'bitonic_sort': ReferenceImplementations.bitonic_sort,
    'tensor_contractions': ReferenceImplementations.tensor_contraction_2d,
    'fp16_operations': ReferenceImplementations.fp32_to_fp16,
    'int8_operations': ReferenceImplementations.quantize_fp32_to_int8,
    'multi_head_self_attention': ReferenceImplementations.multi_head_attention,
    'multihead_self_attention': ReferenceImplementations.multi_head_attention,
    'softmax_attention': ReferenceImplementations.multi_head_attention,
    'gaussian_blur': ReferenceImplementations.gaussian_blur,
    'monte_carlo_integration': ReferenceImplementations.monte_carlo_integration,
    'sparse_matrix_vector_mult': ReferenceImplementations.sparse_matrix_vector_mult,
    'pooling_2d': ReferenceImplementations.pooling_2d,
    'kmeans': ReferenceImplementations.kmeans,
    'kmeans_clustering': ReferenceImplementations.kmeans,
    'adamw': ReferenceImplementations.adamw_step,
    'sgd_momentum': ReferenceImplementations.sgd_momentum_step,
} 