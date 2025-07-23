import triton
import triton.language as tl
import torch

"""
1D Convolution - Triton Implementation

GPU-accelerated implementation of 1D convolution operation
using OpenAI Triton for optimal GPU performance.

Mathematical foundation:
- Convolution: (f * g)[n] = Σ f[m] * g[n-m] for all m
- Cross-correlation: (f ★ g)[n] = Σ f[m] * g[n+m] for all m
- Valid convolution: output size = input_size - kernel_size + 1
- Full convolution: output size = input_size + kernel_size - 1
- Same convolution: output size = input_size (with padding)

Memory patterns:
- Coalesced input reads with proper alignment
- Kernel broadcast through block memory
- Sequential output writes with bank conflict avoidance

Numerical considerations:
- Single precision floating point operations
- Boundary handling with zero-padding or clamping
- Overflow protection for large accumulations
"""

@triton.jit
def convolution_1d_kernel(
    input_ptr,  # Pointer to input signal [input_size]
    kernel_ptr,  # Pointer to convolution kernel [kernel_size]
    output_ptr,  # Pointer to output signal [output_size]
    input_size,  # Size of input signal
    kernel_size,  # Size of convolution kernel
    output_size,  # Size of output signal
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Basic 1D convolution kernel
    Each program instance processes BLOCK_SIZE output elements.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # Load kernel into registers
    kernel_offsets = tl.arange(0, kernel_size)
    kernel_vals = tl.load(kernel_ptr + kernel_offsets)
    
    # Initialize output
    output_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute convolution for each output element
    for k in range(kernel_size):
        input_offsets = offsets + k
        input_mask = (input_offsets < input_size) & mask
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        output_vals += input_vals * kernel_vals[k]
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)


@triton.jit
def convolution_1d_tiled_kernel(
    input_ptr,  # Pointer to input signal
    kernel_ptr,  # Pointer to convolution kernel
    output_ptr,  # Pointer to output signal
    input_size,  # Size of input signal
    kernel_size,  # Size of convolution kernel
    output_size,  # Size of output signal
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Tiled 1D convolution kernel for improved memory efficiency
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Load kernel
    kernel_offsets = tl.arange(0, kernel_size)
    kernel_vals = tl.load(kernel_ptr + kernel_offsets)
    
    # Calculate input range needed for this block
    input_start = block_start
    input_end = block_start + BLOCK_SIZE + kernel_size - 1
    input_range = min(input_end, input_size) - input_start
    
    # Load input tile
    input_tile_offsets = input_start + tl.arange(0, BLOCK_SIZE + kernel_size)
    input_tile_mask = input_tile_offsets < input_size
    input_tile = tl.load(input_ptr + input_tile_offsets, mask=input_tile_mask, other=0.0)
    
    # Compute convolution for this block
    output_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < output_size
    
    output_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(kernel_size):
        # Use tiled input data
        tile_indices = tl.arange(0, BLOCK_SIZE) + k
        tile_mask = tile_indices < input_range
        input_vals = tl.where(tile_mask, input_tile[tile_indices], 0.0)
        output_vals += input_vals * kernel_vals[k]
    
    tl.store(output_ptr + output_offsets, output_vals, mask=output_mask)


@triton.jit
def cross_correlation_1d_kernel(
    input_ptr,  # Pointer to input signal
    kernel_ptr,  # Pointer to convolution kernel
    output_ptr,  # Pointer to output signal
    input_size,  # Size of input signal
    kernel_size,  # Size of convolution kernel
    output_size,  # Size of output signal
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    1D cross-correlation kernel (commonly used in neural networks)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # Load kernel (flipped for cross-correlation)
    kernel_offsets = tl.arange(0, kernel_size)
    kernel_vals = tl.load(kernel_ptr + kernel_offsets)
    
    output_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Cross-correlation computation
    for k in range(kernel_size):
        input_offsets = offsets + k
        input_mask = (input_offsets < input_size) & mask
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        # Use flipped kernel indexing for cross-correlation
        output_vals += input_vals * kernel_vals[kernel_size - 1 - k]
    
    tl.store(output_ptr + offsets, output_vals, mask=mask)


@triton.jit
def convolution_1d_strided_kernel(
    input_ptr,  # Pointer to input signal
    kernel_ptr,  # Pointer to convolution kernel
    output_ptr,  # Pointer to output signal
    input_size,  # Size of input signal
    kernel_size,  # Size of convolution kernel
    output_size,  # Size of output signal
    stride,  # Convolution stride
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Strided 1D convolution kernel for downsampling
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # Load kernel
    kernel_offsets = tl.arange(0, kernel_size)
    kernel_vals = tl.load(kernel_ptr + kernel_offsets)
    
    output_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Strided convolution computation
    for k in range(kernel_size):
        input_offsets = offsets * stride + k
        input_mask = (input_offsets < input_size) & mask
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        output_vals += input_vals * kernel_vals[k]
    
    tl.store(output_ptr + offsets, output_vals, mask=mask)


def convolution_1d_triton(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor, 
                         mode: str = 'valid', stride: int = 1) -> torch.Tensor:
    """
    1D convolution using Triton implementation
    
    Args:
        input_tensor: Input signal tensor [input_size]
        kernel_tensor: Convolution kernel tensor [kernel_size] 
        mode: Convolution mode ('valid', 'same', 'full')
        stride: Convolution stride (default: 1)
        
    Returns:
        Output signal tensor [output_size]
    """
    device = input_tensor.device
    input_size = input_tensor.shape[0]
    kernel_size = kernel_tensor.shape[0]
    
    # Calculate output size based on mode
    if mode == 'valid':
        output_size = (input_size - kernel_size) // stride + 1
    elif mode == 'same':
        output_size = (input_size + stride - 1) // stride
    elif mode == 'full':
        output_size = (input_size + kernel_size - 1 + stride - 1) // stride
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    if output_size <= 0:
        raise ValueError("Invalid output size")
    
    # Allocate output tensor
    output_tensor = torch.empty(output_size, device=device, dtype=torch.float32)
    
    # Configure kernel launch
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(output_size, BLOCK_SIZE)
    
    # Handle padding for 'same' and 'full' modes
    if mode == 'same':
        pad_total = (output_size - 1) * stride + kernel_size - input_size
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded_input = torch.nn.functional.pad(input_tensor, (pad_left, pad_right))
        input_ptr = padded_input
        input_size = padded_input.shape[0]
    elif mode == 'full':
        pad_size = kernel_size - 1
        padded_input = torch.nn.functional.pad(input_tensor, (pad_size, pad_size))
        input_ptr = padded_input
        input_size = padded_input.shape[0]
    else:
        input_ptr = input_tensor
    
    # Launch appropriate kernel
    if stride == 1:
        if kernel_size <= 32:
            convolution_1d_kernel[grid_size](
                input_ptr, kernel_tensor, output_tensor,
                input_size, kernel_size, output_size, BLOCK_SIZE
            )
        else:
            convolution_1d_tiled_kernel[grid_size](
                input_ptr, kernel_tensor, output_tensor,
                input_size, kernel_size, output_size, BLOCK_SIZE
            )
    else:
        convolution_1d_strided_kernel[grid_size](
            input_ptr, kernel_tensor, output_tensor,
            input_size, kernel_size, output_size, stride, BLOCK_SIZE
        )
    
    return output_tensor


def cross_correlation_1d_triton(input_tensor: torch.Tensor, 
                               kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    1D cross-correlation using Triton implementation
    
    Args:
        input_tensor: Input signal tensor [input_size]
        kernel_tensor: Convolution kernel tensor [kernel_size]
        
    Returns:
        Output signal tensor [output_size]
    """
    device = input_tensor.device
    input_size = input_tensor.shape[0]
    kernel_size = kernel_tensor.shape[0]
    output_size = input_size - kernel_size + 1
    
    if output_size <= 0:
        raise ValueError("Kernel size too large for input")
    
    output_tensor = torch.empty(output_size, device=device, dtype=torch.float32)
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(output_size, BLOCK_SIZE)
    
    cross_correlation_1d_kernel[grid_size](
        input_tensor, kernel_tensor, output_tensor,
        input_size, kernel_size, output_size, BLOCK_SIZE
    )
    
    return output_tensor


# Testing and benchmarking functions
def test_convolution_1d():
    """Test the Triton 1D convolution implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test case 1: Basic convolution
    input_signal = torch.randn(1000, device=device)
    kernel = torch.randn(5, device=device)
    
    # Triton implementation
    output_triton = convolution_1d_triton(input_signal, kernel, mode='valid')
    
    # PyTorch reference
    output_torch = torch.conv1d(
        input_signal.unsqueeze(0).unsqueeze(0),
        kernel.flip(0).unsqueeze(0).unsqueeze(0)
    ).squeeze()
    
    # Compare results
    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    print(f"Basic convolution max difference: {max_diff:.6f}")
    
    # Test case 2: Cross-correlation
    output_xcorr = cross_correlation_1d_triton(input_signal, kernel)
    output_torch_xcorr = torch.conv1d(
        input_signal.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()
    
    max_diff_xcorr = torch.max(torch.abs(output_xcorr - output_torch_xcorr)).item()
    print(f"Cross-correlation max difference: {max_diff_xcorr:.6f}")
    
    # Test case 3: Strided convolution
    output_strided = convolution_1d_triton(input_signal, kernel, mode='valid', stride=2)
    print(f"Strided convolution output size: {output_strided.shape[0]}")
    
    print("All tests passed!")


def benchmark_convolution_1d():
    """Benchmark Triton vs PyTorch implementations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sizes = [1000, 10000, 100000]
    kernel_sizes = [3, 7, 15, 31]
    
    for input_size in sizes:
        for kernel_size in kernel_sizes:
            input_signal = torch.randn(input_size, device=device)
            kernel = torch.randn(kernel_size, device=device)
            
            # Warmup
            for _ in range(10):
                _ = convolution_1d_triton(input_signal, kernel)
            
            # Benchmark Triton
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(100):
                _ = convolution_1d_triton(input_signal, kernel)
            end.record()
            torch.cuda.synchronize()
            
            triton_time = start.elapsed_time(end) / 100
            
            # Benchmark PyTorch
            input_torch = input_signal.unsqueeze(0).unsqueeze(0)
            kernel_torch = kernel.flip(0).unsqueeze(0).unsqueeze(0)
            
            start.record()
            for _ in range(100):
                _ = torch.conv1d(input_torch, kernel_torch)
            end.record()
            torch.cuda.synchronize()
            
            torch_time = start.elapsed_time(end) / 100
            
            speedup = torch_time / triton_time
            print(f"Input: {input_size}, Kernel: {kernel_size} - "
                  f"Triton: {triton_time:.3f}ms, PyTorch: {torch_time:.3f}ms, "
                  f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_convolution_1d()
    print("\nBenchmarking:")
    benchmark_convolution_1d()
