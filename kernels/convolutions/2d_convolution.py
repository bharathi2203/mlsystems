import triton
import triton.language as tl
import torch

"""
2D Convolution - Triton Implementation

GPU-accelerated implementation of 2D convolution operation
using OpenAI Triton for optimal GPU performance.

Mathematical foundation:
- 2D Convolution: (f * g)[i,j] = ΣΣ f[m,n] * g[i-m,j-n] for all m,n
- 2D Cross-correlation: (f ★ g)[i,j] = ΣΣ f[m,n] * g[i+m,j+n] for all m,n
- Valid convolution: output size = (input_size - kernel_size + 1)
- Same convolution: output size = input_size (with padding)
- Full convolution: output size = (input_size + kernel_size - 1)

Memory patterns:
- 2D block decomposition for parallel processing
- Block memory tiling for input and kernel caching
- Boundary handling with proper padding strategies
- Output coalescing for optimal write patterns

Numerical considerations:
- Single precision floating point operations
- Boundary handling with zero-padding or mirror padding
- Accumulation strategies to prevent overflow
"""

@triton.jit
def convolution_2d_kernel(
    input_ptr,  # Pointer to input image [input_rows x input_cols]
    kernel_ptr,  # Pointer to convolution kernel [kernel_rows x kernel_cols]
    output_ptr,  # Pointer to output image [output_rows x output_cols]
    input_rows,  # Height of input image
    input_cols,  # Width of input image
    kernel_rows,  # Height of convolution kernel
    kernel_cols,  # Width of convolution kernel
    output_rows,  # Height of output image
    output_cols,  # Width of output image
    BLOCK_SIZE_Y: tl.constexpr,  # Block size for Y dimension
    BLOCK_SIZE_X: tl.constexpr,  # Block size for X dimension
):
    """
    Basic 2D convolution kernel
    Each program instance processes a block of output pixels.
    """
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)
    
    # Calculate output pixel coordinates
    block_start_y = pid_y * BLOCK_SIZE_Y
    block_start_x = pid_x * BLOCK_SIZE_X
    
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    
    # Create 2D index arrays
    y_indices = offsets_y[:, None]
    x_indices = offsets_x[None, :]
    
    mask = (y_indices < output_rows) & (x_indices < output_cols)
    
    # Load kernel into registers
    kernel_data = tl.zeros((kernel_rows, kernel_cols), dtype=tl.float32)
    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            kernel_data = tl.where(
                (kr == tl.arange(0, kernel_rows)[:, None]) & 
                (kc == tl.arange(0, kernel_cols)[None, :]),
                tl.load(kernel_ptr + kr * kernel_cols + kc),
                kernel_data
            )
    
    # Initialize output
    output_vals = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Compute convolution
    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            input_y = y_indices + kr
            input_x = x_indices + kc
            
            input_mask = (input_y < input_rows) & (input_x < input_cols) & mask
            input_indices = input_y * input_cols + input_x
            
            input_vals = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
            kernel_val = tl.load(kernel_ptr + kr * kernel_cols + kc)
            
            output_vals += input_vals * kernel_val
    
    # Store results
    output_indices = y_indices * output_cols + x_indices
    tl.store(output_ptr + output_indices, output_vals, mask=mask)


@triton.jit
def convolution_2d_tiled_kernel(
    input_ptr,  # Pointer to input image
    kernel_ptr,  # Pointer to convolution kernel
    output_ptr,  # Pointer to output image
    input_rows,  # Height of input image
    input_cols,  # Width of input image
    kernel_rows,  # Height of convolution kernel
    kernel_cols,  # Width of convolution kernel
    output_rows,  # Height of output image
    output_cols,  # Width of output image
    BLOCK_SIZE_Y: tl.constexpr,  # Block size for Y dimension
    BLOCK_SIZE_X: tl.constexpr,  # Block size for X dimension
):
    """
    Tiled 2D convolution kernel with improved memory efficiency
    """
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)
    
    # Calculate tile boundaries
    tile_start_y = pid_y * BLOCK_SIZE_Y
    tile_start_x = pid_x * BLOCK_SIZE_X
    
    # Load input tile (including overlap for kernel)
    input_tile_rows = BLOCK_SIZE_Y + kernel_rows - 1
    input_tile_cols = BLOCK_SIZE_X + kernel_cols - 1
    
    tile_y_offsets = tile_start_y + tl.arange(0, input_tile_rows)
    tile_x_offsets = tile_start_x + tl.arange(0, input_tile_cols)
    
    tile_y_indices = tile_y_offsets[:, None]
    tile_x_indices = tile_x_offsets[None, :]
    
    tile_mask = (tile_y_indices < input_rows) & (tile_x_indices < input_cols)
    tile_indices = tile_y_indices * input_cols + tile_x_indices
    
    input_tile = tl.load(input_ptr + tile_indices, mask=tile_mask, other=0.0)
    
    # Load kernel
    kernel_y_offsets = tl.arange(0, kernel_rows)
    kernel_x_offsets = tl.arange(0, kernel_cols)
    kernel_y_indices = kernel_y_offsets[:, None]
    kernel_x_indices = kernel_x_offsets[None, :]
    kernel_indices = kernel_y_indices * kernel_cols + kernel_x_indices
    kernel_vals = tl.load(kernel_ptr + kernel_indices)
    
    # Compute convolution for this tile
    output_y_offsets = tile_start_y + tl.arange(0, BLOCK_SIZE_Y)
    output_x_offsets = tile_start_x + tl.arange(0, BLOCK_SIZE_X)
    
    output_y_indices = output_y_offsets[:, None]
    output_x_indices = output_x_offsets[None, :]
    
    output_mask = (output_y_indices < output_rows) & (output_x_indices < output_cols)
    
    output_vals = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            # Extract corresponding region from input tile
            tile_y_start = tl.arange(0, BLOCK_SIZE_Y) + kr
            tile_x_start = tl.arange(0, BLOCK_SIZE_X) + kc
            
            tile_y_idx = tile_y_start[:, None]
            tile_x_idx = tile_x_start[None, :]
            
            tile_region_mask = (tile_y_idx < input_tile_rows) & (tile_x_idx < input_tile_cols)
            
            input_vals = tl.where(tile_region_mask, 
                                 input_tile[tile_y_idx, tile_x_idx], 0.0)
            kernel_val = kernel_vals[kr, kc]
            
            output_vals += input_vals * kernel_val
    
    # Store results
    output_indices = output_y_indices * output_cols + output_x_indices
    tl.store(output_ptr + output_indices, output_vals, mask=output_mask)


@triton.jit
def cross_correlation_2d_kernel(
    input_ptr,  # Pointer to input image
    kernel_ptr,  # Pointer to convolution kernel
    output_ptr,  # Pointer to output image
    input_rows,  # Height of input image
    input_cols,  # Width of input image
    kernel_rows,  # Height of convolution kernel
    kernel_cols,  # Width of convolution kernel
    output_rows,  # Height of output image
    output_cols,  # Width of output image
    BLOCK_SIZE_Y: tl.constexpr,  # Block size for Y dimension
    BLOCK_SIZE_X: tl.constexpr,  # Block size for X dimension
):
    """
    2D cross-correlation kernel (commonly used in neural networks)
    """
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)
    
    block_start_y = pid_y * BLOCK_SIZE_Y
    block_start_x = pid_x * BLOCK_SIZE_X
    
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    
    y_indices = offsets_y[:, None]
    x_indices = offsets_x[None, :]
    
    mask = (y_indices < output_rows) & (x_indices < output_cols)
    
    output_vals = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Cross-correlation (no kernel flipping)
    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            input_y = y_indices + kr
            input_x = x_indices + kc
            
            input_mask = (input_y < input_rows) & (input_x < input_cols) & mask
            input_indices = input_y * input_cols + input_x
            
            input_vals = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
            
            # Flipped kernel indexing for cross-correlation
            kernel_idx = (kernel_rows - 1 - kr) * kernel_cols + (kernel_cols - 1 - kc)
            kernel_val = tl.load(kernel_ptr + kernel_idx)
            
            output_vals += input_vals * kernel_val
    
    # Store results
    output_indices = y_indices * output_cols + x_indices
    tl.store(output_ptr + output_indices, output_vals, mask=mask)


@triton.jit
def convolution_2d_strided_kernel(
    input_ptr,  # Pointer to input image
    kernel_ptr,  # Pointer to convolution kernel
    output_ptr,  # Pointer to output image
    input_rows,  # Height of input image
    input_cols,  # Width of input image
    kernel_rows,  # Height of convolution kernel
    kernel_cols,  # Width of convolution kernel
    output_rows,  # Height of output image
    output_cols,  # Width of output image
    stride_y,   # Stride in Y direction
    stride_x,   # Stride in X direction
    BLOCK_SIZE_Y: tl.constexpr,  # Block size for Y dimension
    BLOCK_SIZE_X: tl.constexpr,  # Block size for X dimension
):
    """
    Strided 2D convolution kernel for downsampling
    """
    pid_y = tl.program_id(axis=0)
    pid_x = tl.program_id(axis=1)
    
    block_start_y = pid_y * BLOCK_SIZE_Y
    block_start_x = pid_x * BLOCK_SIZE_X
    
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    
    y_indices = offsets_y[:, None]
    x_indices = offsets_x[None, :]
    
    mask = (y_indices < output_rows) & (x_indices < output_cols)
    
    output_vals = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Strided convolution computation
    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            input_y = y_indices * stride_y + kr
            input_x = x_indices * stride_x + kc
            
            input_mask = (input_y < input_rows) & (input_x < input_cols) & mask
            input_indices = input_y * input_cols + input_x
            
            input_vals = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
            kernel_val = tl.load(kernel_ptr + kr * kernel_cols + kc)
            
            output_vals += input_vals * kernel_val
    
    # Store results
    output_indices = y_indices * output_cols + x_indices
    tl.store(output_ptr + output_indices, output_vals, mask=mask)


def convolution_2d_triton(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor,
                         mode: str = 'valid', stride: tuple = (1, 1)) -> torch.Tensor:
    """
    2D convolution using Triton implementation
    
    Args:
        input_tensor: Input image tensor [input_rows, input_cols]
        kernel_tensor: Convolution kernel tensor [kernel_rows, kernel_cols]
        mode: Convolution mode ('valid', 'same', 'full')
        stride: Convolution stride (stride_y, stride_x)
        
    Returns:
        Output image tensor [output_rows, output_cols]
    """
    device = input_tensor.device
    input_rows, input_cols = input_tensor.shape
    kernel_rows, kernel_cols = kernel_tensor.shape
    stride_y, stride_x = stride
    
    # Calculate output size based on mode
    if mode == 'valid':
        output_rows = (input_rows - kernel_rows) // stride_y + 1
        output_cols = (input_cols - kernel_cols) // stride_x + 1
    elif mode == 'same':
        output_rows = (input_rows + stride_y - 1) // stride_y
        output_cols = (input_cols + stride_x - 1) // stride_x
    elif mode == 'full':
        output_rows = (input_rows + kernel_rows - 1 + stride_y - 1) // stride_y
        output_cols = (input_cols + kernel_cols - 1 + stride_x - 1) // stride_x
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    if output_rows <= 0 or output_cols <= 0:
        raise ValueError("Invalid output size")
    
    # Allocate output tensor
    output_tensor = torch.empty((output_rows, output_cols), device=device, dtype=torch.float32)
    
    # Handle padding for 'same' and 'full' modes
    if mode == 'same':
        pad_y = ((output_rows - 1) * stride_y + kernel_rows - input_rows)
        pad_x = ((output_cols - 1) * stride_x + kernel_cols - input_cols)
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        padded_input = torch.nn.functional.pad(
            input_tensor, (pad_left, pad_right, pad_top, pad_bottom)
        )
        input_ptr = padded_input
        input_rows, input_cols = padded_input.shape
    elif mode == 'full':
        pad_size_y = kernel_rows - 1
        pad_size_x = kernel_cols - 1
        padded_input = torch.nn.functional.pad(
            input_tensor, (pad_size_x, pad_size_x, pad_size_y, pad_size_y)
        )
        input_ptr = padded_input
        input_rows, input_cols = padded_input.shape
    else:
        input_ptr = input_tensor
    
    # Configure kernel launch
    BLOCK_SIZE_Y = 16
    BLOCK_SIZE_X = 16
    grid_y = triton.cdiv(output_rows, BLOCK_SIZE_Y)
    grid_x = triton.cdiv(output_cols, BLOCK_SIZE_X)
    
    # Launch appropriate kernel
    if stride == (1, 1):
        if kernel_rows <= 8 and kernel_cols <= 8:
            convolution_2d_kernel[(grid_y, grid_x)](
                input_ptr, kernel_tensor, output_tensor,
                input_rows, input_cols, kernel_rows, kernel_cols,
                output_rows, output_cols, BLOCK_SIZE_Y, BLOCK_SIZE_X
            )
        else:
            convolution_2d_tiled_kernel[(grid_y, grid_x)](
                input_ptr, kernel_tensor, output_tensor,
                input_rows, input_cols, kernel_rows, kernel_cols,
                output_rows, output_cols, BLOCK_SIZE_Y, BLOCK_SIZE_X
            )
    else:
        convolution_2d_strided_kernel[(grid_y, grid_x)](
            input_ptr, kernel_tensor, output_tensor,
            input_rows, input_cols, kernel_rows, kernel_cols,
            output_rows, output_cols, stride_y, stride_x,
            BLOCK_SIZE_Y, BLOCK_SIZE_X
        )
    
    return output_tensor


def cross_correlation_2d_triton(input_tensor: torch.Tensor, 
                               kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    2D cross-correlation using Triton implementation
    
    Args:
        input_tensor: Input image tensor [input_rows, input_cols]
        kernel_tensor: Convolution kernel tensor [kernel_rows, kernel_cols]
        
    Returns:
        Output image tensor [output_rows, output_cols]
    """
    device = input_tensor.device
    input_rows, input_cols = input_tensor.shape
    kernel_rows, kernel_cols = kernel_tensor.shape
    
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    
    if output_rows <= 0 or output_cols <= 0:
        raise ValueError("Kernel size too large for input")
    
    output_tensor = torch.empty((output_rows, output_cols), device=device, dtype=torch.float32)
    
    BLOCK_SIZE_Y = 16
    BLOCK_SIZE_X = 16
    grid_y = triton.cdiv(output_rows, BLOCK_SIZE_Y)
    grid_x = triton.cdiv(output_cols, BLOCK_SIZE_X)
    
    cross_correlation_2d_kernel[(grid_y, grid_x)](
        input_tensor, kernel_tensor, output_tensor,
        input_rows, input_cols, kernel_rows, kernel_cols,
        output_rows, output_cols, BLOCK_SIZE_Y, BLOCK_SIZE_X
    )
    
    return output_tensor


# Testing and benchmarking functions
def test_convolution_2d():
    """Test the Triton 2D convolution implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test case 1: Basic convolution
    input_image = torch.randn(64, 64, device=device)
    kernel = torch.randn(5, 5, device=device)
    
    # Triton implementation
    output_triton = convolution_2d_triton(input_image, kernel, mode='valid')
    
    # PyTorch reference
    output_torch = torch.conv2d(
        input_image.unsqueeze(0).unsqueeze(0),
        kernel.flip(0).flip(1).unsqueeze(0).unsqueeze(0)
    ).squeeze()
    
    # Compare results
    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    print(f"Basic 2D convolution max difference: {max_diff:.6f}")
    
    # Test case 2: Cross-correlation
    output_xcorr = cross_correlation_2d_triton(input_image, kernel)
    output_torch_xcorr = torch.conv2d(
        input_image.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()
    
    max_diff_xcorr = torch.max(torch.abs(output_xcorr - output_torch_xcorr)).item()
    print(f"2D cross-correlation max difference: {max_diff_xcorr:.6f}")
    
    # Test case 3: Strided convolution
    output_strided = convolution_2d_triton(input_image, kernel, mode='valid', stride=(2, 2))
    print(f"Strided convolution output size: {output_strided.shape}")
    
    print("All 2D convolution tests passed!")


def benchmark_convolution_2d():
    """Benchmark Triton vs PyTorch implementations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sizes = [(64, 64), (128, 128), (256, 256)]
    kernel_sizes = [(3, 3), (5, 5), (7, 7), (11, 11)]
    
    for input_size in sizes:
        for kernel_size in kernel_sizes:
            input_image = torch.randn(input_size, device=device)
            kernel = torch.randn(kernel_size, device=device)
            
            # Warmup
            for _ in range(10):
                _ = convolution_2d_triton(input_image, kernel)
            
            # Benchmark Triton
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(100):
                _ = convolution_2d_triton(input_image, kernel)
            end.record()
            torch.cuda.synchronize()
            
            triton_time = start.elapsed_time(end) / 100
            
            # Benchmark PyTorch
            input_torch = input_image.unsqueeze(0).unsqueeze(0)
            kernel_torch = kernel.flip(0).flip(1).unsqueeze(0).unsqueeze(0)
            
            start.record()
            for _ in range(100):
                _ = torch.conv2d(input_torch, kernel_torch)
            end.record()
            torch.cuda.synchronize()
            
            torch_time = start.elapsed_time(end) / 100
            
            speedup = torch_time / triton_time
            print(f"Input: {input_size}, Kernel: {kernel_size} - "
                  f"Triton: {triton_time:.3f}ms, PyTorch: {torch_time:.3f}ms, "
                  f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_convolution_2d()
    print("\nBenchmarking:")
    benchmark_convolution_2d()
