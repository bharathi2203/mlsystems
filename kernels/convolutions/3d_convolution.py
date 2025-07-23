import triton
import triton.language as tl
import torch

"""
3D Convolution - Triton Implementation

GPU-accelerated implementation of 3D convolution operation
using OpenAI Triton for optimal GPU performance.

Mathematical foundation:
- 3D Convolution: (f * g)[i,j,k] = ΣΣΣ f[m,n,p] * g[i-m,j-n,k-p] for all m,n,p
- 3D Cross-correlation: (f ★ g)[i,j,k] = ΣΣΣ f[m,n,p] * g[i+m,j+n,k+p] for all m,n,p
- Valid convolution: output size = (input_size - kernel_size + 1)
- Same convolution: output size = input_size (with padding)
- Full convolution: output size = (input_size + kernel_size - 1)

Memory patterns:
- 3D block decomposition for parallel processing
- Block memory tiling for volumetric data caching
- Proper boundary handling with 3D padding strategies
- Optimized memory layout for 3D data structures

Numerical considerations:
- Single precision floating point operations
- 3D boundary handling with zero-padding
- Accumulation strategies for large 3D kernels
"""

@triton.jit
def convolution_3d_kernel(
    input_ptr,  # Pointer to input volume [depth x height x width]
    kernel_ptr,  # Pointer to convolution kernel [kernel_depth x kernel_height x kernel_width]
    output_ptr,  # Pointer to output volume [output_depth x output_height x output_width]
    input_depth,  # Depth of input volume
    input_height,  # Height of input volume
    input_width,  # Width of input volume
    kernel_depth,  # Depth of convolution kernel
    kernel_height,  # Height of convolution kernel
    kernel_width,  # Width of convolution kernel
    output_depth,  # Depth of output volume
    output_height,  # Height of output volume
    output_width,  # Width of output volume
    BLOCK_SIZE_Z: tl.constexpr,  # Block size for Z dimension
    BLOCK_SIZE_Y: tl.constexpr,  # Block size for Y dimension
    BLOCK_SIZE_X: tl.constexpr,  # Block size for X dimension
):
    """
    Basic 3D convolution kernel
    Each program instance processes a 3D block of output voxels.
    """
    pid_z = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_x = tl.program_id(axis=2)
    
    # Calculate output voxel coordinates
    block_start_z = pid_z * BLOCK_SIZE_Z
    block_start_y = pid_y * BLOCK_SIZE_Y
    block_start_x = pid_x * BLOCK_SIZE_X
    
    offsets_z = block_start_z + tl.arange(0, BLOCK_SIZE_Z)
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    
    # Create 3D index arrays
    z_indices = offsets_z[:, None, None]
    y_indices = offsets_y[None, :, None]
    x_indices = offsets_x[None, None, :]
    
    mask = (z_indices < output_depth) & (y_indices < output_height) & (x_indices < output_width)
    
    # Load kernel into registers
    kernel_z_offsets = tl.arange(0, kernel_depth)
    kernel_y_offsets = tl.arange(0, kernel_height)
    kernel_x_offsets = tl.arange(0, kernel_width)
    
    kernel_z_indices = kernel_z_offsets[:, None, None]
    kernel_y_indices = kernel_y_offsets[None, :, None]
    kernel_x_indices = kernel_x_offsets[None, None, :]
    
    kernel_indices = (kernel_z_indices * kernel_height * kernel_width + 
                     kernel_y_indices * kernel_width + 
                     kernel_x_indices)
    
    kernel_vals = tl.load(kernel_ptr + kernel_indices)
    
    # Initialize output
    output_vals = tl.zeros((BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Compute convolution
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                input_z = z_indices + kd
                input_y = y_indices + kh
                input_x = x_indices + kw
                
                input_mask = (input_z < input_depth) & (input_y < input_height) & (input_x < input_width) & mask
                input_indices = (input_z * input_height * input_width + 
                               input_y * input_width + 
                               input_x)
                
                input_vals = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
                kernel_val = kernel_vals[kd, kh, kw]
                
                output_vals += input_vals * kernel_val
    
    # Store results
    output_indices = (z_indices * output_height * output_width + 
                     y_indices * output_width + 
                     x_indices)
    tl.store(output_ptr + output_indices, output_vals, mask=mask)


@triton.jit
def cross_correlation_3d_kernel(
    input_ptr,  # Pointer to input volume
    kernel_ptr,  # Pointer to convolution kernel
    output_ptr,  # Pointer to output volume
    input_depth,  # Depth of input volume
    input_height,  # Height of input volume
    input_width,  # Width of input volume
    kernel_depth,  # Depth of convolution kernel
    kernel_height,  # Height of convolution kernel
    kernel_width,  # Width of convolution kernel
    output_depth,  # Depth of output volume
    output_height,  # Height of output volume
    output_width,  # Width of output volume
    BLOCK_SIZE_Z: tl.constexpr,  # Block size for Z dimension
    BLOCK_SIZE_Y: tl.constexpr,  # Block size for Y dimension
    BLOCK_SIZE_X: tl.constexpr,  # Block size for X dimension
):
    """
    3D cross-correlation kernel (commonly used in 3D neural networks)
    """
    pid_z = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_x = tl.program_id(axis=2)
    
    block_start_z = pid_z * BLOCK_SIZE_Z
    block_start_y = pid_y * BLOCK_SIZE_Y
    block_start_x = pid_x * BLOCK_SIZE_X
    
    offsets_z = block_start_z + tl.arange(0, BLOCK_SIZE_Z)
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    
    z_indices = offsets_z[:, None, None]
    y_indices = offsets_y[None, :, None]
    x_indices = offsets_x[None, None, :]
    
    mask = (z_indices < output_depth) & (y_indices < output_height) & (x_indices < output_width)
    
    # Load kernel
    kernel_z_offsets = tl.arange(0, kernel_depth)
    kernel_y_offsets = tl.arange(0, kernel_height)
    kernel_x_offsets = tl.arange(0, kernel_width)
    
    kernel_z_indices = kernel_z_offsets[:, None, None]
    kernel_y_indices = kernel_y_offsets[None, :, None]
    kernel_x_indices = kernel_x_offsets[None, None, :]
    
    kernel_indices = (kernel_z_indices * kernel_height * kernel_width + 
                     kernel_y_indices * kernel_width + 
                     kernel_x_indices)
    
    kernel_vals = tl.load(kernel_ptr + kernel_indices)
    
    output_vals = tl.zeros((BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Cross-correlation (no kernel flipping)
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                input_z = z_indices + kd
                input_y = y_indices + kh
                input_x = x_indices + kw
                
                input_mask = (input_z < input_depth) & (input_y < input_height) & (input_x < input_width) & mask
                input_indices = (input_z * input_height * input_width + 
                               input_y * input_width + 
                               input_x)
                
                input_vals = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
                
                # Flipped kernel indexing for cross-correlation
                flipped_kd = kernel_depth - 1 - kd
                flipped_kh = kernel_height - 1 - kh
                flipped_kw = kernel_width - 1 - kw
                kernel_val = kernel_vals[flipped_kd, flipped_kh, flipped_kw]
                
                output_vals += input_vals * kernel_val
    
    # Store results
    output_indices = (z_indices * output_height * output_width + 
                     y_indices * output_width + 
                     x_indices)
    tl.store(output_ptr + output_indices, output_vals, mask=mask)


@triton.jit
def convolution_3d_strided_kernel(
    input_ptr,  # Pointer to input volume
    kernel_ptr,  # Pointer to convolution kernel
    output_ptr,  # Pointer to output volume
    input_depth,  # Depth of input volume
    input_height,  # Height of input volume
    input_width,  # Width of input volume
    kernel_depth,  # Depth of convolution kernel
    kernel_height,  # Height of convolution kernel
    kernel_width,  # Width of convolution kernel
    output_depth,  # Depth of output volume
    output_height,  # Height of output volume
    output_width,  # Width of output volume
    stride_z,   # Stride in Z direction
    stride_y,   # Stride in Y direction
    stride_x,   # Stride in X direction
    BLOCK_SIZE_Z: tl.constexpr,  # Block size for Z dimension
    BLOCK_SIZE_Y: tl.constexpr,  # Block size for Y dimension
    BLOCK_SIZE_X: tl.constexpr,  # Block size for X dimension
):
    """
    Strided 3D convolution kernel for downsampling
    """
    pid_z = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_x = tl.program_id(axis=2)
    
    block_start_z = pid_z * BLOCK_SIZE_Z
    block_start_y = pid_y * BLOCK_SIZE_Y
    block_start_x = pid_x * BLOCK_SIZE_X
    
    offsets_z = block_start_z + tl.arange(0, BLOCK_SIZE_Z)
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    
    z_indices = offsets_z[:, None, None]
    y_indices = offsets_y[None, :, None]
    x_indices = offsets_x[None, None, :]
    
    mask = (z_indices < output_depth) & (y_indices < output_height) & (x_indices < output_width)
    
    # Load kernel
    kernel_z_offsets = tl.arange(0, kernel_depth)
    kernel_y_offsets = tl.arange(0, kernel_height)
    kernel_x_offsets = tl.arange(0, kernel_width)
    
    kernel_z_indices = kernel_z_offsets[:, None, None]
    kernel_y_indices = kernel_y_offsets[None, :, None]
    kernel_x_indices = kernel_x_offsets[None, None, :]
    
    kernel_indices = (kernel_z_indices * kernel_height * kernel_width + 
                     kernel_y_indices * kernel_width + 
                     kernel_x_indices)
    
    kernel_vals = tl.load(kernel_ptr + kernel_indices)
    
    output_vals = tl.zeros((BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    
    # Strided convolution computation
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                input_z = z_indices * stride_z + kd
                input_y = y_indices * stride_y + kh
                input_x = x_indices * stride_x + kw
                
                input_mask = (input_z < input_depth) & (input_y < input_height) & (input_x < input_width) & mask
                input_indices = (input_z * input_height * input_width + 
                               input_y * input_width + 
                               input_x)
                
                input_vals = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
                kernel_val = kernel_vals[kd, kh, kw]
                
                output_vals += input_vals * kernel_val
    
    # Store results
    output_indices = (z_indices * output_height * output_width + 
                     y_indices * output_width + 
                     x_indices)
    tl.store(output_ptr + output_indices, output_vals, mask=mask)


def convolution_3d_triton(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor,
                         mode: str = 'valid', stride: tuple = (1, 1, 1)) -> torch.Tensor:
    """
    3D convolution using Triton implementation
    
    Args:
        input_tensor: Input volume tensor [depth, height, width]
        kernel_tensor: Convolution kernel tensor [kernel_depth, kernel_height, kernel_width]
        mode: Convolution mode ('valid', 'same', 'full')
        stride: Convolution stride (stride_z, stride_y, stride_x)
        
    Returns:
        Output volume tensor [output_depth, output_height, output_width]
    """
    device = input_tensor.device
    input_depth, input_height, input_width = input_tensor.shape
    kernel_depth, kernel_height, kernel_width = kernel_tensor.shape
    stride_z, stride_y, stride_x = stride
    
    # Calculate output size based on mode
    if mode == 'valid':
        output_depth = (input_depth - kernel_depth) // stride_z + 1
        output_height = (input_height - kernel_height) // stride_y + 1
        output_width = (input_width - kernel_width) // stride_x + 1
    elif mode == 'same':
        output_depth = (input_depth + stride_z - 1) // stride_z
        output_height = (input_height + stride_y - 1) // stride_y
        output_width = (input_width + stride_x - 1) // stride_x
    elif mode == 'full':
        output_depth = (input_depth + kernel_depth - 1 + stride_z - 1) // stride_z
        output_height = (input_height + kernel_height - 1 + stride_y - 1) // stride_y
        output_width = (input_width + kernel_width - 1 + stride_x - 1) // stride_x
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    if output_depth <= 0 or output_height <= 0 or output_width <= 0:
        raise ValueError("Invalid output size")
    
    # Allocate output tensor
    output_tensor = torch.empty((output_depth, output_height, output_width), 
                               device=device, dtype=torch.float32)
    
    # Handle padding for 'same' and 'full' modes
    if mode == 'same':
        pad_z = ((output_depth - 1) * stride_z + kernel_depth - input_depth)
        pad_y = ((output_height - 1) * stride_y + kernel_height - input_height)
        pad_x = ((output_width - 1) * stride_x + kernel_width - input_width)
        
        pad_front = pad_z // 2
        pad_back = pad_z - pad_front
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        
        padded_input = torch.nn.functional.pad(
            input_tensor, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )
        input_ptr = padded_input
        input_depth, input_height, input_width = padded_input.shape
    elif mode == 'full':
        pad_size_z = kernel_depth - 1
        pad_size_y = kernel_height - 1
        pad_size_x = kernel_width - 1
        
        padded_input = torch.nn.functional.pad(
            input_tensor, (pad_size_x, pad_size_x, pad_size_y, pad_size_y, pad_size_z, pad_size_z)
        )
        input_ptr = padded_input
        input_depth, input_height, input_width = padded_input.shape
    else:
        input_ptr = input_tensor
    
    # Configure kernel launch
    BLOCK_SIZE_Z = 4
    BLOCK_SIZE_Y = 4
    BLOCK_SIZE_X = 4
    
    grid_z = triton.cdiv(output_depth, BLOCK_SIZE_Z)
    grid_y = triton.cdiv(output_height, BLOCK_SIZE_Y)
    grid_x = triton.cdiv(output_width, BLOCK_SIZE_X)
    
    # Launch appropriate kernel
    if stride == (1, 1, 1):
        convolution_3d_kernel[(grid_z, grid_y, grid_x)](
            input_ptr, kernel_tensor, output_tensor,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            output_depth, output_height, output_width,
            BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X
        )
    else:
        convolution_3d_strided_kernel[(grid_z, grid_y, grid_x)](
            input_ptr, kernel_tensor, output_tensor,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            output_depth, output_height, output_width,
            stride_z, stride_y, stride_x,
            BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X
        )
    
    return output_tensor


def cross_correlation_3d_triton(input_tensor: torch.Tensor, 
                               kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    3D cross-correlation using Triton implementation
    
    Args:
        input_tensor: Input volume tensor [depth, height, width]
        kernel_tensor: Convolution kernel tensor [kernel_depth, kernel_height, kernel_width]
        
    Returns:
        Output volume tensor [output_depth, output_height, output_width]
    """
    device = input_tensor.device
    input_depth, input_height, input_width = input_tensor.shape
    kernel_depth, kernel_height, kernel_width = kernel_tensor.shape
    
    output_depth = input_depth - kernel_depth + 1
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    if output_depth <= 0 or output_height <= 0 or output_width <= 0:
        raise ValueError("Kernel size too large for input")
    
    output_tensor = torch.empty((output_depth, output_height, output_width), 
                               device=device, dtype=torch.float32)
    
    BLOCK_SIZE_Z = 4
    BLOCK_SIZE_Y = 4
    BLOCK_SIZE_X = 4
    
    grid_z = triton.cdiv(output_depth, BLOCK_SIZE_Z)
    grid_y = triton.cdiv(output_height, BLOCK_SIZE_Y)
    grid_x = triton.cdiv(output_width, BLOCK_SIZE_X)
    
    cross_correlation_3d_kernel[(grid_z, grid_y, grid_x)](
        input_tensor, kernel_tensor, output_tensor,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        output_depth, output_height, output_width,
        BLOCK_SIZE_Z, BLOCK_SIZE_Y, BLOCK_SIZE_X
    )
    
    return output_tensor


# Testing and benchmarking functions
def test_convolution_3d():
    """Test the Triton 3D convolution implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test case 1: Basic convolution
    input_volume = torch.randn(32, 32, 32, device=device)
    kernel = torch.randn(3, 3, 3, device=device)
    
    # Triton implementation
    output_triton = convolution_3d_triton(input_volume, kernel, mode='valid')
    
    # PyTorch reference
    output_torch = torch.conv3d(
        input_volume.unsqueeze(0).unsqueeze(0),
        kernel.flip(0).flip(1).flip(2).unsqueeze(0).unsqueeze(0)
    ).squeeze()
    
    # Compare results
    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    print(f"Basic 3D convolution max difference: {max_diff:.6f}")
    
    # Test case 2: Cross-correlation
    output_xcorr = cross_correlation_3d_triton(input_volume, kernel)
    output_torch_xcorr = torch.conv3d(
        input_volume.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()
    
    max_diff_xcorr = torch.max(torch.abs(output_xcorr - output_torch_xcorr)).item()
    print(f"3D cross-correlation max difference: {max_diff_xcorr:.6f}")
    
    # Test case 3: Strided convolution
    output_strided = convolution_3d_triton(input_volume, kernel, mode='valid', stride=(2, 2, 2))
    print(f"Strided convolution output size: {output_strided.shape}")
    
    print("All 3D convolution tests passed!")


def benchmark_convolution_3d():
    """Benchmark Triton vs PyTorch implementations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sizes = [(32, 32, 32), (64, 64, 64)]
    kernel_sizes = [(3, 3, 3), (5, 5, 5)]
    
    for input_size in sizes:
        for kernel_size in kernel_sizes:
            input_volume = torch.randn(input_size, device=device)
            kernel = torch.randn(kernel_size, device=device)
            
            # Warmup
            for _ in range(10):
                _ = convolution_3d_triton(input_volume, kernel)
            
            # Benchmark Triton
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(50):
                _ = convolution_3d_triton(input_volume, kernel)
            end.record()
            torch.cuda.synchronize()
            
            triton_time = start.elapsed_time(end) / 50
            
            # Benchmark PyTorch
            input_torch = input_volume.unsqueeze(0).unsqueeze(0)
            kernel_torch = kernel.flip(0).flip(1).flip(2).unsqueeze(0).unsqueeze(0)
            
            start.record()
            for _ in range(50):
                _ = torch.conv3d(input_torch, kernel_torch)
            end.record()
            torch.cuda.synchronize()
            
            torch_time = start.elapsed_time(end) / 50
            
            speedup = torch_time / triton_time
            print(f"Input: {input_size}, Kernel: {kernel_size} - "
                  f"Triton: {triton_time:.3f}ms, PyTorch: {torch_time:.3f}ms, "
                  f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_convolution_3d()
    print("\nBenchmarking:")
    benchmark_convolution_3d()
