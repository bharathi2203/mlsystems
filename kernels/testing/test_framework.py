#!/usr/bin/env python3
"""
Comprehensive testing framework for ML Systems kernels

Supports testing across CUDA, Metal, and Triton platforms with:
- Correctness validation against reference implementations
- Performance benchmarking
- Cross-platform consistency checks
- Automated test discovery
"""

import os
import time
import numpy as np
import torch
import pytest
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
import json

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    import metal
    METAL_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    METAL_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()

@dataclass
class KernelTestResult:
    """Results from kernel testing"""
    kernel_name: str
    platform: str
    passed: bool
    execution_time_ms: float
    throughput_gb_s: Optional[float] = None
    error_message: Optional[str] = None
    max_error: Optional[float] = None
    memory_usage_mb: Optional[float] = None

@dataclass
class TestConfig:
    """Configuration for kernel testing"""
    test_sizes: List[Tuple[int, ...]] = None
    data_types: List[torch.dtype] = None
    tolerance: float = 1e-5
    num_warmup: int = 5
    num_iterations: int = 20
    validate_correctness: bool = True
    benchmark_performance: bool = True
    
    def __post_init__(self):
        if self.test_sizes is None:
            self.test_sizes = [
                (1024,), (4096,), (16384,),  # 1D sizes
                (32, 32), (128, 128), (512, 512),  # 2D sizes
                (64, 64, 64), (32, 128, 256),  # 3D sizes
            ]
        if self.data_types is None:
            self.data_types = [torch.float32, torch.float16]

class KernelTester:
    """Base class for testing different kernel platforms"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results: List[KernelTestResult] = []
        
    def generate_test_data(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                          device: str) -> torch.Tensor:
        """Generate test data for given shape and type"""
        if dtype == torch.int8:
            return torch.randint(-128, 127, shape, dtype=dtype, device=device)
        elif dtype == torch.bool:
            return torch.rand(shape, device=device) > 0.5
        else:
            return torch.randn(shape, dtype=dtype, device=device)
    
    def validate_correctness(self, output: torch.Tensor, reference: torch.Tensor,
                           tolerance: float = None) -> Tuple[bool, float]:
        """Validate kernel output against reference implementation"""
        tolerance = tolerance or self.config.tolerance
        
        if output.shape != reference.shape:
            return False, float('inf')
        
        # Handle different data types
        if output.dtype != reference.dtype:
            output = output.to(reference.dtype)
        
        # Compute error metrics
        abs_error = torch.abs(output - reference)
        max_error = torch.max(abs_error).item()
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, float('inf')
        
        # Different validation strategies based on magnitude
        if torch.max(torch.abs(reference)) < 1e-6:  # Near-zero reference
            passed = max_error < tolerance
        else:  # Relative error for larger values
            rel_error = abs_error / (torch.abs(reference) + 1e-8)
            max_rel_error = torch.max(rel_error).item()
            passed = max_rel_error < tolerance and max_error < 1.0
        
        return passed, max_error
    
    def benchmark_kernel(self, kernel_func: Callable, *args, **kwargs) -> float:
        """Benchmark kernel execution time"""
        device = args[0].device if args else 'cpu'
        
        # Warmup
        for _ in range(self.config.num_warmup):
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            result = kernel_func(*args, **kwargs)
            if 'cuda' in str(device):
                torch.cuda.synchronize()
        
        # Benchmark
        if 'cuda' in str(device):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()
        
        for _ in range(self.config.num_iterations):
            result = kernel_func(*args, **kwargs)
        
        if 'cuda' in str(device):
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event) / self.config.num_iterations
        else:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000 / self.config.num_iterations
        
        return elapsed_ms
    
    def calculate_throughput(self, data_size_bytes: int, time_ms: float) -> float:
        """Calculate memory throughput in GB/s"""
        return (data_size_bytes / 1e9) / (time_ms / 1000)

class TritonKernelTester(KernelTester):
    """Tester for Triton kernels"""
    
    def __init__(self, config: TestConfig = None):
        super().__init__(config)
        self.platform = "triton"
        
    def test_kernel(self, kernel_module, kernel_name: str, 
                   reference_func: Callable) -> List[KernelTestResult]:
        """Test a Triton kernel"""
        if not TRITON_AVAILABLE:
            return [KernelTestResult(
                kernel_name=kernel_name,
                platform=self.platform,
                passed=False,
                execution_time_ms=0.0,
                error_message="Triton not available"
            )]
        
        results = []
        
        for shape in self.config.test_sizes:
            for dtype in self.config.data_types:
                try:
                    # Generate test data
                    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
                    input_data = self.generate_test_data(shape, dtype, device)
                    
                    # Get reference result
                    reference_output = reference_func(input_data.cpu())
                    
                    # Test Triton kernel
                    if hasattr(kernel_module, f'{kernel_name}_triton'):
                        triton_func = getattr(kernel_module, f'{kernel_name}_triton')
                        
                        # Benchmark performance
                        if self.config.benchmark_performance:
                            exec_time = self.benchmark_kernel(triton_func, input_data)
                        else:
                            exec_time = 0.0
                        
                        # Validate correctness
                        triton_output = triton_func(input_data)
                        passed, max_error = self.validate_correctness(
                            triton_output.cpu(), reference_output
                        )
                        
                        # Calculate throughput
                        data_size = input_data.numel() * input_data.element_size()
                        throughput = self.calculate_throughput(data_size * 2, exec_time)  # Read + Write
                        
                        results.append(KernelTestResult(
                            kernel_name=f"{kernel_name}_{shape}_{dtype}",
                            platform=self.platform,
                            passed=passed,
                            execution_time_ms=exec_time,
                            throughput_gb_s=throughput,
                            max_error=max_error
                        ))
                    
                except Exception as e:
                    results.append(KernelTestResult(
                        kernel_name=f"{kernel_name}_{shape}_{dtype}",
                        platform=self.platform,
                        passed=False,
                        execution_time_ms=0.0,
                        error_message=str(e)
                    ))
        
        return results

class CUDAKernelTester(KernelTester):
    """Tester for CUDA kernels"""
    
    def __init__(self, config: TestConfig = None):
        super().__init__(config)
        self.platform = "cuda"
    
    def test_kernel(self, kernel_path: str, kernel_name: str,
                   reference_func: Callable) -> List[KernelTestResult]:
        """Test a CUDA kernel"""
        if not CUDA_AVAILABLE:
            return [KernelTestResult(
                kernel_name=kernel_name,
                platform=self.platform,
                passed=False,
                execution_time_ms=0.0,
                error_message="CUDA not available"
            )]
        
        # TODO: Implement CUDA kernel compilation and testing
        # This would require compiling .cu files and calling kernels
        # For now, return placeholder
        return [KernelTestResult(
            kernel_name=kernel_name,
            platform=self.platform,
            passed=False,
            execution_time_ms=0.0,
            error_message="CUDA testing not yet implemented"
        )]

class MetalKernelTester(KernelTester):
    """Tester for Metal kernels"""
    
    def __init__(self, config: TestConfig = None):
        super().__init__(config)
        self.platform = "metal"
    
    def test_kernel(self, kernel_path: str, kernel_name: str,
                   reference_func: Callable) -> List[KernelTestResult]:
        """Test a Metal kernel"""
        if not METAL_AVAILABLE:
            return [KernelTestResult(
                kernel_name=kernel_name,
                platform=self.platform,
                passed=False,
                execution_time_ms=0.0,
                error_message="Metal not available"
            )]
        
        # TODO: Implement Metal kernel compilation and testing
        # This would require compiling .metal files and calling kernels
        # For now, return placeholder
        return [KernelTestResult(
            kernel_name=kernel_name,
            platform=self.platform,
            passed=False,
            execution_time_ms=0.0,
            error_message="Metal testing not yet implemented"
        )]

def discover_kernels() -> Dict[str, Dict[str, str]]:
    """Discover all kernels in the repository"""
    kernels = {}
    # Navigate up one level from testing directory to find kernels
    kernels_dir = Path(__file__).parent.parent
    
    for cu_file in kernels_dir.rglob("*.cu"):
        kernel_name = cu_file.stem
        kernel_dir = cu_file.parent
        
        # Find corresponding files
        metal_file = kernel_dir / f"{kernel_name}.metal"
        triton_file = kernel_dir / f"{kernel_name}.py"
        
        kernels[kernel_name] = {
            'cuda': str(cu_file) if cu_file.exists() else None,
            'metal': str(metal_file) if metal_file.exists() else None,
            'triton': str(triton_file) if triton_file.exists() else None,
            'category': kernel_dir.relative_to(kernels_dir).parts[0] if len(kernel_dir.relative_to(kernels_dir).parts) > 0 else 'misc'
        }
    
    return kernels

def run_all_tests(config: TestConfig = None) -> Dict[str, List[KernelTestResult]]:
    """Run tests for all discovered kernels"""
    config = config or TestConfig()
    
    # Discover kernels
    kernels = discover_kernels()
    
    # Initialize testers
    triton_tester = TritonKernelTester(config)
    cuda_tester = CUDAKernelTester(config)
    metal_tester = MetalKernelTester(config)
    
    all_results = {}
    
    for kernel_name, kernel_files in kernels.items():
        print(f"Testing kernel: {kernel_name}")
        
        # Load reference implementation (you'll need to implement these)
        reference_func = get_reference_implementation(kernel_name)
        
        if reference_func is None:
            print(f"  No reference implementation for {kernel_name}, skipping...")
            continue
        
        kernel_results = []
        
        # Test Triton version
        if kernel_files['triton']:
            try:
                # Dynamically import Triton module
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    f"{kernel_name}_triton", kernel_files['triton']
                )
                triton_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(triton_module)
                
                triton_results = triton_tester.test_kernel(
                    triton_module, kernel_name, reference_func
                )
                kernel_results.extend(triton_results)
            except Exception as e:
                print(f"  Failed to test Triton version: {e}")
        
        # Test CUDA version
        if kernel_files['cuda']:
            cuda_results = cuda_tester.test_kernel(
                kernel_files['cuda'], kernel_name, reference_func
            )
            kernel_results.extend(cuda_results)
        
        # Test Metal version
        if kernel_files['metal']:
            metal_results = metal_tester.test_kernel(
                kernel_files['metal'], kernel_name, reference_func
            )
            kernel_results.extend(metal_results)
        
        all_results[kernel_name] = kernel_results
    
    return all_results

def get_reference_implementation(kernel_name: str) -> Optional[Callable]:
    """Get reference implementation for a kernel"""
    # This function would return numpy/torch reference implementations
    # TODO: Implement reference functions for each kernel type
    
    reference_implementations = {
        'dot_product': lambda x, y: torch.dot(x.flatten(), y.flatten()),
        'elemwise_operations': lambda x, y: x + y,  # Default to addition
        'matrix_transpose': lambda x: x.T,
        'relu_activation': lambda x: torch.relu(x),
        'matmul': lambda x, y: torch.matmul(x, y),
        # Add more reference implementations as needed
    }
    
    return reference_implementations.get(kernel_name)

if __name__ == "__main__":
    # Example usage
    config = TestConfig(
        test_sizes=[(1024,), (32, 32)],
        num_iterations=10,
        tolerance=1e-4
    )
    
    results = run_all_tests(config)
    
    # Print summary
    for kernel_name, kernel_results in results.items():
        print(f"\n{kernel_name}:")
        for result in kernel_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  {result.platform}: {status} "
                  f"({result.execution_time_ms:.2f}ms, "
                  f"{result.throughput_gb_s:.1f}GB/s)") 