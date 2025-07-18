#!/usr/bin/env python3
"""
Example test demonstrating the kernel testing framework

This shows how to test a simple Triton kernel implementation.
"""

import torch
import triton
import triton.language as tl
from test_framework import TritonKernelTester, TestConfig, KernelTestResult
from reference_implementations import ReferenceImplementations

# Simple Triton kernel for demonstration
@triton.jit
def vector_add_kernel(
    a_ptr,  # Pointer to first input vector
    b_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Simple vector addition kernel in Triton"""
    # Calculate starting position for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load input data
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform computation
    output = a + b
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Host function for Triton vector addition"""
    # Validate inputs
    assert a.shape == b.shape, "Input tensors must have same shape"
    assert a.is_contiguous() and b.is_contiguous(), "Tensors must be contiguous"
    
    # Allocate output
    output = torch.empty_like(a)
    n_elements = a.numel()
    
    # Choose block size (must be power of 2)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    vector_add_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

class MockVectorAddModule:
    """Mock module to simulate imported Triton kernel"""
    def __init__(self):
        pass
    
    @staticmethod
    def vector_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return vector_add_triton(a, b)

def test_vector_add_example():
    """Example test function showing how to test a Triton kernel"""
    print("Testing vector addition kernel...")
    
    # Configure test
    config = TestConfig(
        test_sizes=[(1024,), (4096,), (16384,)],
        data_types=[torch.float32],
        tolerance=1e-6,
        num_iterations=10,
        num_warmup=3
    )
    
    # Create tester
    tester = TritonKernelTester(config)
    
    # Reference implementation
    def reference_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b
    
    # Create mock module
    kernel_module = MockVectorAddModule()
    
    # Run test
    results = tester.test_kernel(kernel_module, "vector_add", reference_vector_add)
    
    # Print results
    print(f"\nTest Results:")
    print(f"{'='*50}")
    
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.kernel_name}: {status}")
        print(f"  Platform: {result.platform}")
        print(f"  Execution time: {result.execution_time_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_gb_s:.1f} GB/s")
        print(f"  Max error: {result.max_error:.2e}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        print()
    
    return results

def run_comprehensive_example():
    """Run a more comprehensive test example"""
    print("Running comprehensive kernel testing example...")
    
    # Test multiple kernel types
    kernels_to_test = [
        ("relu_activation", ReferenceImplementations.relu_activation),
        ("elemwise_add", ReferenceImplementations.elemwise_add),
        ("matrix_transpose", ReferenceImplementations.matrix_transpose),
    ]
    
    config = TestConfig(
        test_sizes=[(1024,), (32, 32), (64, 64)],
        data_types=[torch.float32, torch.float16],
        tolerance=1e-5,
        benchmark_performance=True
    )
    
    tester = TritonKernelTester(config)
    all_results = {}
    
    for kernel_name, reference_func in kernels_to_test:
        print(f"\nTesting {kernel_name}...")
        
        # Create mock module (in real usage, this would be imported)
        class MockModule:
            pass
        
        mock_module = MockModule()
        # In real usage, the module would have the actual Triton implementation
        # For this example, we'll simulate a missing implementation
        
        try:
            results = tester.test_kernel(mock_module, kernel_name, reference_func)
            all_results[kernel_name] = results
        except Exception as e:
            print(f"  Failed to test {kernel_name}: {e}")
            all_results[kernel_name] = [KernelTestResult(
                kernel_name=kernel_name,
                platform="triton",
                passed=False,
                execution_time_ms=0.0,
                error_message=str(e)
            )]
    
    # Print summary
    total_tests = sum(len(results) for results in all_results.values())
    passed_tests = sum(1 for results in all_results.values() 
                      for result in results if result.passed)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {100 * passed_tests / total_tests:.1f}%")
    
    return all_results

if __name__ == "__main__":
    # Check if Triton is available
    try:
        import triton
        print("Triton is available - running vector addition test")
        test_vector_add_example()
        
        print("\n" + "="*60)
        print("Running comprehensive example")
        run_comprehensive_example()
        
    except ImportError:
        print("Triton not available - showing framework structure only")
        print("Install Triton with: pip install triton")
        print("\nFramework components:")
        print("- test_framework.py: Core testing infrastructure")
        print("- reference_implementations.py: PyTorch reference functions")
        print("- run_tests.py: Command-line test runner")
        print("- example_test.py: This demonstration file")
    
    print(f"\n{'='*60}")
    print("Example complete!")
    print("To test your actual kernels:")
    print("  1. Implement the kernel following naming conventions")
    print("  2. Add reference implementation if needed")
    print("  3. Run: python run_tests.py --kernel your_kernel_name")
    print(f"{'='*60}") 