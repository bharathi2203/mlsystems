#!/usr/bin/env python3
"""
Test runner for ML Systems kernels

Usage examples:
    python run_tests.py                           # Run all tests
    python run_tests.py --kernel relu_activation  # Test specific kernel
    python run_tests.py --category fundamentals   # Test kernel category
    python run_tests.py --platform triton         # Test specific platform
    python run_tests.py --benchmark              # Focus on performance
    python run_tests.py --quick                  # Quick correctness test
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

# Add testing directory to path
sys.path.append(str(Path(__file__).parent))

from test_framework import TestConfig, run_all_tests, discover_kernels
from reference_implementations import REFERENCE_IMPLEMENTATIONS

def create_test_report(results: dict, output_file: str = None):
    """Create a detailed test report"""
    report = {
        'summary': {
            'total_kernels': len(results),
            'total_tests': sum(len(kernel_results) for kernel_results in results.values()),
            'passed_tests': 0,
            'failed_tests': 0,
            'platforms_tested': set(),
            'categories_tested': set()
        },
        'results': {}
    }
    
    for kernel_name, kernel_results in results.items():
        kernel_report = {
            'tests': [],
            'summary': {
                'total': len(kernel_results),
                'passed': 0,
                'failed': 0,
                'avg_time_ms': 0.0,
                'max_throughput_gb_s': 0.0
            }
        }
        
        total_time = 0.0
        max_throughput = 0.0
        
        for result in kernel_results:
            test_info = {
                'platform': result.platform,
                'passed': result.passed,
                'execution_time_ms': result.execution_time_ms,
                'throughput_gb_s': result.throughput_gb_s,
                'max_error': result.max_error,
                'error_message': result.error_message
            }
            kernel_report['tests'].append(test_info)
            
            if result.passed:
                kernel_report['summary']['passed'] += 1
                report['summary']['passed_tests'] += 1
            else:
                kernel_report['summary']['failed'] += 1
                report['summary']['failed_tests'] += 1
            
            total_time += result.execution_time_ms
            if result.throughput_gb_s:
                max_throughput = max(max_throughput, result.throughput_gb_s)
            
            report['summary']['platforms_tested'].add(result.platform)
        
        kernel_report['summary']['avg_time_ms'] = total_time / len(kernel_results) if kernel_results else 0
        kernel_report['summary']['max_throughput_gb_s'] = max_throughput
        report['results'][kernel_name] = kernel_report
    
    # Convert set to list for JSON serialization
    report['summary']['platforms_tested'] = list(report['summary']['platforms_tested'])
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report saved to {output_file}")
    
    return report

def print_summary(results: dict):
    """Print a human-readable test summary"""
    total_tests = sum(len(kernel_results) for kernel_results in results.values())
    passed_tests = sum(1 for kernel_results in results.values() 
                      for result in kernel_results if result.passed)
    failed_tests = total_tests - passed_tests
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total kernels tested: {len(results)}")
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
    print(f"Failed: {failed_tests} ({100*failed_tests/total_tests:.1f}%)")
    
    # Platform breakdown
    platform_stats = {}
    for kernel_results in results.values():
        for result in kernel_results:
            if result.platform not in platform_stats:
                platform_stats[result.platform] = {'passed': 0, 'failed': 0}
            if result.passed:
                platform_stats[result.platform]['passed'] += 1
            else:
                platform_stats[result.platform]['failed'] += 1
    
    print(f"\nPlatform breakdown:")
    for platform, stats in platform_stats.items():
        total = stats['passed'] + stats['failed']
        success_rate = 100 * stats['passed'] / total if total > 0 else 0
        print(f"  {platform}: {stats['passed']}/{total} ({success_rate:.1f}%)")
    
    # Performance highlights
    all_results = [result for kernel_results in results.values() 
                   for result in kernel_results if result.passed and result.throughput_gb_s]
    
    if all_results:
        best_throughput = max(all_results, key=lambda r: r.throughput_gb_s)
        fastest_kernel = min(all_results, key=lambda r: r.execution_time_ms)
        
        print(f"\nPerformance highlights:")
        print(f"  Best throughput: {best_throughput.throughput_gb_s:.1f} GB/s "
              f"({best_throughput.kernel_name} on {best_throughput.platform})")
        print(f"  Fastest execution: {fastest_kernel.execution_time_ms:.2f} ms "
              f"({fastest_kernel.kernel_name} on {fastest_kernel.platform})")
    
    # Failed tests details
    if failed_tests > 0:
        print(f"\nFailed tests:")
        for kernel_name, kernel_results in results.items():
            failed_results = [r for r in kernel_results if not r.passed]
            if failed_results:
                print(f"  {kernel_name}:")
                for result in failed_results:
                    error_msg = result.error_message or "Correctness check failed"
                    print(f"    {result.platform}: {error_msg}")

def filter_kernels(kernels: dict, kernel_filter: str = None, 
                  category_filter: str = None) -> dict:
    """Filter kernels based on name or category"""
    if not kernel_filter and not category_filter:
        return kernels
    
    filtered = {}
    for name, info in kernels.items():
        include = True
        
        if kernel_filter and kernel_filter.lower() not in name.lower():
            include = False
        
        if category_filter and category_filter.lower() not in info.get('category', '').lower():
            include = False
        
        if include:
            filtered[name] = info
    
    return filtered

def main():
    parser = argparse.ArgumentParser(description='Test ML Systems kernels')
    parser.add_argument('--kernel', type=str, help='Test specific kernel (partial name matching)')
    parser.add_argument('--category', type=str, help='Test specific category (fundamentals, ml_primitives, etc.)')
    parser.add_argument('--platform', type=str, choices=['cuda', 'metal', 'triton'], 
                       help='Test specific platform only')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Focus on performance benchmarking')
    parser.add_argument('--quick', action='store_true',
                       help='Quick correctness test with minimal iterations')
    parser.add_argument('--sizes', type=str, nargs='+', 
                       help='Test sizes (e.g., "1024" "32,32" "64,64,64")')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                       help='Error tolerance for correctness validation')
    parser.add_argument('--output', type=str,
                       help='Output file for detailed JSON report')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output with per-test details')
    
    args = parser.parse_args()
    
    # Configure test parameters
    if args.quick:
        config = TestConfig(
            test_sizes=[(1024,), (32, 32)],
            num_iterations=3,
            num_warmup=1,
            benchmark_performance=False,
            tolerance=args.tolerance
        )
    elif args.benchmark:
        config = TestConfig(
            test_sizes=[(1024,), (4096,), (16384,), (32, 32), (128, 128), (512, 512)],
            num_iterations=50,
            num_warmup=10,
            validate_correctness=True,
            benchmark_performance=True,
            tolerance=args.tolerance
        )
    else:
        # Parse custom sizes if provided
        test_sizes = None
        if args.sizes:
            test_sizes = []
            for size_str in args.sizes:
                size_tuple = tuple(map(int, size_str.split(',')))
                test_sizes.append(size_tuple)
        
        config = TestConfig(
            test_sizes=test_sizes,
            tolerance=args.tolerance
        )
    
    print("Discovering kernels...")
    all_kernels = discover_kernels()
    
    # Filter kernels
    kernels_to_test = filter_kernels(all_kernels, args.kernel, args.category)
    
    if not kernels_to_test:
        print("No kernels found matching the criteria.")
        return
    
    print(f"Found {len(kernels_to_test)} kernels to test:")
    for name in sorted(kernels_to_test.keys()):
        print(f"  {name}")
    
    # Filter by platform if specified
    if args.platform:
        print(f"Testing only {args.platform} platform")
        # This would require modifying the test framework to support platform filtering
        # For now, we'll run all platforms and filter results later
    
    print(f"\nRunning tests with config:")
    print(f"  Test sizes: {config.test_sizes}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Tolerance: {config.tolerance}")
    print(f"  Benchmark: {config.benchmark_performance}")
    
    # Update the test framework to use reference implementations
    import test_framework
    def get_reference_implementation(kernel_name: str):
        return REFERENCE_IMPLEMENTATIONS.get(kernel_name)
    
    test_framework.get_reference_implementation = get_reference_implementation
    
    # Run tests
    print("\nRunning tests...")
    try:
        results = run_all_tests(config)
        
        # Filter results by platform if specified
        if args.platform:
            filtered_results = {}
            for kernel_name, kernel_results in results.items():
                filtered_kernel_results = [r for r in kernel_results if r.platform == args.platform]
                if filtered_kernel_results:
                    filtered_results[kernel_name] = filtered_kernel_results
            results = filtered_results
        
        # Create detailed report
        if args.output:
            create_test_report(results, args.output)
        
        # Print verbose details if requested
        if args.verbose:
            print(f"\nDetailed results:")
            for kernel_name, kernel_results in results.items():
                print(f"\n{kernel_name}:")
                for result in kernel_results:
                    status = "PASS" if result.passed else "FAIL"
                    details = f"{result.execution_time_ms:.2f}ms"
                    if result.throughput_gb_s:
                        details += f", {result.throughput_gb_s:.1f}GB/s"
                    if result.max_error is not None:
                        details += f", max_error={result.max_error:.2e}"
                    if result.error_message:
                        details += f", error: {result.error_message}"
                    print(f"  {result.platform}: {status} ({details})")
        
        # Print summary
        print_summary(results)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nError running tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 