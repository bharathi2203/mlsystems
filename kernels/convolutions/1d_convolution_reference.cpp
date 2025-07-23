/*
 * 1D Convolution - C++ Reference Implementation
 * 
 * Single-threaded CPU implementation of 1D convolution operation
 * for validation and educational purposes.
 * 
 * Mathematical foundation:
 * - Convolution: (f * g)[n] = Σ f[m] * g[n-m] for all m
 * - Cross-correlation: (f ★ g)[n] = Σ f[m] * g[n+m] for all m
 * - Valid convolution: output size = input_size - kernel_size + 1
 * - Full convolution: output size = input_size + kernel_size - 1
 * - Same convolution: output size = input_size (with padding)
 * 
 * Implementation notes:
 * - Single precision floating point operations
 * - Boundary handling with zero-padding
 * - Direct implementation for clarity and correctness
 */

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>

/**
 * Class: Signal1D
 * Simple 1D signal container with basic operations
 */
class Signal1D {
private:
    std::vector<float> data_;
    int size_;

public:
    Signal1D(int size) : size_(size) {
        data_.resize(size, 0.0f);
    }
    
    Signal1D(const std::vector<float>& data) : data_(data), size_(data.size()) {}
    
    Signal1D(const float* data, int size) : size_(size) {
        data_.assign(data, data + size);
    }
    
    // Accessors
    float& operator[](int index) { return data_[index]; }
    const float& operator[](int index) const { return data_[index]; }
    
    int size() const { return size_; }
    const float* data() const { return data_.data(); }
    float* data() { return data_.data(); }
    
    // Basic operations
    void zero() {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    void randomize(float min_val = -1.0f, float max_val = 1.0f) {
        for (int i = 0; i < size_; i++) {
            data_[i] = min_val + (max_val - min_val) * static_cast<float>(rand()) / RAND_MAX;
        }
    }
    
    // Padding operations
    Signal1D pad(int left_pad, int right_pad, float value = 0.0f) const {
        Signal1D padded(size_ + left_pad + right_pad);
        
        // Left padding
        for (int i = 0; i < left_pad; i++) {
            padded[i] = value;
        }
        
        // Original data
        for (int i = 0; i < size_; i++) {
            padded[left_pad + i] = data_[i];
        }
        
        // Right padding
        for (int i = 0; i < right_pad; i++) {
            padded[left_pad + size_ + i] = value;
        }
        
        return padded;
    }
};

/**
 * Class: Convolution1D
 * 1D convolution operations with different modes and options
 */
class Convolution1D {
public:
    enum Mode {
        VALID,  // Output size = input_size - kernel_size + 1
        SAME,   // Output size = input_size (with padding)
        FULL    // Output size = input_size + kernel_size - 1
    };
    
private:
    static int compute_output_size(int input_size, int kernel_size, Mode mode, int stride) {
        switch (mode) {
            case VALID:
                return (input_size - kernel_size) / stride + 1;
            case SAME:
                return (input_size + stride - 1) / stride;
            case FULL:
                return (input_size + kernel_size - 1 + stride - 1) / stride;
            default:
                throw std::invalid_argument("Invalid convolution mode");
        }
    }
    
    static std::pair<int, int> compute_padding(int input_size, int kernel_size, Mode mode, int stride) {
        switch (mode) {
            case VALID:
                return {0, 0};
            case SAME: {
                int output_size = (input_size + stride - 1) / stride;
                int pad_total = std::max(0, (output_size - 1) * stride + kernel_size - input_size);
                int pad_left = pad_total / 2;
                int pad_right = pad_total - pad_left;
                return {pad_left, pad_right};
            }
            case FULL: {
                int pad_size = kernel_size - 1;
                return {pad_size, pad_size};
            }
            default:
                throw std::invalid_argument("Invalid convolution mode");
        }
    }

public:
    /**
     * Basic 1D convolution
     */
    static Signal1D convolve(const Signal1D& input, const Signal1D& kernel, 
                           Mode mode = VALID, int stride = 1) {
        if (stride <= 0) {
            throw std::invalid_argument("Stride must be positive");
        }
        
        int input_size = input.size();
        int kernel_size = kernel.size();
        
        if (kernel_size > input_size && mode == VALID) {
            throw std::invalid_argument("Kernel size cannot be larger than input for valid convolution");
        }
        
        // Compute output size and padding
        int output_size = compute_output_size(input_size, kernel_size, mode, stride);
        auto [pad_left, pad_right] = compute_padding(input_size, kernel_size, mode, stride);
        
        // Create padded input
        Signal1D padded_input = input.pad(pad_left, pad_right, 0.0f);
        Signal1D output(output_size);
        
        // Perform convolution
        for (int out_idx = 0; out_idx < output_size; out_idx++) {
            float sum = 0.0f;
            int input_start = out_idx * stride;
            
            for (int k = 0; k < kernel_size; k++) {
                int input_idx = input_start + k;
                if (input_idx < padded_input.size()) {
                    sum += padded_input[input_idx] * kernel[k];
                }
            }
            
            output[out_idx] = sum;
        }
        
        return output;
    }
    
    /**
     * 1D cross-correlation (commonly used in neural networks)
     */
    static Signal1D cross_correlate(const Signal1D& input, const Signal1D& kernel) {
        int input_size = input.size();
        int kernel_size = kernel.size();
        int output_size = input_size - kernel_size + 1;
        
        if (output_size <= 0) {
            throw std::invalid_argument("Kernel size too large for cross-correlation");
        }
        
        Signal1D output(output_size);
        
        for (int out_idx = 0; out_idx < output_size; out_idx++) {
            float sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int input_idx = out_idx + k;
                // Note: no kernel flipping for cross-correlation
                sum += input[input_idx] * kernel[kernel_size - 1 - k];
            }
            
            output[out_idx] = sum;
        }
        
        return output;
    }
    
    /**
     * Separable convolution (useful for efficiency with separable kernels)
     */
    static Signal1D separable_convolve(const Signal1D& input, const Signal1D& kernel1, 
                                     const Signal1D& kernel2) {
        Signal1D intermediate = convolve(input, kernel1, VALID);
        return convolve(intermediate, kernel2, VALID);
    }
};

/**
 * Utility functions for testing and validation
 */
namespace ConvolutionUtils {
    /**
     * Compare two signals with tolerance
     */
    bool signals_equal(const Signal1D& a, const Signal1D& b, float tolerance = 1e-6f) {
        if (a.size() != b.size()) {
            return false;
        }
        
        for (int i = 0; i < a.size(); i++) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Compute maximum absolute difference between signals
     */
    float max_difference(const Signal1D& a, const Signal1D& b) {
        if (a.size() != b.size()) {
            return INFINITY;
        }
        
        float max_diff = 0.0f;
        for (int i = 0; i < a.size(); i++) {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        
        return max_diff;
    }
    
    /**
     * Print signal values
     */
    void print_signal(const Signal1D& signal, const std::string& name = "", int max_elements = 10) {
        if (!name.empty()) {
            std::cout << name << ": ";
        }
        
        int print_count = std::min(signal.size(), max_elements);
        for (int i = 0; i < print_count; i++) {
            std::cout << signal[i] << " ";
        }
        
        if (signal.size() > max_elements) {
            std::cout << "... (" << signal.size() << " total)";
        }
        
        std::cout << std::endl;
    }
    
    /**
     * Generate test signals
     */
    Signal1D generate_impulse(int size, int position = 0, float amplitude = 1.0f) {
        Signal1D signal(size);
        if (position >= 0 && position < size) {
            signal[position] = amplitude;
        }
        return signal;
    }
    
    Signal1D generate_step(int size, int step_position = 0, float amplitude = 1.0f) {
        Signal1D signal(size);
        for (int i = step_position; i < size; i++) {
            signal[i] = amplitude;
        }
        return signal;
    }
    
    Signal1D generate_gaussian_kernel(int size, float sigma = 1.0f) {
        Signal1D kernel(size);
        int center = size / 2;
        float sum = 0.0f;
        
        for (int i = 0; i < size; i++) {
            float x = static_cast<float>(i - center);
            kernel[i] = std::exp(-x * x / (2.0f * sigma * sigma));
            sum += kernel[i];
        }
        
        // Normalize
        for (int i = 0; i < size; i++) {
            kernel[i] /= sum;
        }
        
        return kernel;
    }
}

/**
 * C-style interface functions matching CUDA/GPU API
 */
int convolution_1d_reference(
    const float* input_data,
    const float* kernel_data,
    float* output_data,
    int input_size,
    int kernel_size,
    int mode = 0  // 0=valid, 1=same, 2=full
) {
    try {
        Signal1D input(input_data, input_size);
        Signal1D kernel(kernel_data, kernel_size);
        
        Convolution1D::Mode conv_mode;
        switch (mode) {
            case 0: conv_mode = Convolution1D::VALID; break;
            case 1: conv_mode = Convolution1D::SAME; break;
            case 2: conv_mode = Convolution1D::FULL; break;
            default: return -1; // Invalid mode
        }
        
        Signal1D output = Convolution1D::convolve(input, kernel, conv_mode);
        
        // Copy result to output buffer
        std::copy(output.data(), output.data() + output.size(), output_data);
        
        return output.size(); // Return output size
    }
    catch (const std::exception& e) {
        return -1; // Error
    }
}

int cross_correlation_1d_reference(
    const float* input_data,
    const float* kernel_data,
    float* output_data,
    int input_size,
    int kernel_size
) {
    try {
        Signal1D input(input_data, input_size);
        Signal1D kernel(kernel_data, kernel_size);
        
        Signal1D output = Convolution1D::cross_correlate(input, kernel);
        
        // Copy result to output buffer
        std::copy(output.data(), output.data() + output.size(), output_data);
        
        return output.size(); // Return output size
    }
    catch (const std::exception& e) {
        return -1; // Error
    }
}

/**
 * Test functions
 */
void test_convolution_1d_reference() {
    std::cout << "Testing 1D Convolution Reference Implementation..." << std::endl;
    
    // Test 1: Impulse response
    Signal1D impulse = ConvolutionUtils::generate_impulse(10, 5, 1.0f);
    Signal1D gaussian = ConvolutionUtils::generate_gaussian_kernel(5, 1.0f);
    
    Signal1D result = Convolution1D::convolve(impulse, gaussian, Convolution1D::VALID);
    ConvolutionUtils::print_signal(result, "Impulse response");
    
    // Test 2: Cross-correlation vs convolution
    Signal1D signal(5);
    signal[0] = 1; signal[1] = 2; signal[2] = 3; signal[3] = 4; signal[4] = 5;
    
    Signal1D kernel(3);
    kernel[0] = 0.5f; kernel[1] = 1.0f; kernel[2] = 0.5f;
    
    Signal1D conv_result = Convolution1D::convolve(signal, kernel, Convolution1D::VALID);
    Signal1D xcorr_result = Convolution1D::cross_correlate(signal, kernel);
    
    ConvolutionUtils::print_signal(conv_result, "Convolution");
    ConvolutionUtils::print_signal(xcorr_result, "Cross-correlation");
    
    // Test 3: Different modes
    Signal1D valid_result = Convolution1D::convolve(signal, kernel, Convolution1D::VALID);
    Signal1D same_result = Convolution1D::convolve(signal, kernel, Convolution1D::SAME);
    Signal1D full_result = Convolution1D::convolve(signal, kernel, Convolution1D::FULL);
    
    std::cout << "Mode sizes - Valid: " << valid_result.size() 
              << ", Same: " << same_result.size() 
              << ", Full: " << full_result.size() << std::endl;
    
    std::cout << "All tests completed!" << std::endl;
}

#ifdef CONVOLUTION_1D_MAIN
int main() {
    test_convolution_1d_reference();
    return 0;
}
#endif 