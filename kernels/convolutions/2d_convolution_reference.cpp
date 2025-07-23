/*
 * 2D Convolution - C++ Reference Implementation
 * 
 * Single-threaded CPU implementation of 2D convolution operation
 * for validation and educational purposes.
 * 
 * Mathematical foundation:
 * - 2D Convolution: (f * g)[i,j] = ΣΣ f[m,n] * g[i-m,j-n] for all m,n
 * - 2D Cross-correlation: (f ★ g)[i,j] = ΣΣ f[m,n] * g[i+m,j+n] for all m,n
 * - Valid convolution: output size = (input_size - kernel_size + 1)
 * - Same convolution: output size = input_size (with padding)
 * - Full convolution: output size = (input_size + kernel_size - 1)
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
#include <random>

/**
 * Class: Image2D
 * Simple 2D image container with basic operations
 */
class Image2D {
private:
    std::vector<float> data_;
    int rows_, cols_;

public:
    Image2D(int rows, int cols) : rows_(rows), cols_(cols) {
        data_.resize(rows * cols, 0.0f);
    }
    
    Image2D(const std::vector<float>& data, int rows, int cols) 
        : data_(data), rows_(rows), cols_(cols) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument("Data size doesn't match dimensions");
        }
    }
    
    Image2D(const float* data, int rows, int cols) 
        : rows_(rows), cols_(cols) {
        data_.assign(data, data + rows * cols);
    }
    
    // Accessors
    float& operator()(int row, int col) {
        return data_[row * cols_ + col];
    }
    
    const float& operator()(int row, int col) const {
        return data_[row * cols_ + col];
    }
    
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    const float* data() const { return data_.data(); }
    float* data() { return data_.data(); }
    
    // Basic operations
    void zero() {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    void randomize(float min_val = -1.0f, float max_val = 1.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        
        for (float& val : data_) {
            val = dis(gen);
        }
    }
    
    // Padding operations
    Image2D pad(int top, int bottom, int left, int right, float value = 0.0f) const {
        int new_rows = rows_ + top + bottom;
        int new_cols = cols_ + left + right;
        Image2D padded(new_rows, new_cols);
        
        // Initialize with padding value
        for (int r = 0; r < new_rows; r++) {
            for (int c = 0; c < new_cols; c++) {
                padded(r, c) = value;
            }
        }
        
        // Copy original data
        for (int r = 0; r < rows_; r++) {
            for (int c = 0; c < cols_; c++) {
                padded(r + top, c + left) = (*this)(r, c);
            }
        }
        
        return padded;
    }
    
    Image2D pad_symmetric(int pad_rows, int pad_cols) const {
        return pad(pad_rows, pad_rows, pad_cols, pad_cols, 0.0f);
    }
};

/**
 * Class: Convolution2D
 * 2D convolution operations with different modes and options
 */
class Convolution2D {
public:
    enum Mode {
        VALID,  // Output size = input_size - kernel_size + 1
        SAME,   // Output size = input_size (with padding)
        FULL    // Output size = input_size + kernel_size - 1
    };
    
private:
    static std::pair<int, int> compute_output_size(
        int input_rows, int input_cols,
        int kernel_rows, int kernel_cols,
        Mode mode, int stride_y = 1, int stride_x = 1
    ) {
        int out_rows, out_cols;
        
        switch (mode) {
            case VALID:
                out_rows = (input_rows - kernel_rows) / stride_y + 1;
                out_cols = (input_cols - kernel_cols) / stride_x + 1;
                break;
            case SAME:
                out_rows = (input_rows + stride_y - 1) / stride_y;
                out_cols = (input_cols + stride_x - 1) / stride_x;
                break;
            case FULL:
                out_rows = (input_rows + kernel_rows - 1 + stride_y - 1) / stride_y;
                out_cols = (input_cols + kernel_cols - 1 + stride_x - 1) / stride_x;
                break;
            default:
                throw std::invalid_argument("Invalid convolution mode");
        }
        
        return {out_rows, out_cols};
    }
    
    static std::tuple<int, int, int, int> compute_padding(
        int input_rows, int input_cols,
        int kernel_rows, int kernel_cols,
        Mode mode, int stride_y = 1, int stride_x = 1
    ) {
        switch (mode) {
            case VALID:
                return {0, 0, 0, 0};
            case SAME: {
                auto [out_rows, out_cols] = compute_output_size(
                    input_rows, input_cols, kernel_rows, kernel_cols, SAME, stride_y, stride_x
                );
                int pad_total_y = std::max(0, (out_rows - 1) * stride_y + kernel_rows - input_rows);
                int pad_total_x = std::max(0, (out_cols - 1) * stride_x + kernel_cols - input_cols);
                int pad_top = pad_total_y / 2;
                int pad_bottom = pad_total_y - pad_top;
                int pad_left = pad_total_x / 2;
                int pad_right = pad_total_x - pad_left;
                return {pad_top, pad_bottom, pad_left, pad_right};
            }
            case FULL: {
                int pad_y = kernel_rows - 1;
                int pad_x = kernel_cols - 1;
                return {pad_y, pad_y, pad_x, pad_x};
            }
            default:
                throw std::invalid_argument("Invalid convolution mode");
        }
    }

public:
    /**
     * Basic 2D convolution
     */
    static Image2D convolve(const Image2D& input, const Image2D& kernel, 
                           Mode mode = VALID, int stride_y = 1, int stride_x = 1) {
        if (stride_y <= 0 || stride_x <= 0) {
            throw std::invalid_argument("Strides must be positive");
        }
        
        int input_rows = input.rows();
        int input_cols = input.cols();
        int kernel_rows = kernel.rows();
        int kernel_cols = kernel.cols();
        
        if ((kernel_rows > input_rows || kernel_cols > input_cols) && mode == VALID) {
            throw std::invalid_argument("Kernel size cannot be larger than input for valid convolution");
        }
        
        // Compute output size and padding
        auto [output_rows, output_cols] = compute_output_size(
            input_rows, input_cols, kernel_rows, kernel_cols, mode, stride_y, stride_x
        );
        auto [pad_top, pad_bottom, pad_left, pad_right] = compute_padding(
            input_rows, input_cols, kernel_rows, kernel_cols, mode, stride_y, stride_x
        );
        
        // Create padded input
        Image2D padded_input = input.pad(pad_top, pad_bottom, pad_left, pad_right, 0.0f);
        Image2D output(output_rows, output_cols);
        
        // Perform convolution
        for (int out_r = 0; out_r < output_rows; out_r++) {
            for (int out_c = 0; out_c < output_cols; out_c++) {
                float sum = 0.0f;
                int input_start_r = out_r * stride_y;
                int input_start_c = out_c * stride_x;
                
                for (int kr = 0; kr < kernel_rows; kr++) {
                    for (int kc = 0; kc < kernel_cols; kc++) {
                        int input_r = input_start_r + kr;
                        int input_c = input_start_c + kc;
                        
                        if (input_r < padded_input.rows() && input_c < padded_input.cols()) {
                            sum += padded_input(input_r, input_c) * kernel(kr, kc);
                        }
                    }
                }
                
                output(out_r, out_c) = sum;
            }
        }
        
        return output;
    }
    
    /**
     * 2D cross-correlation (commonly used in neural networks)
     */
    static Image2D cross_correlate(const Image2D& input, const Image2D& kernel) {
        int input_rows = input.rows();
        int input_cols = input.cols();
        int kernel_rows = kernel.rows();
        int kernel_cols = kernel.cols();
        
        int output_rows = input_rows - kernel_rows + 1;
        int output_cols = input_cols - kernel_cols + 1;
        
        if (output_rows <= 0 || output_cols <= 0) {
            throw std::invalid_argument("Kernel size too large for cross-correlation");
        }
        
        Image2D output(output_rows, output_cols);
        
        for (int out_r = 0; out_r < output_rows; out_r++) {
            for (int out_c = 0; out_c < output_cols; out_c++) {
                float sum = 0.0f;
                
                for (int kr = 0; kr < kernel_rows; kr++) {
                    for (int kc = 0; kc < kernel_cols; kc++) {
                        int input_r = out_r + kr;
                        int input_c = out_c + kc;
                        
                        // Note: kernel indexing for cross-correlation (no flipping)
                        sum += input(input_r, input_c) * kernel(kernel_rows - 1 - kr, kernel_cols - 1 - kc);
                    }
                }
                
                output(out_r, out_c) = sum;
            }
        }
        
        return output;
    }
    
    /**
     * Separable 2D convolution (for efficiency with separable kernels)
     */
    static Image2D separable_convolve(const Image2D& input, 
                                     const std::vector<float>& h_kernel,
                                     const std::vector<float>& v_kernel) {
        // Horizontal pass
        int input_rows = input.rows();
        int input_cols = input.cols();
        int h_kernel_size = h_kernel.size();
        int v_kernel_size = v_kernel.size();
        
        int intermediate_cols = input_cols - h_kernel_size + 1;
        Image2D intermediate(input_rows, intermediate_cols);
        
        for (int r = 0; r < input_rows; r++) {
            for (int c = 0; c < intermediate_cols; c++) {
                float sum = 0.0f;
                for (int k = 0; k < h_kernel_size; k++) {
                    sum += input(r, c + k) * h_kernel[k];
                }
                intermediate(r, c) = sum;
            }
        }
        
        // Vertical pass
        int output_rows = intermediate.rows() - v_kernel_size + 1;
        int output_cols = intermediate.cols();
        Image2D output(output_rows, output_cols);
        
        for (int r = 0; r < output_rows; r++) {
            for (int c = 0; c < output_cols; c++) {
                float sum = 0.0f;
                for (int k = 0; k < v_kernel_size; k++) {
                    sum += intermediate(r + k, c) * v_kernel[k];
                }
                output(r, c) = sum;
            }
        }
        
        return output;
    }
    
    /**
     * Dilated (atrous) convolution
     */
    static Image2D dilated_convolve(const Image2D& input, const Image2D& kernel,
                                   int dilation_y = 1, int dilation_x = 1) {
        int input_rows = input.rows();
        int input_cols = input.cols();
        int kernel_rows = kernel.rows();
        int kernel_cols = kernel.cols();
        
        int effective_kernel_rows = (kernel_rows - 1) * dilation_y + 1;
        int effective_kernel_cols = (kernel_cols - 1) * dilation_x + 1;
        
        int output_rows = input_rows - effective_kernel_rows + 1;
        int output_cols = input_cols - effective_kernel_cols + 1;
        
        if (output_rows <= 0 || output_cols <= 0) {
            throw std::invalid_argument("Dilated kernel size too large for input");
        }
        
        Image2D output(output_rows, output_cols);
        
        for (int out_r = 0; out_r < output_rows; out_r++) {
            for (int out_c = 0; out_c < output_cols; out_c++) {
                float sum = 0.0f;
                
                for (int kr = 0; kr < kernel_rows; kr++) {
                    for (int kc = 0; kc < kernel_cols; kc++) {
                        int input_r = out_r + kr * dilation_y;
                        int input_c = out_c + kc * dilation_x;
                        
                        if (input_r < input_rows && input_c < input_cols) {
                            sum += input(input_r, input_c) * kernel(kr, kc);
                        }
                    }
                }
                
                output(out_r, out_c) = sum;
            }
        }
        
        return output;
    }
};

/**
 * Utility functions for testing and validation
 */
namespace ConvolutionUtils2D {
    /**
     * Compare two images with tolerance
     */
    bool images_equal(const Image2D& a, const Image2D& b, float tolerance = 1e-6f) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            return false;
        }
        
        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.cols(); c++) {
                if (std::abs(a(r, c) - b(r, c)) > tolerance) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * Compute maximum absolute difference between images
     */
    float max_difference(const Image2D& a, const Image2D& b) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            return INFINITY;
        }
        
        float max_diff = 0.0f;
        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.cols(); c++) {
                max_diff = std::max(max_diff, std::abs(a(r, c) - b(r, c)));
            }
        }
        
        return max_diff;
    }
    
    /**
     * Print image values (limited output)
     */
    void print_image(const Image2D& image, const std::string& name = "", 
                     int max_rows = 5, int max_cols = 5) {
        if (!name.empty()) {
            std::cout << name << " (" << image.rows() << "x" << image.cols() << "):" << std::endl;
        }
        
        int print_rows = std::min(image.rows(), max_rows);
        int print_cols = std::min(image.cols(), max_cols);
        
        for (int r = 0; r < print_rows; r++) {
            for (int c = 0; c < print_cols; c++) {
                std::cout << std::fixed << std::setprecision(3) << image(r, c) << " ";
            }
            if (image.cols() > max_cols) {
                std::cout << "...";
            }
            std::cout << std::endl;
        }
        
        if (image.rows() > max_rows) {
            std::cout << "..." << std::endl;
        }
        std::cout << std::endl;
    }
    
    /**
     * Generate test images and kernels
     */
    Image2D generate_impulse_2d(int rows, int cols, int pos_r = -1, int pos_c = -1, float amplitude = 1.0f) {
        Image2D image(rows, cols);
        if (pos_r < 0) pos_r = rows / 2;
        if (pos_c < 0) pos_c = cols / 2;
        
        if (pos_r >= 0 && pos_r < rows && pos_c >= 0 && pos_c < cols) {
            image(pos_r, pos_c) = amplitude;
        }
        
        return image;
    }
    
    Image2D generate_gaussian_kernel_2d(int size, float sigma = 1.0f) {
        Image2D kernel(size, size);
        int center = size / 2;
        float sum = 0.0f;
        
        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                float x = static_cast<float>(r - center);
                float y = static_cast<float>(c - center);
                float val = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
                kernel(r, c) = val;
                sum += val;
            }
        }
        
        // Normalize
        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                kernel(r, c) /= sum;
            }
        }
        
        return kernel;
    }
    
    Image2D generate_sobel_x_kernel() {
        Image2D kernel(3, 3);
        kernel(0, 0) = -1; kernel(0, 1) =  0; kernel(0, 2) =  1;
        kernel(1, 0) = -2; kernel(1, 1) =  0; kernel(1, 2) =  2;
        kernel(2, 0) = -1; kernel(2, 1) =  0; kernel(2, 2) =  1;
        return kernel;
    }
    
    Image2D generate_sobel_y_kernel() {
        Image2D kernel(3, 3);
        kernel(0, 0) = -1; kernel(0, 1) = -2; kernel(0, 2) = -1;
        kernel(1, 0) =  0; kernel(1, 1) =  0; kernel(1, 2) =  0;
        kernel(2, 0) =  1; kernel(2, 1) =  2; kernel(2, 2) =  1;
        return kernel;
    }
}

/**
 * C-style interface functions matching CUDA/GPU API
 */
int convolution_2d_reference(
    const float* input_data,
    const float* kernel_data,
    float* output_data,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols,
    int mode = 0  // 0=valid, 1=same, 2=full
) {
    try {
        Image2D input(input_data, input_rows, input_cols);
        Image2D kernel(kernel_data, kernel_rows, kernel_cols);
        
        Convolution2D::Mode conv_mode;
        switch (mode) {
            case 0: conv_mode = Convolution2D::VALID; break;
            case 1: conv_mode = Convolution2D::SAME; break;
            case 2: conv_mode = Convolution2D::FULL; break;
            default: return -1; // Invalid mode
        }
        
        Image2D output = Convolution2D::convolve(input, kernel, conv_mode);
        
        // Copy result to output buffer
        std::copy(output.data(), output.data() + output.rows() * output.cols(), output_data);
        
        return output.rows() * output.cols(); // Return output size
    }
    catch (const std::exception& e) {
        return -1; // Error
    }
}

int cross_correlation_2d_reference(
    const float* input_data,
    const float* kernel_data,
    float* output_data,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
) {
    try {
        Image2D input(input_data, input_rows, input_cols);
        Image2D kernel(kernel_data, kernel_rows, kernel_cols);
        
        Image2D output = Convolution2D::cross_correlate(input, kernel);
        
        // Copy result to output buffer
        std::copy(output.data(), output.data() + output.rows() * output.cols(), output_data);
        
        return output.rows() * output.cols(); // Return output size
    }
    catch (const std::exception& e) {
        return -1; // Error
    }
}

/**
 * Test functions
 */
void test_convolution_2d_reference() {
    std::cout << "Testing 2D Convolution Reference Implementation..." << std::endl;
    
    // Test 1: Impulse response
    Image2D impulse = ConvolutionUtils2D::generate_impulse_2d(10, 10, 5, 5, 1.0f);
    Image2D gaussian = ConvolutionUtils2D::generate_gaussian_kernel_2d(5, 1.0f);
    
    Image2D result = Convolution2D::convolve(impulse, gaussian, Convolution2D::VALID);
    ConvolutionUtils2D::print_image(result, "Impulse response");
    
    // Test 2: Edge detection
    Image2D test_image(8, 8);
    // Create a simple pattern
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 4; c++) {
            test_image(r, c) = 1.0f;
        }
    }
    
    Image2D sobel_x = ConvolutionUtils2D::generate_sobel_x_kernel();
    Image2D edges = Convolution2D::convolve(test_image, sobel_x, Convolution2D::VALID);
    ConvolutionUtils2D::print_image(edges, "Edge detection");
    
    // Test 3: Different modes
    Image2D small_input(5, 5);
    small_input.randomize();
    Image2D small_kernel(3, 3);
    small_kernel.randomize();
    
    Image2D valid_result = Convolution2D::convolve(small_input, small_kernel, Convolution2D::VALID);
    Image2D same_result = Convolution2D::convolve(small_input, small_kernel, Convolution2D::SAME);
    Image2D full_result = Convolution2D::convolve(small_input, small_kernel, Convolution2D::FULL);
    
    std::cout << "Mode sizes - Valid: " << valid_result.rows() << "x" << valid_result.cols()
              << ", Same: " << same_result.rows() << "x" << same_result.cols()
              << ", Full: " << full_result.rows() << "x" << full_result.cols() << std::endl;
    
    std::cout << "All 2D convolution tests completed!" << std::endl;
}

#ifdef CONVOLUTION_2D_MAIN
int main() {
    test_convolution_2d_reference();
    return 0;
}
#endif 