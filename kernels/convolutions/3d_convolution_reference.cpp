/*
 * 3D Convolution - C++ Reference Implementation
 * 
 * Single-threaded CPU implementation of 3D convolution operation
 * for validation and educational purposes.
 * 
 * Mathematical foundation:
 * - 3D Convolution: (f * g)[i,j,k] = ΣΣΣ f[m,n,p] * g[i-m,j-n,k-p] for all m,n,p
 * - 3D Cross-correlation: (f ★ g)[i,j,k] = ΣΣΣ f[m,n,p] * g[i+m,j+n,k+p] for all m,n,p
 * - Valid convolution: output size = (input_size - kernel_size + 1)
 * - Same convolution: output size = input_size (with padding)
 * - Full convolution: output size = (input_size + kernel_size - 1)
 * 
 * Implementation notes:
 * - Single precision floating point operations
 * - 3D boundary handling with zero-padding
 * - Direct implementation for clarity and correctness
 */

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>

/**
 * Class: Volume3D
 * Simple 3D volume container with basic operations
 */
class Volume3D {
private:
    std::vector<float> data_;
    int depth_, height_, width_;

public:
    Volume3D(int depth, int height, int width) 
        : depth_(depth), height_(height), width_(width) {
        data_.resize(depth * height * width, 0.0f);
    }
    
    Volume3D(const std::vector<float>& data, int depth, int height, int width) 
        : data_(data), depth_(depth), height_(height), width_(width) {
        if (data.size() != depth * height * width) {
            throw std::invalid_argument("Data size doesn't match dimensions");
        }
    }
    
    Volume3D(const float* data, int depth, int height, int width) 
        : depth_(depth), height_(height), width_(width) {
        data_.assign(data, data + depth * height * width);
    }
    
    // Accessors
    float& operator()(int d, int h, int w) {
        return data_[d * height_ * width_ + h * width_ + w];
    }
    
    const float& operator()(int d, int h, int w) const {
        return data_[d * height_ * width_ + h * width_ + w];
    }
    
    int depth() const { return depth_; }
    int height() const { return height_; }
    int width() const { return width_; }
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
    Volume3D pad(int front, int back, int top, int bottom, int left, int right, float value = 0.0f) const {
        int new_depth = depth_ + front + back;
        int new_height = height_ + top + bottom;
        int new_width = width_ + left + right;
        Volume3D padded(new_depth, new_height, new_width);
        
        // Initialize with padding value
        for (int d = 0; d < new_depth; d++) {
            for (int h = 0; h < new_height; h++) {
                for (int w = 0; w < new_width; w++) {
                    padded(d, h, w) = value;
                }
            }
        }
        
        // Copy original data
        for (int d = 0; d < depth_; d++) {
            for (int h = 0; h < height_; h++) {
                for (int w = 0; w < width_; w++) {
                    padded(d + front, h + top, w + left) = (*this)(d, h, w);
                }
            }
        }
        
        return padded;
    }
    
    Volume3D pad_symmetric(int pad_d, int pad_h, int pad_w) const {
        return pad(pad_d, pad_d, pad_h, pad_h, pad_w, pad_w, 0.0f);
    }
};

/**
 * Class: Convolution3D
 * 3D convolution operations with different modes and options
 */
class Convolution3D {
public:
    enum Mode {
        VALID,  // Output size = input_size - kernel_size + 1
        SAME,   // Output size = input_size (with padding)
        FULL    // Output size = input_size + kernel_size - 1
    };
    
private:
    static std::tuple<int, int, int> compute_output_size(
        int input_depth, int input_height, int input_width,
        int kernel_depth, int kernel_height, int kernel_width,
        Mode mode, int stride_d = 1, int stride_h = 1, int stride_w = 1
    ) {
        int out_depth, out_height, out_width;
        
        switch (mode) {
            case VALID:
                out_depth = (input_depth - kernel_depth) / stride_d + 1;
                out_height = (input_height - kernel_height) / stride_h + 1;
                out_width = (input_width - kernel_width) / stride_w + 1;
                break;
            case SAME:
                out_depth = (input_depth + stride_d - 1) / stride_d;
                out_height = (input_height + stride_h - 1) / stride_h;
                out_width = (input_width + stride_w - 1) / stride_w;
                break;
            case FULL:
                out_depth = (input_depth + kernel_depth - 1 + stride_d - 1) / stride_d;
                out_height = (input_height + kernel_height - 1 + stride_h - 1) / stride_h;
                out_width = (input_width + kernel_width - 1 + stride_w - 1) / stride_w;
                break;
            default:
                throw std::invalid_argument("Invalid convolution mode");
        }
        
        return {out_depth, out_height, out_width};
    }
    
    static std::tuple<int, int, int, int, int, int> compute_padding(
        int input_depth, int input_height, int input_width,
        int kernel_depth, int kernel_height, int kernel_width,
        Mode mode, int stride_d = 1, int stride_h = 1, int stride_w = 1
    ) {
        switch (mode) {
            case VALID:
                return {0, 0, 0, 0, 0, 0};
            case SAME: {
                auto [out_depth, out_height, out_width] = compute_output_size(
                    input_depth, input_height, input_width,
                    kernel_depth, kernel_height, kernel_width,
                    SAME, stride_d, stride_h, stride_w
                );
                
                int pad_total_d = std::max(0, (out_depth - 1) * stride_d + kernel_depth - input_depth);
                int pad_total_h = std::max(0, (out_height - 1) * stride_h + kernel_height - input_height);
                int pad_total_w = std::max(0, (out_width - 1) * stride_w + kernel_width - input_width);
                
                int pad_front = pad_total_d / 2;
                int pad_back = pad_total_d - pad_front;
                int pad_top = pad_total_h / 2;
                int pad_bottom = pad_total_h - pad_top;
                int pad_left = pad_total_w / 2;
                int pad_right = pad_total_w - pad_left;
                
                return {pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right};
            }
            case FULL: {
                int pad_d = kernel_depth - 1;
                int pad_h = kernel_height - 1;
                int pad_w = kernel_width - 1;
                return {pad_d, pad_d, pad_h, pad_h, pad_w, pad_w};
            }
            default:
                throw std::invalid_argument("Invalid convolution mode");
        }
    }

public:
    /**
     * Basic 3D convolution
     */
    static Volume3D convolve(const Volume3D& input, const Volume3D& kernel, 
                           Mode mode = VALID, int stride_d = 1, int stride_h = 1, int stride_w = 1) {
        if (stride_d <= 0 || stride_h <= 0 || stride_w <= 0) {
            throw std::invalid_argument("Strides must be positive");
        }
        
        int input_depth = input.depth();
        int input_height = input.height();
        int input_width = input.width();
        int kernel_depth = kernel.depth();
        int kernel_height = kernel.height();
        int kernel_width = kernel.width();
        
        if ((kernel_depth > input_depth || kernel_height > input_height || kernel_width > input_width) 
            && mode == VALID) {
            throw std::invalid_argument("Kernel size cannot be larger than input for valid convolution");
        }
        
        // Compute output size and padding
        auto [output_depth, output_height, output_width] = compute_output_size(
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            mode, stride_d, stride_h, stride_w
        );
        
        auto [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right] = compute_padding(
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            mode, stride_d, stride_h, stride_w
        );
        
        // Create padded input
        Volume3D padded_input = input.pad(pad_front, pad_back, pad_top, pad_bottom, 
                                         pad_left, pad_right, 0.0f);
        Volume3D output(output_depth, output_height, output_width);
        
        // Perform convolution
        for (int out_d = 0; out_d < output_depth; out_d++) {
            for (int out_h = 0; out_h < output_height; out_h++) {
                for (int out_w = 0; out_w < output_width; out_w++) {
                    float sum = 0.0f;
                    int input_start_d = out_d * stride_d;
                    int input_start_h = out_h * stride_h;
                    int input_start_w = out_w * stride_w;
                    
                    for (int kd = 0; kd < kernel_depth; kd++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int input_d = input_start_d + kd;
                                int input_h = input_start_h + kh;
                                int input_w = input_start_w + kw;
                                
                                if (input_d < padded_input.depth() && 
                                    input_h < padded_input.height() && 
                                    input_w < padded_input.width()) {
                                    sum += padded_input(input_d, input_h, input_w) * kernel(kd, kh, kw);
                                }
                            }
                        }
                    }
                    
                    output(out_d, out_h, out_w) = sum;
                }
            }
        }
        
        return output;
    }
    
    /**
     * 3D cross-correlation (commonly used in 3D neural networks)
     */
    static Volume3D cross_correlate(const Volume3D& input, const Volume3D& kernel) {
        int input_depth = input.depth();
        int input_height = input.height();
        int input_width = input.width();
        int kernel_depth = kernel.depth();
        int kernel_height = kernel.height();
        int kernel_width = kernel.width();
        
        int output_depth = input_depth - kernel_depth + 1;
        int output_height = input_height - kernel_height + 1;
        int output_width = input_width - kernel_width + 1;
        
        if (output_depth <= 0 || output_height <= 0 || output_width <= 0) {
            throw std::invalid_argument("Kernel size too large for cross-correlation");
        }
        
        Volume3D output(output_depth, output_height, output_width);
        
        for (int out_d = 0; out_d < output_depth; out_d++) {
            for (int out_h = 0; out_h < output_height; out_h++) {
                for (int out_w = 0; out_w < output_width; out_w++) {
                    float sum = 0.0f;
                    
                    for (int kd = 0; kd < kernel_depth; kd++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int input_d = out_d + kd;
                                int input_h = out_h + kh;
                                int input_w = out_w + kw;
                                
                                // Note: kernel indexing for cross-correlation (no flipping)
                                sum += input(input_d, input_h, input_w) * 
                                      kernel(kernel_depth - 1 - kd, kernel_height - 1 - kh, kernel_width - 1 - kw);
                            }
                        }
                    }
                    
                    output(out_d, out_h, out_w) = sum;
                }
            }
        }
        
        return output;
    }
    
    /**
     * Separable 3D convolution (for efficiency with separable kernels)
     */
    static Volume3D separable_convolve(const Volume3D& input, 
                                     const std::vector<float>& kernel_d,
                                     const std::vector<float>& kernel_h,
                                     const std::vector<float>& kernel_w) {
        int input_depth = input.depth();
        int input_height = input.height();
        int input_width = input.width();
        int kd_size = kernel_d.size();
        int kh_size = kernel_h.size();
        int kw_size = kernel_w.size();
        
        // First pass: convolve along width dimension
        int intermediate1_width = input_width - kw_size + 1;
        Volume3D intermediate1(input_depth, input_height, intermediate1_width);
        
        for (int d = 0; d < input_depth; d++) {
            for (int h = 0; h < input_height; h++) {
                for (int w = 0; w < intermediate1_width; w++) {
                    float sum = 0.0f;
                    for (int k = 0; k < kw_size; k++) {
                        sum += input(d, h, w + k) * kernel_w[k];
                    }
                    intermediate1(d, h, w) = sum;
                }
            }
        }
        
        // Second pass: convolve along height dimension
        int intermediate2_height = input_height - kh_size + 1;
        Volume3D intermediate2(input_depth, intermediate2_height, intermediate1_width);
        
        for (int d = 0; d < input_depth; d++) {
            for (int h = 0; h < intermediate2_height; h++) {
                for (int w = 0; w < intermediate1_width; w++) {
                    float sum = 0.0f;
                    for (int k = 0; k < kh_size; k++) {
                        sum += intermediate1(d, h + k, w) * kernel_h[k];
                    }
                    intermediate2(d, h, w) = sum;
                }
            }
        }
        
        // Third pass: convolve along depth dimension
        int output_depth = input_depth - kd_size + 1;
        Volume3D output(output_depth, intermediate2_height, intermediate1_width);
        
        for (int d = 0; d < output_depth; d++) {
            for (int h = 0; h < intermediate2_height; h++) {
                for (int w = 0; w < intermediate1_width; w++) {
                    float sum = 0.0f;
                    for (int k = 0; k < kd_size; k++) {
                        sum += intermediate2(d + k, h, w) * kernel_d[k];
                    }
                    output(d, h, w) = sum;
                }
            }
        }
        
        return output;
    }
};

/**
 * Utility functions for testing and validation
 */
namespace ConvolutionUtils3D {
    /**
     * Compare two volumes with tolerance
     */
    bool volumes_equal(const Volume3D& a, const Volume3D& b, float tolerance = 1e-6f) {
        if (a.depth() != b.depth() || a.height() != b.height() || a.width() != b.width()) {
            return false;
        }
        
        for (int d = 0; d < a.depth(); d++) {
            for (int h = 0; h < a.height(); h++) {
                for (int w = 0; w < a.width(); w++) {
                    if (std::abs(a(d, h, w) - b(d, h, w)) > tolerance) {
                        return false;
                    }
                }
            }
        }
        
        return true;
    }
    
    /**
     * Compute maximum absolute difference between volumes
     */
    float max_difference(const Volume3D& a, const Volume3D& b) {
        if (a.depth() != b.depth() || a.height() != b.height() || a.width() != b.width()) {
            return INFINITY;
        }
        
        float max_diff = 0.0f;
        for (int d = 0; d < a.depth(); d++) {
            for (int h = 0; h < a.height(); h++) {
                for (int w = 0; w < a.width(); w++) {
                    max_diff = std::max(max_diff, std::abs(a(d, h, w) - b(d, h, w)));
                }
            }
        }
        
        return max_diff;
    }
    
    /**
     * Print volume values (limited output)
     */
    void print_volume(const Volume3D& volume, const std::string& name = "", 
                     int max_slices = 3, int max_rows = 3, int max_cols = 3) {
        if (!name.empty()) {
            std::cout << name << " (" << volume.depth() << "x" << volume.height() 
                      << "x" << volume.width() << "):" << std::endl;
        }
        
        int print_slices = std::min(volume.depth(), max_slices);
        int print_rows = std::min(volume.height(), max_rows);
        int print_cols = std::min(volume.width(), max_cols);
        
        for (int d = 0; d < print_slices; d++) {
            std::cout << "Slice " << d << ":" << std::endl;
            for (int h = 0; h < print_rows; h++) {
                for (int w = 0; w < print_cols; w++) {
                    std::cout << std::fixed << std::setprecision(3) << volume(d, h, w) << " ";
                }
                if (volume.width() > max_cols) {
                    std::cout << "...";
                }
                std::cout << std::endl;
            }
            if (volume.height() > max_rows) {
                std::cout << "..." << std::endl;
            }
            std::cout << std::endl;
        }
        
        if (volume.depth() > max_slices) {
            std::cout << "..." << std::endl;
        }
    }
    
    /**
     * Generate test volumes and kernels
     */
    Volume3D generate_impulse_3d(int depth, int height, int width, 
                                int pos_d = -1, int pos_h = -1, int pos_w = -1, 
                                float amplitude = 1.0f) {
        Volume3D volume(depth, height, width);
        if (pos_d < 0) pos_d = depth / 2;
        if (pos_h < 0) pos_h = height / 2;
        if (pos_w < 0) pos_w = width / 2;
        
        if (pos_d >= 0 && pos_d < depth && pos_h >= 0 && pos_h < height && 
            pos_w >= 0 && pos_w < width) {
            volume(pos_d, pos_h, pos_w) = amplitude;
        }
        
        return volume;
    }
    
    Volume3D generate_gaussian_kernel_3d(int size, float sigma = 1.0f) {
        Volume3D kernel(size, size, size);
        int center = size / 2;
        float sum = 0.0f;
        
        for (int d = 0; d < size; d++) {
            for (int h = 0; h < size; h++) {
                for (int w = 0; w < size; w++) {
                    float x = static_cast<float>(d - center);
                    float y = static_cast<float>(h - center);
                    float z = static_cast<float>(w - center);
                    float val = std::exp(-(x * x + y * y + z * z) / (2.0f * sigma * sigma));
                    kernel(d, h, w) = val;
                    sum += val;
                }
            }
        }
        
        // Normalize
        for (int d = 0; d < size; d++) {
            for (int h = 0; h < size; h++) {
                for (int w = 0; w < size; w++) {
                    kernel(d, h, w) /= sum;
                }
            }
        }
        
        return kernel;
    }
}

/**
 * C-style interface functions matching CUDA/GPU API
 */
int convolution_3d_reference(
    const float* input_data,
    const float* kernel_data,
    float* output_data,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int mode = 0  // 0=valid, 1=same, 2=full
) {
    try {
        Volume3D input(input_data, input_depth, input_height, input_width);
        Volume3D kernel(kernel_data, kernel_depth, kernel_height, kernel_width);
        
        Convolution3D::Mode conv_mode;
        switch (mode) {
            case 0: conv_mode = Convolution3D::VALID; break;
            case 1: conv_mode = Convolution3D::SAME; break;
            case 2: conv_mode = Convolution3D::FULL; break;
            default: return -1; // Invalid mode
        }
        
        Volume3D output = Convolution3D::convolve(input, kernel, conv_mode);
        
        // Copy result to output buffer
        int output_size = output.depth() * output.height() * output.width();
        std::copy(output.data(), output.data() + output_size, output_data);
        
        return output_size; // Return output size
    }
    catch (const std::exception& e) {
        return -1; // Error
    }
}

int cross_correlation_3d_reference(
    const float* input_data,
    const float* kernel_data,
    float* output_data,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width
) {
    try {
        Volume3D input(input_data, input_depth, input_height, input_width);
        Volume3D kernel(kernel_data, kernel_depth, kernel_height, kernel_width);
        
        Volume3D output = Convolution3D::cross_correlate(input, kernel);
        
        // Copy result to output buffer
        int output_size = output.depth() * output.height() * output.width();
        std::copy(output.data(), output.data() + output_size, output_data);
        
        return output_size; // Return output size
    }
    catch (const std::exception& e) {
        return -1; // Error
    }
}

/**
 * Test functions
 */
void test_convolution_3d_reference() {
    std::cout << "Testing 3D Convolution Reference Implementation..." << std::endl;
    
    // Test 1: Impulse response
    Volume3D impulse = ConvolutionUtils3D::generate_impulse_3d(8, 8, 8, 4, 4, 4, 1.0f);
    Volume3D gaussian = ConvolutionUtils3D::generate_gaussian_kernel_3d(3, 1.0f);
    
    Volume3D result = Convolution3D::convolve(impulse, gaussian, Convolution3D::VALID);
    ConvolutionUtils3D::print_volume(result, "Impulse response");
    
    // Test 2: Different modes
    Volume3D small_input(4, 4, 4);
    small_input.randomize();
    Volume3D small_kernel(2, 2, 2);
    small_kernel.randomize();
    
    Volume3D valid_result = Convolution3D::convolve(small_input, small_kernel, Convolution3D::VALID);
    Volume3D same_result = Convolution3D::convolve(small_input, small_kernel, Convolution3D::SAME);
    Volume3D full_result = Convolution3D::convolve(small_input, small_kernel, Convolution3D::FULL);
    
    std::cout << "Mode sizes - Valid: " << valid_result.depth() << "x" << valid_result.height() << "x" << valid_result.width()
              << ", Same: " << same_result.depth() << "x" << same_result.height() << "x" << same_result.width()
              << ", Full: " << full_result.depth() << "x" << full_result.height() << "x" << full_result.width() << std::endl;
    
    // Test 3: Cross-correlation vs convolution
    Volume3D test_input(5, 5, 5);
    test_input.randomize();
    Volume3D test_kernel(3, 3, 3);
    test_kernel.randomize();
    
    Volume3D conv_result = Convolution3D::convolve(test_input, test_kernel, Convolution3D::VALID);
    Volume3D xcorr_result = Convolution3D::cross_correlate(test_input, test_kernel);
    
    std::cout << "Convolution result size: " << conv_result.depth() << "x" << conv_result.height() << "x" << conv_result.width() << std::endl;
    std::cout << "Cross-correlation result size: " << xcorr_result.depth() << "x" << xcorr_result.height() << "x" << xcorr_result.width() << std::endl;
    
    std::cout << "All 3D convolution tests completed!" << std::endl;
}

#ifdef CONVOLUTION_3D_MAIN
int main() {
    test_convolution_3d_reference();
    return 0;
}
#endif 