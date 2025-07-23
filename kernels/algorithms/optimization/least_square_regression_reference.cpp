/*
 * Least Squares Regression - C++ Reference Implementation
 * 
 * Simple, single-threaded CPU implementation of linear least squares regression
 * for educational purposes and correctness validation.
 * 
 * This reference implementation serves as:
 * - Educational baseline for understanding the algorithm
 * - Correctness validation for GPU implementations
 * - Performance comparison baseline
 * - Debugging reference when GPU results seem incorrect
 * 
 * Mathematical foundation:
 * - Solves: min ||Ax - b||² where A is feature matrix, x is parameters, b is targets
 * - Normal equation: x = (A^T A)^(-1) A^T b
 * - Gradient descent: x_new = x_old - α * ∇f(x)
 * - Regularized version: min ||Ax - b||² + λ||x||² (Ridge regression)
 * 
 * Advantages:
 * - Simple and easy to understand
 * - Straightforward debugging
 * - No GPU memory management complexity
 * - Standard linear algebra libraries
 * 
 * Disadvantages:
 * - Single-threaded, slow for large datasets
 * - No GPU acceleration
 * - Limited scalability
 * - Memory intensive for large feature matrices
 * 
 * Applications:
 * - Small to medium datasets
 * - Educational purposes
 * - Algorithm validation
 * - Prototyping before GPU implementation
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

/**
 * Matrix class for simple linear algebra operations
 */
class Matrix {
public:
    std::vector<float> data;
    int rows, cols;
    
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
    
    float& operator()(int i, int j) { return data[i * cols + j]; }
    const float& operator()(int i, int j) const { return data[i * cols + j]; }
    
    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }
    
    // Matrix transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    // Add regularization to diagonal
    void add_diagonal(float value) {
        for (int i = 0; i < std::min(rows, cols); i++) {
            (*this)(i, i) += value;
        }
    }
};

/**
 * Vector class for simple vector operations
 */
class Vector {
public:
    std::vector<float> data;
    int size;
    
    Vector(int s) : size(s), data(s, 0.0f) {}
    Vector(const std::vector<float>& d) : data(d), size(d.size()) {}
    
    float& operator[](int i) { return data[i]; }
    const float& operator[](int i) const { return data[i]; }
    
    // Vector subtraction
    Vector operator-(const Vector& other) const {
        Vector result(size);
        for (int i = 0; i < size; i++) {
            result[i] = data[i] - other[i];
        }
        return result;
    }
    
    // Scalar multiplication
    Vector operator*(float scalar) const {
        Vector result(size);
        for (int i = 0; i < size; i++) {
            result[i] = data[i] * scalar;
        }
        return result;
    }
    
    // Dot product
    float dot(const Vector& other) const {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += data[i] * other[i];
        }
        return sum;
    }
    
    // L2 norm
    float norm() const {
        return std::sqrt(dot(*this));
    }
    
    // L2 norm squared
    float norm_squared() const {
        return dot(*this);
    }
};

/**
 * Matrix-vector multiplication
 */
Vector matrix_vector_multiply(const Matrix& A, const Vector& x) {
    Vector result(A.rows);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            result[i] += A(i, j) * x[j];
        }
    }
    return result;
}

/**
 * Solve linear system using Gaussian elimination with partial pivoting
 * Solves Ax = b for x
 */
Vector solve_linear_system(Matrix A, Vector b) {
    int n = A.rows;
    
    // Forward elimination with partial pivoting
    for (int i = 0; i < n; i++) {
        // Find pivot
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A(k, i)) > std::abs(A(max_row, i))) {
                max_row = k;
            }
        }
        
        // Swap rows
        if (max_row != i) {
            for (int j = 0; j < n; j++) {
                std::swap(A(i, j), A(max_row, j));
            }
            std::swap(b[i], b[max_row]);
        }
        
        // Make all rows below this one 0 in current column
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A(i, i)) > 1e-10f) {
                float factor = A(k, i) / A(i, i);
                for (int j = i; j < n; j++) {
                    A(k, j) -= factor * A(i, j);
                }
                b[k] -= factor * b[i];
            }
        }
    }
    
    // Back substitution
    Vector x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A(i, j) * x[j];
        }
        if (std::abs(A(i, i)) > 1e-10f) {
            x[i] /= A(i, i);
        }
    }
    
    return x;
}

/**
 * Solve least squares using normal equations
 * Solves (A^T A + λI) x = A^T b
 * 
 * @param A: Feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param lambda: Regularization parameter
 * @return: Solution vector x [n_features]
 */
Vector solve_normal_equations_reference(
    const Matrix& A,
    const Vector& b,
    float lambda = 0.0f
) {
    // Compute A^T
    Matrix AT = A.transpose();
    
    // Compute A^T * A
    Matrix ATA = AT * A;
    
    // Add regularization
    if (lambda > 0.0f) {
        ATA.add_diagonal(lambda);
    }
    
    // Compute A^T * b
    Vector ATb = matrix_vector_multiply(AT, b);
    
    // Solve linear system
    return solve_linear_system(ATA, ATb);
}

/**
 * Compute gradient for least squares
 * gradient = A^T * (A*x - b) / n_samples + lambda * x
 */
Vector compute_gradient(
    const Matrix& A,
    const Vector& b,
    const Vector& x,
    float lambda = 0.0f
) {
    // Compute predictions
    Vector predictions = matrix_vector_multiply(A, x);
    
    // Compute residual
    Vector residual = predictions - b;
    
    // Compute A^T * residual
    Matrix AT = A.transpose();
    Vector gradient = matrix_vector_multiply(AT, residual);
    
    // Scale by number of samples and add regularization
    for (int i = 0; i < gradient.size; i++) {
        gradient[i] = gradient[i] / A.rows + lambda * x[i];
    }
    
    return gradient;
}

/**
 * Compute loss function (Mean Squared Error + Regularization)
 */
float compute_loss(
    const Matrix& A,
    const Vector& b,
    const Vector& x,
    float lambda = 0.0f
) {
    // Compute predictions
    Vector predictions = matrix_vector_multiply(A, x);
    
    // Compute residual
    Vector residual = predictions - b;
    
    // Compute MSE
    float mse = residual.norm_squared() / A.rows;
    
    // Add regularization term
    float reg_term = lambda * x.norm_squared();
    
    return mse + reg_term;
}

/**
 * Solve least squares using gradient descent
 * 
 * @param A: Feature matrix [n_samples x n_features]
 * @param b: Target vector [n_samples]
 * @param x_init: Initial solution guess [n_features]
 * @param learning_rate: Step size for gradient descent
 * @param lambda: Regularization parameter
 * @param max_iterations: Maximum number of iterations
 * @param tolerance: Convergence tolerance
 * @return: Solution vector x [n_features]
 */
Vector solve_gradient_descent_reference(
    const Matrix& A,
    const Vector& b,
    const Vector& x_init,
    float learning_rate = 0.01f,
    float lambda = 0.0f,
    int max_iterations = 1000,
    float tolerance = 1e-6f
) {
    Vector x = x_init;
    float prev_loss = std::numeric_limits<float>::max();
    
    std::cout << "Starting gradient descent optimization..." << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute gradient
        Vector gradient = compute_gradient(A, b, x, lambda);
        
        // Update parameters
        for (int i = 0; i < x.size; i++) {
            x[i] -= learning_rate * gradient[i];
        }
        
        // Check convergence every 10 iterations
        if (iter % 10 == 0) {
            float current_loss = compute_loss(A, b, x, lambda);
            
            std::cout << "Iteration " << iter 
                      << ", Loss: " << current_loss 
                      << ", Gradient norm: " << gradient.norm() << std::endl;
            
            if (std::abs(current_loss - prev_loss) < tolerance) {
                std::cout << "Converged after " << iter << " iterations." << std::endl;
                break;
            }
            prev_loss = current_loss;
        }
    }
    
    return x;
}

/**
 * Main interface functions matching CUDA API
 */
int solve_normal_equations_reference(
    const float* A_data,
    const float* b_data,
    float* x_data,
    int n_samples,
    int n_features,
    float lambda = 0.0f
) {
    // Convert to Matrix/Vector format
    Matrix A(n_samples, n_features);
    Vector b(n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            A(i, j) = A_data[i * n_features + j];
        }
        b[i] = b_data[i];
    }
    
    // Solve using normal equations
    Vector x = solve_normal_equations_reference(A, b, lambda);
    
    // Copy result back
    for (int i = 0; i < n_features; i++) {
        x_data[i] = x[i];
    }
    
    return 0;
}

int solve_gradient_descent_reference(
    const float* A_data,
    const float* b_data,
    float* x_data,
    int n_samples,
    int n_features,
    float learning_rate = 0.01f,
    float lambda = 0.0f,
    int max_iterations = 1000,
    float tolerance = 1e-6f
) {
    // Convert to Matrix/Vector format
    Matrix A(n_samples, n_features);
    Vector b(n_samples);
    Vector x_init(n_features);
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            A(i, j) = A_data[i * n_features + j];
        }
        b[i] = b_data[i];
    }
    
    for (int i = 0; i < n_features; i++) {
        x_init[i] = x_data[i];
    }
    
    // Solve using gradient descent
    Vector x = solve_gradient_descent_reference(A, b, x_init, learning_rate, 
                                              lambda, max_iterations, tolerance);
    
    // Copy result back
    for (int i = 0; i < n_features; i++) {
        x_data[i] = x[i];
    }
    
    return max_iterations; // Return max iterations for consistency with CUDA API
}

/**
 * Utility function: Generate synthetic regression data for testing
 */
void generate_synthetic_data(
    float* A,
    float* b,
    const float* true_x,
    int n_samples,
    int n_features,
    float noise_level = 0.1f
) {
    // Initialize random seed
    srand(42);
    
    // Generate random feature matrix
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            A[i * n_features + j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
    
    // Generate targets using true parameters + noise
    for (int i = 0; i < n_samples; i++) {
        b[i] = 0.0f;
        for (int j = 0; j < n_features; j++) {
            b[i] += A[i * n_features + j] * true_x[j];
        }
        // Add noise
        b[i] += noise_level * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

/**
 * Utility function: Print solution statistics
 */
void print_solution_stats(
    const float* A,
    const float* b,
    const float* x,
    int n_samples,
    int n_features,
    const char* method_name
) {
    Matrix A_mat(n_samples, n_features);
    Vector b_vec(n_samples);
    Vector x_vec(n_features);
    
    // Convert arrays to Matrix/Vector
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            A_mat(i, j) = A[i * n_features + j];
        }
        b_vec[i] = b[i];
    }
    
    for (int i = 0; i < n_features; i++) {
        x_vec[i] = x[i];
    }
    
    // Compute final loss
    float final_loss = compute_loss(A_mat, b_vec, x_vec);
    
    std::cout << "\n" << method_name << " Solution Statistics:" << std::endl;
    std::cout << "Final loss: " << final_loss << std::endl;
    std::cout << "Solution norm: " << x_vec.norm() << std::endl;
    std::cout << "Solution: [";
    for (int i = 0; i < std::min(5, n_features); i++) {
        std::cout << x[i];
        if (i < std::min(5, n_features) - 1) std::cout << ", ";
    }
    if (n_features > 5) std::cout << ", ...";
    std::cout << "]" << std::endl;
} 