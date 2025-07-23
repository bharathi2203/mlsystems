/*
 * Logistic Regression - C++ Reference Implementation
 * 
 * Simple, single-threaded CPU implementation of logistic regression
 * for binary and multi-class classification tasks.
 * 
 * This reference implementation serves as:
 * - Educational baseline for understanding the algorithm
 * - Correctness validation for GPU implementations
 * - Performance comparison baseline
 * - Debugging reference when GPU results seem incorrect
 * 
 * Mathematical foundation:
 * - Binary classification: P(y=1|x) = σ(w^T x + b) where σ is sigmoid
 * - Multi-class: P(y=k|x) = softmax(W^T x + b)_k
 * - Loss function: Cross-entropy L = -Σ y_i log(p_i) + (1-y_i) log(1-p_i)
 * - Gradient: ∇w = X^T (p - y) / n + λw (with regularization)
 * 
 * Advantages:
 * - Simple and easy to understand
 * - Straightforward debugging
 * - No GPU memory management complexity
 * - Standard math library functions
 * 
 * Disadvantages:
 * - Single-threaded, slow for large datasets
 * - No GPU acceleration
 * - Limited scalability
 * - Sequential gradient computation
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
#include <limits>

/**
 * Numerically stable sigmoid function
 * Uses the identity: sigmoid(x) = 1/(1+exp(-x)) = exp(x)/(1+exp(x))
 */
float sigmoid(float x) {
    if (x >= 0) {
        float exp_neg_x = std::exp(-x);
        return 1.0f / (1.0f + exp_neg_x);
    } else {
        float exp_x = std::exp(x);
        return exp_x / (1.0f + exp_x);
    }
}

/**
 * Numerically stable log-sigmoid function
 */
float log_sigmoid(float x) {
    if (x >= 0) {
        return -std::log(1.0f + std::exp(-x));
    } else {
        return x - std::log(1.0f + std::exp(x));
    }
}

/**
 * Softmax function with LogSumExp trick for numerical stability
 */
std::vector<float> softmax(const std::vector<float>& logits) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    std::vector<float> exp_logits(logits.size());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < logits.size(); i++) {
        exp_logits[i] = std::exp(logits[i] - max_logit);
        sum_exp += exp_logits[i];
    }
    
    for (size_t i = 0; i < logits.size(); i++) {
        exp_logits[i] /= sum_exp;
    }
    
    return exp_logits;
}

/**
 * Binary logistic regression class
 */
class BinaryLogisticRegression {
public:
    std::vector<float> weights;
    float bias;
    float learning_rate;
    float lambda;
    
    BinaryLogisticRegression(int n_features, float lr = 0.01f, float reg = 0.0f) 
        : weights(n_features, 0.0f), bias(0.0f), learning_rate(lr), lambda(reg) {}
    
    /**
     * Compute prediction for a single sample
     */
    float predict_sample(const std::vector<float>& x) const {
        float linear_output = bias;
        for (size_t i = 0; i < weights.size(); i++) {
            linear_output += weights[i] * x[i];
        }
        return sigmoid(linear_output);
    }
    
    /**
     * Compute predictions for all samples
     */
    std::vector<float> predict(const std::vector<std::vector<float>>& X) const {
        std::vector<float> predictions;
        predictions.reserve(X.size());
        
        for (const auto& sample : X) {
            predictions.push_back(predict_sample(sample));
        }
        
        return predictions;
    }
    
    /**
     * Compute binary cross-entropy loss
     */
    float compute_loss(const std::vector<std::vector<float>>& X, 
                      const std::vector<float>& y) const {
        std::vector<float> predictions = predict(X);
        float loss = 0.0f;
        
        for (size_t i = 0; i < y.size(); i++) {
            float p = std::max(std::min(predictions[i], 1.0f - 1e-7f), 1e-7f); // Clip for stability
            loss += -(y[i] * std::log(p) + (1.0f - y[i]) * std::log(1.0f - p));
        }
        
        // Average loss
        loss /= y.size();
        
        // Add regularization term
        if (lambda > 0.0f) {
            float reg_term = 0.0f;
            for (float w : weights) {
                reg_term += w * w;
            }
            loss += 0.5f * lambda * reg_term;
        }
        
        return loss;
    }
    
    /**
     * Compute gradients
     */
    void compute_gradients(const std::vector<std::vector<float>>& X,
                          const std::vector<float>& y,
                          std::vector<float>& grad_w,
                          float& grad_b) const {
        std::vector<float> predictions = predict(X);
        
        // Initialize gradients
        std::fill(grad_w.begin(), grad_w.end(), 0.0f);
        grad_b = 0.0f;
        
        // Compute gradients: X^T * (p - y)
        for (size_t i = 0; i < X.size(); i++) {
            float error = predictions[i] - y[i];
            
            for (size_t j = 0; j < weights.size(); j++) {
                grad_w[j] += X[i][j] * error;
            }
            grad_b += error;
        }
        
        // Average and add regularization
        for (size_t j = 0; j < weights.size(); j++) {
            grad_w[j] = grad_w[j] / X.size() + lambda * weights[j];
        }
        grad_b /= X.size();
    }
    
    /**
     * Train using gradient descent
     */
    int train(const std::vector<std::vector<float>>& X,
             const std::vector<float>& y,
             int max_iterations = 1000,
             float tolerance = 1e-6f,
             bool verbose = false) {
        
        std::vector<float> grad_w(weights.size());
        float grad_b;
        float prev_loss = std::numeric_limits<float>::max();
        
        if (verbose) {
            std::cout << "Training binary logistic regression..." << std::endl;
            std::cout << std::fixed << std::setprecision(6);
        }
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute gradients
            compute_gradients(X, y, grad_w, grad_b);
            
            // Update parameters
            for (size_t j = 0; j < weights.size(); j++) {
                weights[j] -= learning_rate * grad_w[j];
            }
            bias -= learning_rate * grad_b;
            
            // Check convergence every 10 iterations
            if (iter % 10 == 0) {
                float current_loss = compute_loss(X, y);
                
                if (verbose) {
                    std::cout << "Iteration " << iter 
                              << ", Loss: " << current_loss << std::endl;
                }
                
                if (std::abs(current_loss - prev_loss) < tolerance) {
                    if (verbose) {
                        std::cout << "Converged after " << iter << " iterations." << std::endl;
                    }
                    return iter;
                }
                prev_loss = current_loss;
            }
        }
        
        return max_iterations;
    }
    
    /**
     * Compute accuracy on test data
     */
    float accuracy(const std::vector<std::vector<float>>& X, 
                  const std::vector<float>& y) const {
        std::vector<float> predictions = predict(X);
        int correct = 0;
        
        for (size_t i = 0; i < y.size(); i++) {
            int predicted_class = predictions[i] >= 0.5f ? 1 : 0;
            int true_class = static_cast<int>(y[i]);
            if (predicted_class == true_class) {
                correct++;
            }
        }
        
        return static_cast<float>(correct) / y.size();
    }
};

/**
 * Multi-class logistic regression class
 */
class MultiClassLogisticRegression {
public:
    std::vector<std::vector<float>> weights; // [n_features x n_classes]
    std::vector<float> bias; // [n_classes]
    float learning_rate;
    float lambda;
    int n_features;
    int n_classes;
    
    MultiClassLogisticRegression(int n_feat, int n_cls, float lr = 0.01f, float reg = 0.0f)
        : weights(n_feat, std::vector<float>(n_cls, 0.0f)),
          bias(n_cls, 0.0f),
          learning_rate(lr),
          lambda(reg),
          n_features(n_feat),
          n_classes(n_cls) {}
    
    /**
     * Compute prediction for a single sample
     */
    std::vector<float> predict_sample(const std::vector<float>& x) const {
        std::vector<float> logits(n_classes, 0.0f);
        
        for (int c = 0; c < n_classes; c++) {
            logits[c] = bias[c];
            for (int f = 0; f < n_features; f++) {
                logits[c] += weights[f][c] * x[f];
            }
        }
        
        return softmax(logits);
    }
    
    /**
     * Compute predictions for all samples
     */
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& X) const {
        std::vector<std::vector<float>> predictions;
        predictions.reserve(X.size());
        
        for (const auto& sample : X) {
            predictions.push_back(predict_sample(sample));
        }
        
        return predictions;
    }
    
    /**
     * Compute categorical cross-entropy loss
     */
    float compute_loss(const std::vector<std::vector<float>>& X,
                      const std::vector<std::vector<float>>& Y) const {
        auto predictions = predict(X);
        float loss = 0.0f;
        
        for (size_t i = 0; i < Y.size(); i++) {
            for (int c = 0; c < n_classes; c++) {
                float p = std::max(predictions[i][c], 1e-7f); // Clip for stability
                loss += Y[i][c] * std::log(p);
            }
        }
        
        loss = -loss / Y.size();
        
        // Add regularization term
        if (lambda > 0.0f) {
            float reg_term = 0.0f;
            for (int f = 0; f < n_features; f++) {
                for (int c = 0; c < n_classes; c++) {
                    reg_term += weights[f][c] * weights[f][c];
                }
            }
            loss += 0.5f * lambda * reg_term;
        }
        
        return loss;
    }
    
    /**
     * Compute gradients
     */
    void compute_gradients(const std::vector<std::vector<float>>& X,
                          const std::vector<std::vector<float>>& Y,
                          std::vector<std::vector<float>>& grad_W,
                          std::vector<float>& grad_b) const {
        auto predictions = predict(X);
        
        // Initialize gradients
        for (int f = 0; f < n_features; f++) {
            std::fill(grad_W[f].begin(), grad_W[f].end(), 0.0f);
        }
        std::fill(grad_b.begin(), grad_b.end(), 0.0f);
        
        // Compute gradients: X^T * (P - Y)
        for (size_t i = 0; i < X.size(); i++) {
            for (int c = 0; c < n_classes; c++) {
                float error = predictions[i][c] - Y[i][c];
                
                for (int f = 0; f < n_features; f++) {
                    grad_W[f][c] += X[i][f] * error;
                }
                grad_b[c] += error;
            }
        }
        
        // Average and add regularization
        for (int f = 0; f < n_features; f++) {
            for (int c = 0; c < n_classes; c++) {
                grad_W[f][c] = grad_W[f][c] / X.size() + lambda * weights[f][c];
            }
        }
        
        for (int c = 0; c < n_classes; c++) {
            grad_b[c] /= X.size();
        }
    }
    
    /**
     * Train using gradient descent
     */
    int train(const std::vector<std::vector<float>>& X,
             const std::vector<std::vector<float>>& Y,
             int max_iterations = 1000,
             float tolerance = 1e-6f,
             bool verbose = false) {
        
        std::vector<std::vector<float>> grad_W(n_features, std::vector<float>(n_classes));
        std::vector<float> grad_b(n_classes);
        float prev_loss = std::numeric_limits<float>::max();
        
        if (verbose) {
            std::cout << "Training multi-class logistic regression..." << std::endl;
            std::cout << std::fixed << std::setprecision(6);
        }
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute gradients
            compute_gradients(X, Y, grad_W, grad_b);
            
            // Update parameters
            for (int f = 0; f < n_features; f++) {
                for (int c = 0; c < n_classes; c++) {
                    weights[f][c] -= learning_rate * grad_W[f][c];
                }
            }
            
            for (int c = 0; c < n_classes; c++) {
                bias[c] -= learning_rate * grad_b[c];
            }
            
            // Check convergence every 10 iterations
            if (iter % 10 == 0) {
                float current_loss = compute_loss(X, Y);
                
                if (verbose) {
                    std::cout << "Iteration " << iter 
                              << ", Loss: " << current_loss << std::endl;
                }
                
                if (std::abs(current_loss - prev_loss) < tolerance) {
                    if (verbose) {
                        std::cout << "Converged after " << iter << " iterations." << std::endl;
                    }
                    return iter;
                }
                prev_loss = current_loss;
            }
        }
        
        return max_iterations;
    }
    
    /**
     * Compute accuracy on test data
     */
    float accuracy(const std::vector<std::vector<float>>& X,
                  const std::vector<std::vector<float>>& Y) const {
        auto predictions = predict(X);
        int correct = 0;
        
        for (size_t i = 0; i < Y.size(); i++) {
            // Find predicted class (highest probability)
            int predicted_class = 0;
            float max_prob = predictions[i][0];
            for (int c = 1; c < n_classes; c++) {
                if (predictions[i][c] > max_prob) {
                    max_prob = predictions[i][c];
                    predicted_class = c;
                }
            }
            
            // Find true class (one-hot encoded)
            int true_class = 0;
            for (int c = 0; c < n_classes; c++) {
                if (Y[i][c] == 1.0f) {
                    true_class = c;
                    break;
                }
            }
            
            if (predicted_class == true_class) {
                correct++;
            }
        }
        
        return static_cast<float>(correct) / Y.size();
    }
};

/**
 * C-style interface functions matching CUDA API
 */
int train_binary_logistic_regression_reference(
    const float* X_data,
    const float* y_data,
    float* w_data,
    float* b_data,
    int n_samples,
    int n_features,
    float learning_rate = 0.01f,
    float lambda = 0.0f,
    int max_iterations = 1000,
    float tolerance = 1e-6f
) {
    // Convert to C++ format
    std::vector<std::vector<float>> X(n_samples, std::vector<float>(n_features));
    std::vector<float> y(n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X[i][j] = X_data[i * n_features + j];
        }
        y[i] = y_data[i];
    }
    
    // Create and train model
    BinaryLogisticRegression model(n_features, learning_rate, lambda);
    
    // Initialize with provided weights and bias
    for (int i = 0; i < n_features; i++) {
        model.weights[i] = w_data[i];
    }
    model.bias = *b_data;
    
    int iterations = model.train(X, y, max_iterations, tolerance, false);
    
    // Copy results back
    for (int i = 0; i < n_features; i++) {
        w_data[i] = model.weights[i];
    }
    *b_data = model.bias;
    
    return iterations;
}

int train_multiclass_logistic_regression_reference(
    const float* X_data,
    const float* Y_data,
    float* W_data,
    float* b_data,
    int n_samples,
    int n_features,
    int n_classes,
    float learning_rate = 0.01f,
    float lambda = 0.0f,
    int max_iterations = 1000,
    float tolerance = 1e-6f
) {
    // Convert to C++ format
    std::vector<std::vector<float>> X(n_samples, std::vector<float>(n_features));
    std::vector<std::vector<float>> Y(n_samples, std::vector<float>(n_classes));
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X[i][j] = X_data[i * n_features + j];
        }
        for (int c = 0; c < n_classes; c++) {
            Y[i][c] = Y_data[i * n_classes + c];
        }
    }
    
    // Create and train model
    MultiClassLogisticRegression model(n_features, n_classes, learning_rate, lambda);
    
    // Initialize with provided weights and bias
    for (int f = 0; f < n_features; f++) {
        for (int c = 0; c < n_classes; c++) {
            model.weights[f][c] = W_data[f * n_classes + c];
        }
    }
    for (int c = 0; c < n_classes; c++) {
        model.bias[c] = b_data[c];
    }
    
    int iterations = model.train(X, Y, max_iterations, tolerance, false);
    
    // Copy results back
    for (int f = 0; f < n_features; f++) {
        for (int c = 0; c < n_classes; c++) {
            W_data[f * n_classes + c] = model.weights[f][c];
        }
    }
    for (int c = 0; c < n_classes; c++) {
        b_data[c] = model.bias[c];
    }
    
    return iterations;
}

/**
 * Utility function: Generate synthetic classification data for testing
 */
void generate_synthetic_binary_data(
    float* X,
    float* y,
    const float* true_w,
    float true_b,
    int n_samples,
    int n_features,
    float noise_level = 0.1f
) {
    // Initialize random seed
    srand(42);
    
    // Generate random feature matrix
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            X[i * n_features + j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
    
    // Generate labels using true parameters + noise
    for (int i = 0; i < n_samples; i++) {
        float linear_output = true_b;
        for (int j = 0; j < n_features; j++) {
            linear_output += X[i * n_features + j] * true_w[j];
        }
        
        // Add noise
        linear_output += noise_level * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        
        // Apply sigmoid and threshold
        float prob = sigmoid(linear_output);
        y[i] = (prob >= 0.5f) ? 1.0f : 0.0f;
    }
} 