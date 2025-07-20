# Model Compression Notes

## Overview

Model compression is essential for deploying NLP systems efficiently and equitably. While training large models is expensive, **inference costs often exceed training costs within just one week of deployment**. For models used over months or years, inference costs greatly eclipse the one-time training costs.

The main question: **How can we cheaply, efficiently, and equitably deploy NLP systems without sacrificing performance?**

## Why Model Compression Works

### Overparameterization
- Models have more parameters than training data or statistical ML theory suggests
- GPT-3 had 170 billion parameters - definitely overparameterized
- Overparameterized models are easier to train for complex tasks
- Many parameters help sidestep saddle points and local optima during optimization
- **Key insight**: You don't need all parameters for inference - they're primarily a training-time trick

## Three Main Compression Techniques

### 1. Quantization

**Definition**: Reduce precision of model weights without changing architecture or parameters.

#### Basic Quantization Concepts

**Post-Training Quantization**:
- Train model at full precision, then reduce precision of weights
- Example: 65B parameter model with 32-bit floats = 260GB
- With 4-bit precision = 32GB (8x reduction)
- With 1-bit (binary) = 8GB (32x reduction)

**Floating Point Representation**:
- **Sign bit**: Positive/negative
- **Fractional bits**: Range of values  
- **Exponent bits**: Scaling factor
- **Float16**: 10 fraction bits, 5 exponent bits
- **BFloat16**: Designed for ML, larger range but fewer precision choices

#### Quantization Methods

**Absolute Maximum (AbsMax) Quantization**:
- Map floats to integer range (e.g., -127 to 127 for int8)
- Find largest absolute value, scale everything proportionally
- Example: If max value is 20, 20 becomes 127, 0.5 becomes 3

**Binary Quantization Issues**:
- Rounding to 0/1 destroys information
- Different float vectors become identical after quantization
- **Solution**: Train with quantization in mind

**Model-Aware Quantization**:
- Study weight statistics to learn better representations
- Most weights cluster around mean, few outliers
- Store outliers in full precision, quantize the rest
- **LLaMA.int**: Quantize each row/column separately for better precision

#### Advanced Quantization Techniques

**Conservative Quantization (INT8)**:
- Direct quantization without retraining often works well
- **Offline Calibration**: Gather statistics before deployment
- **Online Calibration**: Calculate min/max dynamically at runtime
- **Outlier Handling**: Use statistical measures to clip range intelligently
- **Per-channel Quantization**: Different scale factors per output channel

**Aggressive Quantization (INT4 and Lower)**:
- Usually requires retraining for reasonable accuracy
- **Bootstrapping**: Start with trained FP32 weights
- **Activation Function Replacement**: Replace ReLU with bounded functions
- **Network Structure Modification**: Use wider layers to compensate
- **First/Last Layer Preservation**: Keep at FP32 or use conservative quantization
- **Mixed Precision**: Higher precision for activations than weights

#### Quantization-Aware Training (QAT)

**Straight-Through Estimator (STE)**:
- Problem: Quantization functions have zero derivative almost everywhere
- Solution: Pass gradients through quantization functions as-is
- Enables backpropagation through discrete-valued functions

**Training Process**:
1. Maintain full precision copy of weights throughout training
2. Quantize weights as integral part of training graph
3. Backpropagate through quantization operations
4. Use quantized weights only for inference

**Historical Examples**:
- **Binary Neural Networks (2016)**: All weights/activations -1 or 1
- Achieved 10% error on CIFAR-10 (state-of-art was ~12%)
- **Layer-by-layer distillation**: Train quantized layers to match full-precision outputs
- **QLoRA**: Parameter-efficient fine-tuning for 4-bit models

#### Hardware Considerations
- Some data types not supported (e.g., int3)
- PyTorch modules may not support quantization (e.g., RNNs)
- Custom hardware accelerators often required
- Trade-off: Small models may not speed up, but large models can double inference speed

#### Mathematical Foundations

**Integer vs FP32 Trade-offs**:
- **FP32**: Dynamic range ±3.4×10³⁸, ~4.2×10⁹ representable values
- **INT8**: Dynamic range [-128, 127], 256 representable values
- **INT4**: Dynamic range [-8, 7], 16 representable values

**Energy Efficiency Comparison**:
| Operation | Energy Saving vs FP32 | Area Saving vs FP32 |
|-----------|----------------------|-------------------|
| Add | 30x | 116x |
| Multiply | 18.5x | 27x |

**Range-Based Linear Quantization**:

*Asymmetric Mode*:
- Maps min/max float range to min/max integer range
- Uses zero-point (quantization bias) in addition to scale factor
- Formula: `x_q = round(q_x * (x_f - min_x_f) / (max_x_f - min_x_f))`
- Zero-point: `zp_x = round(min_x_f / q_x)`
- Ensures zero is exactly representable

*Symmetric Mode*:
- Uses maximum absolute value between min/max
- No zero-point required
- Quantized range symmetric with respect to zero
- **Full Range**: [-128, 127] for 8-bit
- **Restricted Range**: [-127, 127] for 8-bit

**Scale Factor Approximation**:
- Replace floating-point scale factor with integer multiplication + bit shift
- Formula: `Q ≈ A/2ⁿ` where Q is FP32 scale factor, A is integer multiplier
- Hardware-friendly: `A = ⌊2ⁿQ⌋`, `n = ⌊log₂((2ᵐ-1)/Q)⌋`

### 2. Pruning

**Definition**: Completely eliminate some parameters while leaving others unchanged.

#### Magnitude Pruning
- Set to zero parameters with least magnitude
- Intuition: Parameters close to zero aren't doing much
- Can remove ~50% of parameters with minimal performance impact
- **Unstructured pruning**: Remove parameters anywhere in model

#### Lottery Ticket Hypothesis
- Subnetworks of trained models can be better than random initialization
- Prune model, retrain, and it generalizes better than original
- 20% size model can outperform full model after retraining

#### WANDA (CMU Research)
- Considers input magnitude, not just weight magnitude
- Example: Small weight B processing large inputs (avg=1000) vs large weight A processing small inputs (avg=1)
- B may have outsized impact despite smaller magnitude
- Use calibration data to learn average input magnitudes

#### Structured vs Unstructured Pruning

**Unstructured Pruning Problems**:
- Hardware doesn't support sparse operations well
- Still multiply zeros in dense operations
- No performance benefits with current hardware

**Structured Pruning**:
- Remove entire components (attention heads, layers)
- Immediate impact on model performance
- **Attention Head Pruning**: Remove half of attention heads with negligible impact
- **Two-level masking**: Coarse masks (entire layers) + fine masks (individual dimensions)

#### Gradient-Free Pruning
- **Problem**: Pruning requires as much compute as training
- **Solution**: Randomly mask modules, measure performance, learn regression
- Sample from combinatorial space to predict module interactions
- No gradients needed, just forward passes

#### Advanced Pruning Algorithms

**Magnitude Pruner**:
- Basic thresholding: `thresh(w_i) = {w_i if |w_i| > λ, 0 if |w_i| ≤ λ}`
- Different threshold per layer's weights tensor

**Sensitivity Pruner**:
- Uses standard deviation as normalizing factor
- Threshold: `λ = s * σ_l` where σ_l is std of layer l
- About 68% of elements have |w_i| < σ in normal distribution
- Set s based on sensitivity analysis results

**Level Pruner**:
- Specify target sparsity level (e.g., 0.5 = 50% sparsity)
- More stable than sensitivity pruner
- Sort weights by absolute values, mask smallest until target reached

**Automated Gradual Pruner (AGP)**:
- Increases sparsity from initial value s_i to final value s_f over n steps
- Mathematical formula controls sparsity growth
- Prunes rapidly initially, gradually reduces pruning rate
- Requires minimal hyperparameter tuning

**Structure Ranking Pruners**:
- **L1RankedStructureParameterPruner**: Uses mean absolute value of structures
- **ActivationAPoZRankedFilterPruner**: Uses average percentage of zeros in activations
- **GradientRankedFilterPruner**: Uses product of gradients and filter values
- **RandomRankedFilterPruner**: For research comparison purposes

**Hybrid Pruning**:
- Combine different pruning techniques in single schedule
- Mix pruning and regularization
- Apply different methods to same tensor
- Example: Filter pruning → thinning → element-wise pruning

### 3. Distillation

**Definition**: Train small model to replicate behavior of large model.

#### Knowledge Distillation Types

**Hard Targets**:
- Use teacher's predicted label as target
- Simple and intuitive
- Example: LLaMA predicts "positive" → student learns "positive"

**Soft Targets**:
- Match full probability distribution over labels
- Richer information than single labels
- Optimize distribution difference, not just correct answer probability
- **Key insight**: Usually not possible with human annotators

#### Self-Distillation Results
- Distill model to itself using soft targets
- Consistently improves performance
- **Intuition**: Soft targets provide richer knowledge interface
- Conveys uncertainties and alternative answers

#### Sequence-Level Distillation
- **Word-level**: Match teacher's word distribution at each step
- **Sequence-level**: Generate full sentence from teacher, maximize probability
- **Combination**: Use both objectives together
- Addresses exposure bias (teacher/student divergence during generation)

#### DistilBERT Example
- Take every other layer of BERT (12→6 layers)
- Initialize from original BERT layers
- Soft target distillation + language modeling
- **Finding**: Supervised objective doesn't help much with good teacher
- Maintain embedding space geometry similarity
- Nearly as good as full BERT on most tasks

#### Advanced Distillation Applications

**Self-Instruct**:
- Vanilla language model → instruction-following model
- Generate instructions → generate responses → train on own behavior
- **Key trick**: Generate class first, then inputs (reverse order)
- Decompose hard problems into easier subproblems

**PromptModel**:
- Combine retrieved datasets + generated data
- Beat GPT-3 (the teacher) by leveraging existing data
- **Synthetic Data Generation**: Current hot research topic
- PyTorch-like toolkits emerging for data generation pipelines

#### Mathematical Framework

**Softmax Temperature**:
- Standard softmax: `p_i = exp(z_i) / Σ_j exp(z_j)`
- Temperature scaling: `p_i = exp(z_i/T) / Σ_j exp(z_j/T)`
- Higher T produces softer probability distributions
- Reveals "dark knowledge" about class similarities

**Loss Function**:
```
L(x; W) = α * L(y, σ(z_s; T=1)) + β * L(σ(z_t; T=τ), σ(z_s; T=τ))
```
Where:
- `α`, `β`: Weighting coefficients
- `z_s`, `z_t`: Student and teacher logits
- `τ`: Temperature parameter
- `σ`: Softmax function

**Hyperparameter Guidelines**:
- **Temperature (τ)**: Range 1-20, lower for smaller student models
- **Weighting (α, β)**: Usually α << β, but can use α = β = 0.5
- **Student Capacity**: Smaller models may not capture rich soft-label information

## Implementation Considerations

### When to Use Each Technique

**Quantization**:
- Best for large models where memory is bottleneck
- Requires hardware/framework support
- Can double inference speed for very large models

**Pruning**:
- Best for reducing model size with minimal performance loss
- Structured pruning more practical than unstructured
- Requires significant compute for training masks

**Distillation**:
- Most flexible - can change architecture completely
- Requires unlabeled data matching expected inputs
- Can unlock capabilities impossible with traditional learning

### Hardware Limitations
- Not all data types supported by processors
- Framework support varies (PyTorch, etc.)
- Custom accelerators often needed for optimal performance
- Sparse operations not well-supported currently

### Trade-offs
- **Quantization**: Speed vs precision
- **Pruning**: Size vs performance  
- **Distillation**: Flexibility vs training complexity

## PyTorch Implementation Guide

### Quantization in PyTorch

PyTorch supports three quantization modes starting from version 1.3:

#### 1. Dynamic Quantization
- Easiest method - converts weights to int8 and activations on-the-fly
- Computations use efficient int8 matrix multiplication
- Activations read/written in floating point format
- **API**: `torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)`

#### 2. Post-Training Static Quantization
- Converts networks to use both integer arithmetic and int8 memory accesses
- Requires calibration data to compute activation distributions
- **Features**:
  - Custom observers for statistics collection
  - Operator fusion to save memory access
  - Per-channel quantization for higher accuracy
- **API**:
  ```python
  model.qconfig = torch.quantization.get_default_config('fbgemm')
  torch.quantization.prepare(model, inplace=True)
  # Calibrate with data
  torch.quantization.convert(model, inplace=True)
  ```

#### 3. Quantization-Aware Training (QAT)
- Highest accuracy of the three methods
- Weights and activations "fake quantized" during training
- **API**:
  ```python
  model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
  torch.quantization.prepare_qat(model, inplace=True)
  # Train model
  quantized_model = torch.quantization.convert(model.eval(), inplace=False)
  ```

### Pruning in PyTorch

#### Basic Pruning Operations
```python
import torch.nn.utils.prune as prune

# Random unstructured pruning
prune.random_unstructured(module, name="weight", amount=0.3)

# L1 unstructured pruning
prune.l1_unstructured(module, name="weight", amount=0.2)

# Structured pruning (L2 norm)
prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

# Global pruning across multiple parameters
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
```

#### Pruning Mechanics
- Pruning creates `weight_orig` parameter and `weight_mask` buffer
- Forward pass uses masked weights automatically
- Can remove re-parametrization with `prune.remove(module, 'weight')`
- Model state_dict includes all masks for serialization

#### Custom Pruning Methods
```python
class CustomPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # Custom pruning logic here
        return mask
```

### Model Selection Guidelines

| Model Type | Preferred Scheme | Why |
|------------|------------------|-----|
| LSTM/RNN | Dynamic Quantization | Throughput dominated by compute/memory bandwidth for weights |
| BERT/Transformer | Dynamic Quantization | Throughput dominated by compute/memory bandwidth for weights |
| CNN | Static Quantization | Throughput limited by memory bandwidth for activations |
| CNN | Quantization Aware Training | When accuracy can't be achieved with static quantization |

### Performance Results

**Quantization Performance**:
- 4x reduction in model size
- 2-4x reduction in memory bandwidth  
- 2-4x faster inference (varies by hardware)

**Accuracy Results** (ImageNet):
- ResNet-50: 76.1% → 75.9% (static quantization)
- MobileNet-v2: 71.9% → 71.6% (QAT)
- BERT (GLUE MRPC): 0.902 → 0.895 (dynamic quantization)

## Advanced Pruning Concepts

### Sparsity Definition
- **Sparsity**: Measure of zero elements relative to tensor size
- **L0 "norm"**: Counts non-zero elements: ‖x‖₀ = |x₁|⁰ + |x₂|⁰ + ... + |xₙ|⁰
- **Density**: Complement of sparsity (density = 1 - sparsity)

### Pruning Granularity

#### Element-wise Pruning
- Prunes individual weight elements
- Also called fine-grained pruning
- Most flexible but hardware support limited

#### Structured Pruning  
- Prunes entire groups of elements
- Examples: filter pruning, channel pruning
- Better hardware support for inference speedup

### Pruning Schedules

#### One-shot Pruning
- Prune trained model once
- "Free lunch" effect: 2x connection reduction without accuracy loss
- Limited potential compared to iterative approaches

#### Iterative Pruning
- Prune → retrain → prune → retrain...
- Much higher sparsity at no accuracy loss
- Requires careful scheduling decisions

### Sensitivity Analysis
- Method to rank layers by pruning sensitivity
- Test different sparsity levels per layer
- Helps determine optimal pruning ratios
- **Process**:
  1. Set pruning level for specific layer
  2. Prune once and evaluate
  3. Repeat for all layers at multiple sparsity levels
  4. Rank layers by sensitivity

**Key Findings**:
- Feature detection layers (conv) more sensitive to pruning
- Sensitivity decreases with layer depth
- Fully-connected layers less sensitive (good for parameter reduction)
- Layers in same stage often have similar sensitivity



### Advanced Quantization Techniques

#### Conservative Quantization (INT8)
- Direct quantization without retraining often works well
- **Offline Calibration**: Gather statistics before deployment
- **Online Calibration**: Calculate min/max dynamically at runtime
- **Outlier Handling**: Use statistical measures to clip range intelligently
- **Per-channel Quantization**: Different scale factors per output channel

#### Aggressive Quantization (INT4 and Lower)
- Usually requires retraining for reasonable accuracy
- **Bootstrapping**: Start with trained FP32 weights
- **Activation Function Replacement**: Replace ReLU with bounded functions
- **Network Structure Modification**: Use wider layers to compensate
- **First/Last Layer Preservation**: Keep at FP32 or use conservative quantization
- **Mixed Precision**: Higher precision for activations than weights

### Quantization-Aware Training (QAT)

#### Straight-Through Estimator (STE)
- Problem: Quantization functions have zero derivative almost everywhere
- Solution: Pass gradients through quantization functions as-is
- Enables backpropagation through discrete-valued functions

#### Training Process
1. Maintain full precision copy of weights throughout training
2. Quantize weights as integral part of training graph
3. Backpropagate through quantization operations
4. Use quantized weights only for inference





## Key Takeaways

1. **Overparameterization enables compression**: Many parameters are training-time tricks
2. **Hardware matters**: Compression effectiveness limited by hardware/framework support
3. **Combination approaches**: Often best results from combining techniques
4. **Distillation is most flexible**: Can change architecture and unlock new capabilities
5. **Synthetic data generation**: Emerging as major research direction

## References

- Binary Neural Networks (2016)
- WANDA pruning method
- Lottery Ticket Hypothesis
- DistilBERT
- Self-Instruct
- PromptModel
- QLoRA
- Various quantization methods (AbsMax, LLaMA.int, etc.)
