
# 01_Hardware_and_Low-Level_Foundations

### GPU Architecture & Tensor Cores

- **Streaming Multiprocessor (SM)**: The core compute unit. Each SM (e.g. Ampere A100) has 64 FP32 CUDA cores and 4 Tensor Cores:contentReference[oaicite:0]{index=0}. The A100 GPU implements 108 SMs (6912 FP32 cores, 432 3rd-gen tensor cores):contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}, whereas the Volta V100 had 80 SMs (5120 cores, 640 2nd-gen tensor cores):contentReference[oaicite:3]{index=3}.  
- **Tensor Cores**: Specialized 4×4×fma matrix units for mixed-precision math. Ampere’s 3rd-gen tensor cores support FP16, BF16, TF32, INT8, etc., and double throughput with structured sparsity:contentReference[oaicite:4]{index=4}. For example, each A100 SM has 4 tensor cores delivering 1024 FP16/FP32 FMA ops per clock:contentReference[oaicite:5]{index=5}. The Hopper H100 further increases Tensor Core count (528 per GPU in SXM form):contentReference[oaicite:6]{index=6} and adds FP8 precision.  
- **Memory & Cache**: Each SM includes ∼100 KB of shared memory plus 192 KB of L1 cache (Ampere):contentReference[oaicite:7]{index=7}. The A100 has 40 MB L2 cache (vs 6 MB on V100) and 40 GB HBM2e with ~1555 GB/s bandwidth:contentReference[oaicite:8]{index=8}. (The RTX4090 has 24 GB GDDR6X @1008 GB/s:contentReference[oaicite:9]{index=9}.)  

:contentReference[oaicite:10]{index=10}*Figure: NVIDIA GA100 (Ampere) GPU block diagram (full GA100 has 128 SMs; A100 uses 108 SMs):contentReference[oaicite:11]{index=11}.*  
The NVIDIA GA100 GPU (basis of A100) consists of 128 SMs (A100 has 108):contentReference[oaicite:12]{index=12}. Each SM packs 64 FP32 cores and 4 Tensor Cores, yielding 6912 FP32 cores and 432 tensor cores on A100:contentReference[oaicite:13]{index=13}. For comparison, the earlier Tesla V100 (Volta) had 80 SMs (5120 CUDA cores, 640 tensor cores):contentReference[oaicite:14]{index=14}. The A100 also features 40 MB of L2 cache and 40 GB HBM2e memory at ~1555 GB/s:contentReference[oaicite:15]{index=15} (V100 had 16 GB HBM2 at 900 GB/s:contentReference[oaicite:16]{index=16}). 

### Memory Hierarchy

Registers → Shared Memory → L1 Cache → L2 Cache → HBM/VRAM

- **Registers**: Fastest per-thread storage (~255 registers/thread).  
- **Shared Memory**: Low-latency on-chip memory (e.g. 128 KB per SM on Ampere, used for data reuse).  
- **L1 Cache**: Per-SM cache (128–192 KB on recent GPUs) for shared/global memory.  
- **L2 Cache**: Large global cache (40–50 MB on data-center GPUs) shared by all SMs.  
- **HBM/VRAM**: Off-chip memory (e.g. 40–80 GB HBM2/3) providing terabytes/sec bandwidth.  

### Key Specs (Examples)

- **V100**: 80 SMs, 5120 CUDA cores, 640 Tensor Cores, 16 GB HBM2, 900 GB/s
- **A100**: 108 SMs, 6912 CUDA cores, 432 Tensor Cores, 40 GB HBM2e, ~1555 GB/s 
- **H100 (SXM)**: 132 SMs, 16896 CUDA cores, 528 Tensor Cores, 80 GB HBM3, ~3000+ GB/s   
- **RTX 4090**: 128 SMs, 16384 CUDA cores, 512 Tensor Cores, 24 GB GDDR6X, 1008 GB/s 

### Data Types & Precision

| Type  | Bits | Exponent | Mantissa | Range (approx.)   | Precision (digits) | Use Case            |
|-------|------|----------|----------|-------------------|--------------------|---------------------|
| **FP32**  | 32   | 8        | 23       | ~±3.4×10³⁸        | ~7.2               | General compute     |
| **TF32**  | 19   | 8        | 10       | ~±3.4×10³⁸        | ~2.8               | Mixed-precision (Tensor Cores):contentReference[oaicite:26]{index=26} |
| **BF16**  | 16   | 8        | 7 (+1)   | ~±3.4×10³⁸        | ~1.6               | Training (stable FP32 range):contentReference[oaicite:27]{index=27} |
| **FP16**  | 16   | 5        | 10       | ~±65,504          | ~3.3               | Tensor Cores (inference/training)    |
| **INT8**  | 8    | —        | 8 (int)  | –128…127          | Exact integers     | Inference quantization |
| **FP8**   | 8    | 4–5      | 2–3      | ±448 (approx.)    | ~1.2               | Hopper Tensor Cores (training/inference) |

TF32 is an NVIDIA innovation: 8-bit exponent + 10-bit mantissa (one bit more than FP16), giving FP32-like range with reduced precision. BFloat16 (BF16) uses the same 8-bit exponent as FP32 but only 7 explicit fraction bits, preserving FP32 range (±3×10³⁸) with lower precision (~1.5–2 digits).

### PTX: CUDA’s Virtual ISA

**What is PTX?**  

PTX (Parallel Thread eXecution) is NVIDIA’s intermediate ISA and virtual machine for GPUs. It is a RISC-like, platform-independent code with instructions like `add.f32`, memory qualifiers (`.global`, `.shared`), and supports predicated execution. The CUDA compiler (nvcc) generates PTX, which the driver JIT-compiles to native GPU code at runtime. This allows forward-compatibility: a single PTX can run on multiple GPU generations.

**Workflow**:
1. Write CUDA C/C++ ➔ nvcc compiles to PTX (and/or other cuda binaries).  
2. At load time, the CUDA **Driver API** or **Runtime** JIT-compiles PTX to the target GPU’s SASS binary.  
3. The JIT compiler may use `sm_XX` targets or fallback if architecture is new.

### CUDA Driver API

The CUDA **Driver API** is a low-level C API for managing devices, contexts, and modules. Common sequence:

```cpp
cuInit(0);                            // Initialize driver
cuDeviceGet(&dev, 0);                 // Get device handle
cuCtxCreate(&ctx, 0, dev);            // Create context on device
cuModuleLoad(&mod, "kernel.ptx");     // Load compiled PTX/CUBIN module
cuModuleGetFunction(&func, mod, "vec_add"); // Get kernel function
// Launch kernel
void* args[] = { &dA, &dB, &dC, &N };
cuLaunchKernel(func,
               gridX,1,1, blockX,1,1,
               0, 0, args, 0);
cuCtxSynchronize();                   // Wait for completion
cuModuleUnload(mod);
cuCtxDestroy(ctx);
```

Each function returns a `CUresult` status. The driver API provides fine control over JIT options, contexts on multiple GPUs, and loading of raw cubin/ptx, compared to the higher-level Runtime API.

### Software Stack Dependency

NVIDIA’s CUDA stack builds upward from hardware:

```
Hardware (SMs, Tensor Cores)
    ↓
CUDA Driver API & Runtime (JIT compilation, kernel launch)
    ↓
PTX Virtual ISA
    ↓
CUDA Toolkit Libraries (cuBLAS, cuDNN, cuSPARSE, cuFFT, etc.; CUTLASS templates)
    ↓
Inference Engines (TensorRT, etc.)
    ↓
Frameworks & Compilers (PyTorch, TensorFlow, JAX)
    ↓
torch.compile (Inductor + NVFuser, etc.)
```

Each layer leverages primitives below. For example, cuBLAS/CUTLASS implement GEMM on top of hardware; cuDNN uses cuBLAS (GEMMs) for convolutions; TensorRT builds on cuDNN/cuBLAS; PyTorch uses cuDNN/cuBLAS by default for tensors.

---

# 02\_CUTLASS\_Deep-Dive

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) is an open-source, header-only library of CUDA C++ template abstractions for GEMM (matrix multiply) and related operations. It provides reference implementations of high-performance kernels (for GEMM, convolutions, reductions, etc.) with fine-grained control over tiling, memory layout, and pipelining. CUTLASS lets researchers and developers **customize every stage** of a matrix multiply (threadblock size, warp size, instruction shape, epilogue ops) while still leveraging NVIDIA’s Tensor Cores.

CUTLASS’s core hierarchy includes:

```
cutlass/
├─ gemm/
│  ├─ device/   (high-level GEMM templates, e.g. cutlass::gemm::device::Gemm)
│  ├─ kernel/   (GPU kernel implementations)
│  └─ threadblock/
├─ conv/         (Conv2d forward/backward kernels)
├─ reduction/    (reductions, e.g. row/col reductions)
├─ epilogue/     (post-processing: bias, activations, etc.)
└─ layout/       (data layouts: RowMajor, ColumnMajor, Interleaved, TensorNHWC, etc.)
```

### GEMM Example

Basic CUTLASS GEMM usage (matrix multiply C = α A·B + β C):

```cpp
using ElementA = cutlass::half_t;
using LayoutA  = cutlass::layout::ColumnMajor;
using ElementB = cutlass::half_t;
using LayoutB  = cutlass::layout::RowMajor;
using ElementC = cutlass::half_t;
using LayoutC  = cutlass::layout::RowMajor;
using ElementD = cutlass::half_t;
using LayoutD  = cutlass::layout::RowMajor;

using GemmOp = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementD, LayoutD,
    float,                         // accumulator type
    cutlass::arch::OpClassTensorOp, // use Tensor Cores
    cutlass::arch::Sm80,            // target Ampere SM
    cutlass::gemm::GemmShape<128,128,32>,  // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,  // warp tile
    cutlass::gemm::GemmShape<8,  8,  4>    // instruction tile
>;

GemmOp gemm_op;
cutlass::Status status = gemm_op(
    {M, N, K},     // GEMM dimensions
    alpha,         // scalar alpha
    A, lda,        // pointer, leading dim of A
    B, ldb,        // B
    beta,          // scalar beta
    C, ldc,        // C (output)
    D, ldd         // D (if different from C, e.g. for bias)
);
```

This runs a matrix multiply using FP16 inputs on Tensor Cores (OpClassTensorOp). CUTLASS automatically schedules threads/warps/blocks.

### Custom Epilogue (Bias + Activation)

CUTLASS allows custom epilogue functors. For instance, add bias and apply a GELU activation:

```cpp
template <typename ElementOutput, typename ElementAccum>
struct BiasGeluEpilogue {
  using FragmentOut = cutlass::Array<ElementOutput, 128>;
  using FragmentAcc = cutlass::Array<ElementAccum, 128>;

  __device__ void operator()(FragmentOut &output,
                             FragmentAcc const &accum,
                             FragmentOut const &bias) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentOut::kElements; ++i) {
      float x = float(accum[i] + bias[i]);
      // Gaussian Error Linear Unit approximation
      output[i] = ElementOutput(0.5f * x * (1.0f + erff(x * 0.70710678f)));
    }
  }
};
```

This functor can be plugged into a `cutlass::gemm::device::Gemm` as the epilogue to apply bias + GELU after accumulation.

### Convolution Example

CUTLASS also supports convolutions via its `cutlass::conv` namespace. Example: 2D forward convolution using tensor cores:

```cpp
using Conv2dFpropOp = cutlass::conv::device::Conv2dFprop<
    float, cutlass::layout::TensorNHWC<float>,  // input layout
    float, cutlass::layout::TensorNHWC<float>,  // filter layout
    float, cutlass::layout::TensorNHWC<float>,  // output layout
    float,                                  // accumulator
    cutlass::arch::OpClassTensorOp,         // tensor cores
    cutlass::arch::Sm80,                    // Ampere
    cutlass::conv::Conv2dFprop::ThreadblockShape,
    cutlass::conv::Conv2dFprop::WarpShape,
    cutlass::conv::Conv2dFprop::InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        float,                           // Data type for output
        128/sizeof(float),               // Vector width
        float, float                    // Accum & compute type
    >,
    cutlass::conv::threadblock::DefaultConv2dFpropThreadblockSwizzle<>,
    3,  // stages (pipeline depth)
    1,  // alignmentA
    1   // alignmentB
>;

cutlass::conv::Conv2dProblemSize problem(
    batch, in_ch, in_h, in_w,
    filt_ch, filt_h, filt_w,
    out_ch, out_h, out_w,
    pad_h, pad_w, str_h, str_w,
    dil_h, dil_w);

Conv2dFpropOp conv_op;
conv_op.initialize(problem, alpha, A, B, beta, C, D);
conv_op();
```

This runs a convolution (forward prop) using the specified tensor-core-based kernel. The `LinearCombination` epilogue just does a simple α·accum + β·C.

### Memory Layouts

CUTLASS includes layout types for different tensor formats:

```cpp
using RowMajor  = cutlass::layout::RowMajor;
using ColMajor  = cutlass::layout::ColumnMajor;
using Interleaved32 = cutlass::layout::ColumnMajorInterleaved<32>;
using TensorNHWC = cutlass::layout::TensorNHWC<float>; 
using TensorNCHW = cutlass::layout::TensorNCHW<float>;
```

Layouts control how multi-dimensional arrays map to linear memory, enabling support for blocked/interleaved data for performance or specialized formats (e.g. NCHW vs NHWC).

### Performance & Use Cases

CUTLASS kernels can achieve near-peak performance by tuning threadblock/warp/instruction shapes and using shared memory efficiently. For example, customized CUTLASS GEMMs often match or slightly exceed cuBLAS speeds for non-standard shapes. Key uses:

* **Research**: design new tiling algorithms or sparsity formats
* **Custom layers**: implement novel tensor operations not in cuBLAS/cuDNN
* **Maximal tuning**: squeeze extra percent performance on a given GPU

**Trade-off**: CUTLASS gives fine control and high performance, but requires deep CUDA knowledge and more code complexity than simply calling cuBLAS/cuDNN.

*Citation*: NVIDIA’s CUTLASS provides template libraries for high-performance GEMM and convolutions.

---

# 03\_cuBLAS\_Advanced

**cuBLAS** is NVIDIA’s GPU-accelerated BLAS library (Level-1/2/3) with highly tuned kernels, including automatic Tensor Core usage. It supports standard BLAS calls and extended precision (FP16, BF16, INT8) via `cublasGemmEx`.

### Handle & Stream Setup

```cpp
cublasHandle_t handle;
cublasCreate(&handle);                         // Initialize cuBLAS handle
cublasSetStream(handle, stream);               // (optional) use CUDA stream
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // Enable Tensor Core paths:contentReference[oaicite:33]{index=33}
```

The cuBLAS handle (context) must be created before use and destroyed with `cublasDestroy(handle)`. Setting `CUBLAS_TENSOR_OP_MATH` (now default) allows automatic use of Tensor Cores.

### GEMM and Mixed Precision

**FP16/BF16 GEMM**: Use `cublasGemmEx` with `CUDA_R_16F` (or `CUDA_R_16BF` for BF16) and `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for tensor cores. Example: FP16 compute, FP16 storage:

```cpp
// A, B as FP16, compute in FP32 (prefer tensor cores)
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K,
             &alpha,
             A, CUDA_R_16F, lda,
             B, CUDA_R_16F, ldb,
             &beta,
             C, CUDA_R_16F, ldc,
             CUDA_R_32F,                 // compute in FP32
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

For **BF16**, use `CUDA_R_16BF` similarly (compute in FP32). For **INT8** inference GEMM:

```cpp
// A_int8, B_int8 -> C_int32, compute in INT32
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K,
             &alpha,
             A_int8, CUDA_R_8I, lda,
             B_int8, CUDA_R_8I, ldb,
             &beta,
             C_int32, CUDA_R_32I, ldc,
             CUDA_R_32I,
             CUBLAS_GEMM_ALGO3); // INT8-optimized algorithm
```

This uses 8-bit integer computation (INT8 inputs, INT32 accum).

### Strided & Batched Operations

For many small GEMMs, use **Strided Batched GEMM** to amortize launch overhead:

```cpp
// Perform batchCount independent GEMMs in one call
cublasSgemmStridedBatched(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    A, lda, strideA,
    B, ldb, strideB,
    &beta,
    C, ldc, strideC,
    batchCount);
```

This computes `C[i] = α * A[i]*B[i] + β*C[i]` for i=0..batchCount-1, where each matrix is offset by given strides. Very useful in CNNs or transformer batching.

### Algorithm Tuning

cuBLAS provides multiple GEMM algorithms. You can query the fastest algorithm for a given shape:

```cpp
cublasGemmAlgo_t algo;
cublasGetGemmAlgo(handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  M, N, K,
                  CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                  &algo);
printf("Best algorithm: %d\n", algo);
```

You can also benchmark by looping over `cublasGemmEx` with different `cublasGemmAlgo_t` values to find the lowest latency.

### Integration with PyTorch

PyTorch automatically uses cuBLAS for CPU→GPU tensor operations:

```python
import torch

A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
# Uses cuBLAS SGEMM internally:
C = torch.mm(A, B)

# Mixed precision uses Tensor Cores via cuBLAS:
A16 = A.half(); B16 = B.half()
C16 = torch.mm(A16, B16)
```

### Best Practices

* Always set `CUBLAS_TENSOR_OP_MATH` (Tensor Cores) if using FP16/BF16.
* Use batched/strided BLAS for multiple small ops.
* Query/benchmark algorithms at startup (caching the best).
* Pin host memory when doing frequent CPU→GPU transfers to speed DMA.

*Citations*: cuBLAS requires creating a handle with `cublasCreate()` and destroying it with `cublasDestroy()`. Enabling `CUBLAS_TENSOR_OP_MATH` allows Tensor Core acceleration.

---

# 04\_cuDNN\_Deep-Dive

**cuDNN** is NVIDIA’s GPU-accelerated library of deep learning primitives (conv, pooling, normalization, RNNs, etc.). It provides highly optimized implementations with automatic algorithm selection and Tensor Core support.

### Handle & Descriptor Setup

```cpp
cudnnHandle_t handle;
cudnnCreate(&handle);                    // Initialize cuDNN handle
cudnnSetStream(handle, stream);          // (optional) use CUDA stream
cudnnSetTensorMathType(handle, CUDNN_TENSOR_OP_MATH); // Enable Tensor Cores:contentReference[oaicite:39]{index=39}
```

The cuDNN handle (context) is created once and passed to all cuDNN calls; destroy it with `cudnnDestroy(handle)` when done.

Descriptors define tensor shapes:

```cpp
cudnnTensorDescriptor_t xDesc, yDesc;
cudnnFilterDescriptor_t wDesc;
cudnnConvolutionDescriptor_t convDesc;
cudnnCreateTensorDescriptor(&xDesc);
cudnnCreateFilterDescriptor(&wDesc);
cudnnCreateConvolutionDescriptor(&convDesc);

cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW,
                           CUDNN_DATA_FLOAT, N, C, H, W);
cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                           K, C, R, S);
cudnnSetConvolution2dDescriptor(convDesc,
    pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
```

The math type `CUDNN_TENSOR_OP_MATH` tells cuDNN to use tensor cores where possible.

### Convolutions (Forward)

Choose the best algorithm and workspace:

```cpp
cudnnConvolutionFwdAlgo_t algo;
cudnnGetConvolutionForwardAlgorithm(handle,
    xDesc, wDesc, convDesc, yDesc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    /*memoryLimit=*/0, &algo);
size_t ws = 0;
cudnnGetConvolutionForwardWorkspaceSize(handle,
    xDesc, wDesc, convDesc, yDesc,
    algo, &ws);
void* workspace = nullptr;
if (ws > 0) cudaMalloc(&workspace, ws);
```

Then launch forward convolution:

```cpp
float alpha = 1.0f, beta = 0.0f;
cudnnConvolutionForward(handle, &alpha,
    xDesc, x, wDesc, w,
    convDesc, algo, workspace, ws,
    &beta, yDesc, y);
```

cuDNN automatically selects the kernel (FFT, Winograd, GEMM, etc.) based on shape/precision.

### Pooling

```cpp
cudnnPoolingDescriptor_t poolDesc;
cudnnCreatePoolingDescriptor(&poolDesc);
cudnnSetPooling2dDescriptor(poolDesc,
    CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
    window_h, window_w, pad_h, pad_w,
    stride_h, stride_w);
cudnnPoolingForward(handle, poolDesc,
    &alpha, xDesc, x,
    &beta, yDesc, y);
```

Supports max/avg/L2 pooling and backward.

### Normalization

* **BatchNorm (Training)**:

  ```cpp
  cudnnBatchNormalizationForwardTraining(
      handle, CUDNN_BATCHNORM_SPATIAL,
      &alpha, &beta,
      xDesc, x, yDesc, y,
      bnDesc, scale, bias, 1.0,
      runningMean, runningVar,
      epsilon, savedMean, savedInvVar);
  ```
* **BatchNorm (Inference)**:

  ```cpp
  cudnnBatchNormalizationForwardInference(
      handle, CUDNN_BATCHNORM_SPATIAL,
      &alpha, &beta,
      xDesc, x, yDesc, y,
      bnDesc, scale, bias, estimatedMean, estimatedVar,
      epsilon);
  ```
* **LayerNorm** (from cuDNN v8):

  ```cpp
  cudnnLayerNormForwardInference(
      handle, &alpha, &beta,
      xDesc, x, yDesc, y,
      scaleDesc, scale, biasDesc, bias, epsilon);
  ```

### RNN (LSTM/GRU)

Configure an RNN descriptor (v8 API):

```cpp
cudnnRNNDescriptor_t rnnDesc;
cudnnCreateRNNDescriptor(&rnnDesc);
cudnnSetRNNDescriptor_v8(
    rnnDesc, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
    CUDNN_RNN_SINGLE_INP_DIRECTION, CUDNN_LINEAR_INPUT,
    CUDNN_TENSOR_OP_MATH, CUDNN_DEFAULT_MATH,
    inputSize, hiddenSize, numLayers, dropoutDesc, 
    CUDNN_RNN_PADDED_IO_ENABLED);
```

Then call `cudnnRNNForwardTraining` for a sequence:

```cpp
cudnnRNNForwardTraining(handle, rnnDesc, seqLength,
    xDesc, x, hxDesc, hx, cxDesc, cx,
    wDesc, weights,
    yDesc, y, hyDesc, hy, cyDesc, cy,
    workspace, wsSize, reserveSpace, rsSize);
```

### PyTorch Integration

PyTorch’s `nn.Conv2d`, `nn.Linear`, etc. call into cuDNN/cuBLAS under the hood. Example:

```python
import torch, torch.backends.cudnn as bc
bc.benchmark = True
bc.deterministic = False
conv = torch.nn.Conv2d(3,64,3,padding=1).cuda()
x = torch.randn(32,3,224,224,device='cuda')
y = conv(x)  # Uses cuDNN convolution
```

Setting `torch.backends.cudnn.benchmark = True` lets cuDNN choose the fastest convolution algorithm for given shapes.

### Best Practices

* **Tensor Cores**: Enable via `CUDNN_TENSOR_OP_MATH` (if doing fp16/bf16).
* **Workspace**: Allocate a single reusable workspace to avoid repeated `cudaMalloc`.
* **Algorithm Selection**: Let cuDNN pick the fastest algorithm (benchmarking mode) or manually query as above.
* **Mixed Precision**: FP16/INT8 convs leverage Tensor Cores. Ensure input dims are multiples of 8 for max performance.

*Citations*: Initialize cuDNN with `cudnnCreate` and destroy with `cudnnDestroy`. Setting math mode `CUDNN_TENSOR_OP_MATH` enables Tensor Cores.

---

# 05\_TensorRT\_and\_Frameworks

**TensorRT** is NVIDIA’s high-performance deep learning inference optimizer and runtime. It loads trained models (ONNX, TensorFlow, etc.), applies optimizations (layer fusion, quantization), and produces a serialized engine for fast inference. Key features include FP16/INT8 quantization, dynamic shape support, and automatic use of Tensor Cores.

### Building an Engine

```cpp
// 1) Create builder and network
auto builder = createInferBuilder(gLogger);
auto network = builder->createNetworkV2(1U << (int)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

// 2) Parse model (ONNX example)
auto parser = nvonnxparser::createParser(*network, gLogger);
parser->parseFromFile("model.onnx", 0);

// 3) Configure builder
auto config = builder->createBuilderConfig();
config->setMaxWorkspaceSize(1 << 30);      // 1GB workspace
config->setFlag(BuilderFlag::kFP16);       // Enable FP16
config->setFlag(BuilderFlag::kINT8);       // Enable INT8 (with calibrator if needed)

// 4) Build engine
auto engine = builder->buildEngineWithConfig(*network, *config);
if (!engine) { std::cerr << "Engine build failed\n"; }
```

This will apply optimizations like fusing Conv+ReLU, enabling tensor cores for eligible layers, and quantizing weights for INT8 if calibrator is provided.

### Runtime Execution

```cpp
// Deserialize engine (or load from file)
auto runtime = createInferRuntime(gLogger);
auto engine  = runtime->deserializeCudaEngine(engineData, size);
auto context = engine->createExecutionContext();

// Allocate GPU buffers for inputs/outputs
std::vector<void*> buffers(engine->getNbBindings());
for (int i = 0; i < engine->getNbBindings(); ++i) {
    size_t bytes = volume(engine->getBindingDimensions(i)) * sizeof(float);
    cudaMalloc(&buffers[i], bytes);
}
// Copy input data to GPU (omitted)
// Run inference
context->executeV2(buffers.data());
// Copy outputs back (omitted)
```

`executeV2` runs the optimized engine. TensorRT automatically batches/schedules kernels for maximal throughput.

### PyTorch Integration

PyTorch’s `torch_tensorrt` can compile a PyTorch model to TensorRT:

```python
import torch
import torch_tensorrt as trt

model = torch.load('model.pth').eval().cuda()
trt_model = trt.compile(model,
    inputs=[trt.Input((1,3,224,224))],
    enabled_precisions={torch.float16, torch.int8},
    workspace_size=1<<30)
torch.save(trt_model, 'model_trt.pth')
```

This applies layer fusion and quantization to produce a faster inference model.

### Best Practices

* **Profile & Optimize Early**: Use tools like **nsight** or PyTorch Profiler to identify bottlenecks.
* **Batch & Fuse**: Larger batches and fused operations (Conv+BN+ReLU, etc.) improve throughput.
* **Precision**: Try FP16 or INT8 for inference to reduce latency/size.
* **Platform**: TensorRT excels in deployment/inference; for training, use PyTorch with `torch.compile` or TensorFlow.
* **Use Proper Layer**: For custom ops, CUTLASS; for standard GEMMs, cuBLAS; for NN layers, cuDNN; for full model inference, TensorRT.

