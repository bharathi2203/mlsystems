#include <cuda_runtime.h>
#include <cmath>

constexpr int THREADS_PER_BLOCK = 256;

// Binary ops (A, B → C)
template<typename T>
__global__ void vector_add(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

template<typename T>
__global__ void vector_sub(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] - B[idx];
}

template<typename T>
__global__ void vector_mul(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] * B[idx];
}

template<typename T>
__global__ void vector_div(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] / B[idx];
}

template<typename T>
__global__ void vector_pow(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::pow(A[idx], B[idx]);
}

template<typename T>
__global__ void vector_max(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] > B[idx] ? A[idx] : B[idx];
}

template<typename T>
__global__ void vector_min(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] < B[idx] ? A[idx] : B[idx];
}

template<typename T>
__global__ void vector_eq(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = (A[idx] == B[idx]);
}

template<typename T>
__global__ void vector_neq(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = (A[idx] != B[idx]);
}

template<typename T>
__global__ void vector_gt(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = (A[idx] > B[idx]);
}

template<typename T>
__global__ void vector_ge(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = (A[idx] >= B[idx]);
}

template<typename T>
__global__ void vector_lt(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = (A[idx] < B[idx]);
}

template<typename T>
__global__ void vector_le(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = (A[idx] <= B[idx]);
}

template<typename T>
__global__ void vector_and(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] & B[idx];
}

template<typename T>
__global__ void vector_or(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] | B[idx];
}

template<typename T>
__global__ void vector_xor(const T* A, const T* B, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] ^ B[idx];
}

template<typename T>
__global__ void vector_fma(const T* A, const T* B, const T* D, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::fma(A[idx], B[idx], D[idx]);
}

template<typename T>
__global__ void vector_clip(const T* A, T min_val, T max_val, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T v = A[idx];
        C[idx] = v < min_val ? min_val : (v > max_val ? max_val : v);
    }
}

// Unary ops (A → C)
template<typename T>
__global__ void vector_copy(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx];
}

template<typename T>
__global__ void vector_neg(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = -A[idx];
}

template<typename T>
__global__ void vector_abs(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::fabs(A[idx]);
}

template<typename T>
__global__ void vector_exp(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::exp(A[idx]);
}

template<typename T>
__global__ void vector_log(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::log(A[idx]);
}

template<typename T>
__global__ void vector_sqrt(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::sqrt(A[idx]);
}

template<typename T>
__global__ void vector_rsqrt(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = T(1) / std::sqrt(A[idx]);
}

template<typename T>
__global__ void vector_sin(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::sin(A[idx]);
}

template<typename T>
__global__ void vector_cos(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::cos(A[idx]);
}

template<typename T>
__global__ void vector_tan(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::tan(A[idx]);
}

template<typename T>
__global__ void vector_tanh(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::tanh(A[idx]);
}

template<typename T>
__global__ void vector_sigmoid(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T x = A[idx];
        C[idx] = T(1) / (T(1) + std::exp(-x));
    }
}

template<typename T>
__global__ void vector_relu(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] > T(0) ? A[idx] : T(0);
}

template<typename T>
__global__ void vector_leaky_relu(const T* A, T alpha, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T x = A[idx];
        C[idx] = x > T(0) ? x : alpha * x;
    }
}

template<typename T>
__global__ void vector_gelu(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T x = A[idx];
        C[idx] = T(0.5) * x * (T(1) + std::erf(x / std::sqrt(T(2))));
    }
}

template<typename T>
__global__ void vector_floor(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::floor(A[idx]);
}

template<typename T>
__global__ void vector_ceil(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::ceil(A[idx]);
}

template<typename T>
__global__ void vector_round(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::round(A[idx]);
}

template<typename T>
__global__ void vector_trunc(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = std::trunc(A[idx]);
}

template<typename T>
__global__ void vector_sign(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = T(A[idx] > T(0)) - T(A[idx] < T(0));
}

template<typename T>
__global__ void vector_reciprocal(const T* A, T* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = T(1) / A[idx];
}

// -------------------------------------------
// Host‐side launch wrappers
// -------------------------------------------

// Binary launches
template<typename T> void solve_add    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_add  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_sub    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_sub  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_mul    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_mul  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_div    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_div  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_pow    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_pow  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_max    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_max  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_min    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_min  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_eq     (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_eq   <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_neq    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_neq  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_gt     (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_gt   <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_ge     (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_ge   <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_lt     (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_lt   <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_le     (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_le   <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_and    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_and  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_or     (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_or   <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_xor    (const T* A,const T* B, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_xor  <T><<<b,THREADS_PER_BLOCK>>>(A,B,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_fma    (const T* A,const T* B,const T* D, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_fma  <T><<<b,THREADS_PER_BLOCK>>>(A,B,D,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_clip   (const T* A, T minv, T maxv, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_clip <T><<<b,THREADS_PER_BLOCK>>>(A,minv,maxv,C,N); cudaDeviceSynchronize();}

// Unary launches
template<typename T> void solve_copy      (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_copy      <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_neg       (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_neg       <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_abs       (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_abs       <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_exp       (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_exp       <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_log       (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_log       <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_sqrt      (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_sqrt      <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_rsqrt     (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_rsqrt     <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_sin       (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_sin       <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_cos       (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_cos       <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_tan       (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_tan       <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_tanh      (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_tanh      <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_sigmoid   (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_sigmoid   <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_relu      (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_relu      <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_leaky_relu(const T* A, T a, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_leaky_relu<T><<<b,THREADS_PER_BLOCK>>>(A,a,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_gelu      (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_gelu      <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_floor     (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_floor     <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_ceil      (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_ceil      <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_round     (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_round     <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_trunc     (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_trunc     <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_sign      (const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_sign      <T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
template<typename T> void solve_reciprocal(const T* A, T* C,int N){int b=(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK; vector_reciprocal<T><<<b,THREADS_PER_BLOCK>>>(A,C,N); cudaDeviceSynchronize();}
