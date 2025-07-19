#include <metal_stdlib>
using namespace metal;

constexpr uint THREADS_PER_THREADGROUP = 256;

kernel void vector_add(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = A[gid] + B[gid];
    }
}

kernel void vector_sub(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = A[gid] - B[gid];
    }
}

kernel void vector_mul(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = A[gid] * B[gid];
    }
}

kernel void vector_div(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = A[gid] / B[gid];
    }
}

kernel void vector_pow(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = pow(A[gid], B[gid]);
    }
}

kernel void vector_max(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = max(A[gid], B[gid]);
    }
}

kernel void vector_min(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = min(A[gid], B[gid]);
    }
}

kernel void vector_eq(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = (A[gid] == B[gid]) ? 1.0f : 0.0f;
    }
}

kernel void vector_neq(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = (A[gid] != B[gid]) ? 1.0f : 0.0f;
    }
}

kernel void vector_gt(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = (A[gid] >  B[gid]) ? 1.0f : 0.0f;
    }
}

kernel void vector_ge(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = (A[gid] >= B[gid]) ? 1.0f : 0.0f;
    }
}

kernel void vector_lt(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = (A[gid] <  B[gid]) ? 1.0f : 0.0f;
    }
}

kernel void vector_le(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = (A[gid] <= B[gid]) ? 1.0f : 0.0f;
    }
}

kernel void vector_and(
    device const uint*  A       [[ buffer(0) ]],
    device const uint*  B       [[ buffer(1) ]],
    device       uint*  C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = A[gid] & B[gid];
    }
}

kernel void vector_or(
    device const uint*  A       [[ buffer(0) ]],
    device const uint*  B       [[ buffer(1) ]],
    device       uint*  C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = A[gid] | B[gid];
    }
}

kernel void vector_xor(
    device const uint*  A       [[ buffer(0) ]],
    device const uint*  B       [[ buffer(1) ]],
    device       uint*  C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = A[gid] ^ B[gid];
    }
}

kernel void vector_fma(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device const float* D       [[ buffer(2) ]],
    device       float* C       [[ buffer(3) ]],
    constant     uint&  N       [[ buffer(4) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = fma(A[gid], B[gid], D[gid]);
    }
}

kernel void vector_clip(
    device const float* A       [[ buffer(0) ]],
    constant     float& minVal  [[ buffer(1) ]],
    constant     float& maxVal  [[ buffer(2) ]],
    device       float* C       [[ buffer(3) ]],
    constant     uint&  N       [[ buffer(4) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        float v = A[gid];
        C[gid]  = (v < minVal ? minVal : (v > maxVal ? maxVal : v));
    }
}

// Unary ops (A → C)

kernel void vector_copy(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = A[gid];
    }
}

kernel void vector_neg(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = -A[gid];
    }
}

kernel void vector_abs(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = fabs(A[gid]);
    }
}

kernel void vector_exp(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = exp(A[gid]);
    }
}

kernel void vector_log(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = log(A[gid]);
    }
}

kernel void vector_sqrt(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = sqrt(A[gid]);
    }
}

kernel void vector_rsqrt(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = 1.0f / sqrt(A[gid]);
    }
}

kernel void vector_sin(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = sin(A[gid]);
    }
}

kernel void vector_cos(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = cos(A[gid]);
    }
}

kernel void vector_tan(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = tan(A[gid]);
    }
}

kernel void vector_tanh(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = tanh(A[gid]);
    }
}

kernel void vector_sigmoid(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = 1.0f / (1.0f + exp(-A[gid]));
    }
}

kernel void vector_relu(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = max(A[gid], 0.0f);
    }
}

kernel void vector_leaky_relu(
    device const float* A       [[ buffer(0) ]],
    constant     float& alpha   [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  N       [[ buffer(3) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        float x = A[gid];
        C[gid]  = (x > 0.0f ? x : alpha * x);
    }
}

kernel void vector_gelu(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        float x = A[gid];
        C[gid]  = 0.5f * x * (1.0f + erf(x / sqrt(2.0f)));
    }
}

kernel void vector_floor(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = floor(A[gid]);
    }
}

kernel void vector_ceil(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = ceil(A[gid]);
    }
}

kernel void vector_round(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = round(A[gid]);
    }
}

kernel void vector_trunc(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = trunc(A[gid]);
    }
}

kernel void vector_sign(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = (A[gid] > 0.0f ? 1.0f : (A[gid] < 0.0f ? -1.0f : 0.0f));
    }
}

kernel void vector_reciprocal(
    device const float* A       [[ buffer(0) ]],
    device       float* C       [[ buffer(1) ]],
    constant     uint&  N       [[ buffer(2) ]],
    uint          gid           [[ thread_position_in_grid ]])
{
    if (gid < N) {
        C[gid] = 1.0f / A[gid];
    }
}
