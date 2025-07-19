import triton
import triton.language as tl

BLOCK_SIZE = 256

@triton.jit
def vector_add_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a + b, mask=mask)

@triton.jit
def vector_sub_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a - b, mask=mask)

@triton.jit
def vector_mul_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a * b, mask=mask)

@triton.jit
def vector_div_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a / b, mask=mask)

@triton.jit
def vector_pow_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, tl.pow(a, b), mask=mask)

@triton.jit
def vector_max_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, tl.max(a, b), mask=mask)

@triton.jit
def vector_min_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, tl.min(a, b), mask=mask)

@triton.jit
def vector_eq_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offs < N
    result = tl.load(A_ptr + offs, mask=mask) == tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, result.to(tl.float32), mask=mask)

@triton.jit
def vector_neq_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offs < N
    result = tl.load(A_ptr + offs, mask=mask) != tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, result.to(tl.float32), mask=mask)

@triton.jit
def vector_gt_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offs < N
    result = tl.load(A_ptr + offs, mask=mask) > tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, result.to(tl.float32), mask=mask)

@triton.jit
def vector_ge_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offs < N
    result = tl.load(A_ptr + offs, mask=mask) >= tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, result.to(tl.float32), mask=mask)

@triton.jit
def vector_lt_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offs < N
    result = tl.load(A_ptr + offs, mask=mask) < tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, result.to(tl.float32), mask=mask)

@triton.jit
def vector_le_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = offs < N
    result = tl.load(A_ptr + offs, mask=mask) <= tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, result.to(tl.float32), mask=mask)

@triton.jit
def vector_and_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a & b, mask=mask)

@triton.jit
def vector_or_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a | b, mask=mask)

@triton.jit
def vector_xor_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a ^ b, mask=mask)

@triton.jit
def vector_fma_kernel(A_ptr, B_ptr, D_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    a     = tl.load(A_ptr + offs, mask=mask)
    b     = tl.load(B_ptr + offs, mask=mask)
    d     = tl.load(D_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, tl.fma(a, b, d), mask=mask)

@triton.jit
def vector_clip_kernel(A_ptr, min_val, max_val, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    v    = tl.load(A_ptr + offs, mask=mask)
    tl.store(C_ptr + offs,
             tl.where(v < min_val, min_val, tl.where(v > max_val, max_val, v)),
             mask=mask)

# Unary ops (A → C)

@triton.jit
def vector_copy_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.load(A_ptr + offs, mask=mask), mask=mask)

@triton.jit
def vector_neg_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, -tl.load(A_ptr + offs, mask=mask), mask=mask)

@triton.jit
def vector_abs_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.abs(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_exp_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.exp(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_log_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.log(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_sqrt_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.sqrt(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_rsqrt_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.rsqrt(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_sin_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.sin(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_cos_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.cos(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_tan_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.tan(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_tanh_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.tanh(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_sigmoid_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x    = tl.load(A_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, 1 / (1 + tl.exp(-x)), mask=mask)

@triton.jit
def vector_relu_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x    = tl.load(A_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, tl.where(x > 0, x, 0), mask=mask)

@triton.jit
def vector_leaky_relu_kernel(A_ptr, alpha, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x    = tl.load(A_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, tl.where(x > 0, x, alpha * x), mask=mask)

@triton.jit
def vector_gelu_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x    = tl.load(A_ptr + offs, mask=mask)
    tl.store(C_ptr + offs,
             0.5 * x * (1 + tl.erf(x / tl.sqrt(2.0))),
             mask=mask)

@triton.jit
def vector_floor_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.floor(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_ceil_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.ceil(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_round_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.round(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_trunc_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, tl.trunc(tl.load(A_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def vector_sign_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x    = tl.load(A_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, tl.sign(x), mask=mask)

@triton.jit
def vector_reciprocal_kernel(A_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    tl.store(C_ptr + offs, 1 / tl.load(A_ptr + offs, mask=mask), mask=mask)


# -------------------------------------------
# Python wrapper functions
# -------------------------------------------

def solve_add(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_add_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_sub(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_sub_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_mul(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_mul_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_div(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_div_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_pow(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_pow_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_max(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_max_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_min(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_min_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_eq(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_eq_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_neq(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_neq_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_gt(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_gt_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_ge(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_ge_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_lt(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_lt_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_le(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_le_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_and(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_and_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_or(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_or_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_xor(A, B, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_xor_kernel[grid](A, B, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_fma(A, B, D, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_fma_kernel[grid](A, B, D, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_clip(A, min_val, max_val, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_clip_kernel[grid](A, min_val, max_val, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_copy(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_copy_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_neg(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_neg_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_abs(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_abs_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_exp(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_exp_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_log(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_log_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_sqrt(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_sqrt_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_rsqrt(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_rsqrt_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_sin(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_sin_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_cos(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_cos_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_tan(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_tan_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_tanh(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_tanh_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_sigmoid(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_sigmoid_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_relu(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_relu_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_leaky_relu(A, alpha, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_leaky_relu_kernel[grid](A, alpha, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_gelu(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_gelu_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_floor(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_floor_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_ceil(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_ceil_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_round(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_round_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_trunc(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_trunc_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_sign(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_sign_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)

def solve_reciprocal(A, C):
    n    = A.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_reciprocal_kernel[grid](A, C, n, BLOCK_SIZE=BLOCK_SIZE)
