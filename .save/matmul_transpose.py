# Modded from github.com/nil0x9/flash-muon

import torch, triton
import triton.language as tl
from torch import Tensor

###############################
##  matmul_transpose_triton  ##
###############################
mmt_cfgs = [ triton.Config({'BLOCK_SIZE_M': m, 'BLOCK_SIZE_K': k, 'GROUP_SIZE_M': g}, num_stages=s, num_warps=w)
        for m in [32, 64, 128]  for k in [32, 64]  for g in [8]  for s in [3, 4, 5]  for w in [4, 8] ]
@triton.autotune(configs=mmt_cfgs, key=['M', 'K'],)
@triton.jit
def matmul_transpose_kernel(
        x, y, M, K,
        stride_xm, stride_xk,
        stride_ym, stride_yn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    '''
    pid_m: Block row index
    pid_n: Block column index
        0   1   2   3  ← pid_n (column)
    ┌─────────────────
    0 │ ✅  ✅  ✅  ✅     
    1 │ ❌  ✅  ✅  ✅   
    2 │ ❌  ❌  ✅  ✅   
    ↑ pid_m (row)
    ✅ = Compute (pid_m ≤ pid_n)  
    ❌ = Skip    (pid_m > pid_n)
=>  Bỏ qua những khối ở tam giác phía dưới trong ma trận '''
    if pid_m > pid_n: return

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # we use a & b ptrs to denote different rows of x.
    a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk) 
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        # accumulator += a @ b^T
        accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_xk

    # https://github.com/triton-lang/triton/issues/2252 
    # Vì .to(x.dtype) Có thể fail với some input types!
    c = accumulator.to(x.dtype.element_ty) # <= Triton-compatible type descriptor

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)

    # Store upper triangular
    tl.store(c_ptrs, c, mask=c_mask)
    if pid_m == pid_n: return # đường biên không cần mirror

    # MIRROR TO LOWER TRIANGULAR bằng cách transpose and copy
    # c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :] # <= upper
    ct_ptrs  = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :] # <= lower
    ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(ct_ptrs, tl.permute(c, (1,0)), mask=ct_mask)
    # Transpose kết quả ^^^^^^^^^^^^^^^^^ 


def matmul_transpose_assign(d_in, d_out):
    assert d_in.is_cuda, "Input `d_in` must be a CUDA tensor"
    assert d_out.is_cuda, "Input `d_out` must be a CUDA tensor"
    assert d_in.device == d_out.device, "Inputs `d_in` and `d_out` must be on the same CUDA device"
    assert d_in.dtype == d_out.dtype, "Inputs must have the same data type"
    assert d_in.ndim == 2, "Input `d_in` must be a 2D tensor"
    assert d_out.ndim == 2, "Input `d_out` must be a 2D tensor"
    assert d_in.size(0) == d_out.size(0) == d_out.size(0), \
            "First dimension of `d_in` must match first and second dimension of `d_out`"

    d_in = d_in.contiguous()
    M, K = d_in.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    matmul_transpose_kernel[grid](
        d_in, d_out, M, K,
        d_in.stride(0),  d_in.stride(1),
        d_out.stride(0), d_out.stride(1)
    )

###############################
##  matmul_transpose_int8    ##
###############################

@triton.autotune(configs=mmt_cfgs, key=['M', 'K'])
@triton.jit
def int8_matmul_transpose_kernel(
    x, y, scale_ptr, M, K,  # 🔧 scale_ptr thay vì scalar
    stride_xm, stride_xk,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # 🔄 REUSE: Toàn bộ program ID logic từ original
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    if pid_m > pid_n: return

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk) 

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        accumulator += tl.dot(a, tl.permute(b, (1, 0)))
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_xk

    # 🔧 MODIFY: Load scale và square it
    scale = tl.load(scale_ptr).to(tl.float32)
    scale_squared = scale * scale
    c = accumulator.to(tl.float32) * scale_squared

    # 🔄 REUSE: Store logic hoàn toàn từ original
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)

    if pid_m < pid_n:
        ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1,0)), mask=ct_mask)


@torch.no_grad()
def quantize_int8(tensor: Tensor, dim=-1, eps=1e-12, sr=False) -> Tensor:
    ''' absmax symmetric quantization, clip(cận_dưới_eps) tránh chia cho 0 '''
    scale = tensor.abs().amax(dim, keepdim=True) / 127 # same dtype
    inv_scale = 1.0 / scale.float().clip(eps)       # little bit faster than 
    tensor = tensor.float() * inv_scale.view(-1, 1) # tensor / scale.clip(eps)
    if sr: tensor = (tensor + torch.rand_like(tensor)).floor()
    else:  tensor = tensor.round()# ^^^stochastic rounding^^^^
    return ( tensor.clip(-128, 127).to(torch.int8), scale )


def int8_matmul_transpose_assign(x, output):
    assert x.is_cuda and output.is_cuda
    assert x.device == output.device
    assert x.dtype == output.dtype
    M, K = x.shape
    assert output.shape == (M, M)

    x_int8, scale = quantize_int8(x, dim=None, sr=True)
    assert x_int8.dtype == torch.int8
    assert x_int8.ndim == 2 and output.ndim == 2
    assert scale.numel() == 1, "Only per-tensor quantization supported for transpose. Per-row quantization would be complex for X @ X^T"
    
    x_int8 = x_int8.contiguous()    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(M, META['BLOCK_SIZE_M']),)

    int8_matmul_transpose_kernel[grid](
        x_int8, output, scale, M, K,
        x_int8.stride(0), x_int8.stride(1),
        output.stride(0), output.stride(1)
    )


def _newtonschulz(G: Tensor, steps: int, fast=False) -> Tensor:
    # G: The gradient or momentum matrix to be orthogonalized.
    # steps: Number of Newton-Schulz iterations.
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    if fast and G.ndim == 2:
        A   = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
        AxA = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)

        for _ in range(steps):        
            matmul_transpose_assign(X, A)
            matmul_transpose_assign(A, AxA)
            B = b * A + c * AxA
            X = a * X + B @ X
    else:
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X

    if G.size(-2) > G.size(-1): X = X.mT
    return X


#######################################
## 🎯 Usage với quantize_int8()      ##
#######################################

def test_int8_matmul_transpose_assign():
    torch.manual_seed(42)
    M, K = 1024, 768
    device = 'cuda'
    
    X = torch.randn(M, K, device=device, dtype=torch.float32)
    ref_result = X @ X.T
    
    int8_result = torch.zeros(M, M, device=device, dtype=torch.float32)
    int8_matmul_transpose_assign(X, int8_result)
    
    abs_error = torch.norm(ref_result - int8_result)
    rel_error = abs_error / torch.norm(ref_result)
    
    print(f"Relative error for int8_matmul_transpose: {rel_error:.6f}")    
    return rel_error < 1e-2


def newton_schulz_int8_example():
    """Example trong Newton-Schulz context"""
    W = torch.randn(2048, 1024, device='cuda', dtype=torch.float32)
    
    # Traditional FP32 Newton-Schulz
    G_fp32 = W @ W.T
    
    G_int8 = torch.zeros_like(G_fp32)
    int8_matmul_transpose_assign(W, G_int8)
    
    print(f"Matrix difference: {torch.norm(G_fp32 - G_int8) / torch.norm(G_fp32):.6f}")
    
    # Newton-Schulz iteration test
    def newton_schulz_step(G):
        return 1.5 * torch.eye(G.shape[0], device=G.device) - 0.5 * G @ G
    
    # Compare one iteration
    step_fp32 = newton_schulz_step(G_fp32 / torch.trace(G_fp32) * G_fp32.shape[0])
    step_int8 = newton_schulz_step(G_int8 / torch.trace(G_int8) * G_int8.shape[0])
    
    print(f"Newton-Schulz step difference: {torch.norm(step_fp32 - step_int8) / torch.norm(step_fp32):.6f}")


if __name__ == "__main__":
    test_int8_matmul_transpose_assign()
    newton_schulz_int8_example()
