import torch, os
from torch import Tensor
from pathlib import Path
import torch.utils.cpp_extension

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6;8.9"  # 3050ti, 4090
os.environ['MAX_JOBS'] = "4"

# lib = torch.library.Library("sageattn", "DEF")
# lib_ops = torch.ops.sageattn
# _cutlass_mm = torch.utils.cpp_extension.load(
#     "cutlass_mm",
#     sources=["cutlass_mm.cu"],
#     extra_cuda_cflags=["-O3"],
#     extra_include_paths=["third-party/cutlass/include"],
#     verbose=True,
# )

NVCC_FLAGS = ["-O3", "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math", "--threads=8",
    "-Xptxas=-v", "-diag-suppress=174", # suppress the specific warning
]
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += ["-gencode", f"arch=compute_{89},code=sm_{89}"]
NVCC_FLAGS += ["-gencode", f"arch=compute_{89},code=compute_{89}"]

_qattn_sm89 = torch.utils.cpp_extension.load(
    "_qattn_sm89",
    sources=[
        "csrc/qattn/pybind_sm89.cpp",
        "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",
    ],
    extra_cuda_cflags=NVCC_FLAGS,
)

_fused = torch.utils.cpp_extension.load(
    "_fused",
    sources=[
        "csrc/fused/pybind.cpp",
        "csrc/fused/fused.cu",
    ],
    extra_cuda_cflags=NVCC_FLAGS,
)

_tensor_layout = 0 # "NHD"

def per_warp_int8(
    q: torch.Tensor, 
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int =128,
    WARPQ: int =32,
    BLKK: int =64,
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    b, h_qo, qo_len, head_dim = q.shape
    _, h_kv, kv_len, _ = k.shape
    
    q_scale = torch.empty((b, h_qo, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, BLKQ, WARPQ, _tensor_layout)

    if km is not None:
            km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
            _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:   _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)
    
    return q_int8, q_scale, k_int8, k_scale


def per_channel_fp8(
    v: torch.Tensor,
    scale_max: float = 448.0,
):
    b, h_kv, kv_len, head_dim = v.shape
    padded_len = (kv_len + 63) // 64 * 64
    v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
    
    _fused.transpose_pad_permute_cuda(v, v_transposed_permutted, _tensor_layout)
    v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)

    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
    vm = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)

    _fused.mean_scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, vm, v_scale, kv_len, scale_max, _tensor_layout)
    return v_fp8, v_scale, vm


@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_cuda(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
CUDA SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation.
- ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
- The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``

Parameters
----------
q : torch.Tensor ``[batch_size, num_qo_heads, qo_len, head_dim]``.
k : torch.Tensor ``[batch_size, num_kv_heads, kv_len, head_dim]``.
v : torch.Tensor ``[batch_size, num_kv_heads, kv_len, head_dim]``.
is_causal : bool Only applicable when qo_len == kv_len. Default: False.
sm_scale : Optional[float]. If not provided, will be set to ``1.0 / sqrt(head_dim)``.

Returns
-------
torch.Tensor ``[batch_size, num_qo_heads, qo_len, head_dim]``.
    """
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3  # "per_thread"
    _return_lse = 0

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)
    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    km = k.mean(dim=seq_dim, keepdim=True)

    q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)
    _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]
    return o
