import torch
from typing import Any, List, Literal, Optional, Tuple, Union

from . import _fused
from . import _qattn_sm90

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
def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    is_causal: bool = False,
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

    smooth_k = True
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3  # "per_thread"
    _return_lse = 0

    dtype = q.dtype
    assert SM90_ENABLED, "SM90 kernel is not available. Make sure you GPUs with compute capability 9.0."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)
    head_dim_og = q.size(-1)

    if head_dim_og < 64 or head_dim_og > 128: raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    if head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None: sm_scale = head_dim_og**-0.5
    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k: km = k.mean(dim=seq_dim, keepdim=True)
    else:        km = None

    q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128)
    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    # pad v to multiple of 128
    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        tmp = torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3), dtype=v.dtype, device=v.device)
        v = torch.cat([v, tmp], dim=2)

    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)

    _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]
    return o
