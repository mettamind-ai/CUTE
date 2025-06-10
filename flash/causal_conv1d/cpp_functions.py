# Copyright (c) 2024, Tri Dao.

####################
# Import C extension
####################

import os, time, psutil, torch.utils.cpp_extension
from pathlib import Path

free_memory_gb = round(psutil.virtual_memory().available / (1024 ** 3))
if not os.environ.get("MAX_JOBS"):
    max_jobs = int(free_memory_gb / 5)
    os.environ["MAX_JOBS"] = str(max_jobs)
print(f"causal_conv1d: free_memory_gb {free_memory_gb}, max_jobs {os.environ['MAX_JOBS']}")

NVCC_FLAGS = [
    "-O3", "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--threads=8", # "-diag-suppress=174", # suppress the specific warning
]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += ["-gencode", f"arch=compute_86,code=sm_86"]
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6;8.9" # RTX 30xx, 40xx

abspath = Path(__file__).parent
started_at = time.time()

causal_conv1d_cuda = torch.utils.cpp_extension.load(
    "CUTE_causal_conv1d_cuda",
    sources=[
        abspath / "causal_conv1d.cpp",
        abspath / "causal_conv1d_fwd.cu",
        abspath / "causal_conv1d_bwd.cu",
        abspath / "causal_conv1d_update.cu", # for inference  
    ],
    extra_cuda_cflags=NVCC_FLAGS,
    extra_include_paths=[str(abspath)],
)
# ~/.cache/torch_extensions/py310_cu126/CUTE_flash_attn_2_cuda/
print(f"causal_conv1d: DONE. In {int(time.time() - started_at)} seconds.")
#########################################################################


import torch
LIBRARY_NAME = "DaoAILab"

@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_fwd_cpp", mutates_args={"out", "final_states_out"})
def _causal_conv1d_fwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    out: torch.Tensor,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> None:
    causal_conv1d_cuda.causal_conv1d_fwd(
        x,
        weight,
        bias,
        seq_idx,
        initial_states,
        out,
        final_states_out,
        silu_activation,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_bwd_cpp", mutates_args={
    "dfinal_states",
    "dx",
    "dweight",
    "dbias",
    "dinitial_states",
})
def _causal_conv1d_bwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    dbias: torch.Tensor | None,
    dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> None:
    causal_conv1d_cuda.causal_conv1d_bwd(
        x,
        weight,
        bias,
        dout,
        seq_idx,
        initial_states,
        dfinal_states,
        dx,
        dweight,
        dbias,
        dinitial_states,
        silu_activation,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_update_cpp", mutates_args={"out", "conv_state"})
def _causal_conv1d_update_cpp(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    silu_activation: bool,
    cache_seqlens: torch.Tensor | None,
    conv_state_indices: torch.Tensor | None,
) -> None:
    causal_conv1d_cuda.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        out,
        silu_activation,
        cache_seqlens,
        conv_state_indices
    )


def causal_conv1d_fwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _causal_conv1d_fwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        out=out,
        final_states_out=final_states_out,
        silu_activation=silu_activation,
    )
    return out


def causal_conv1d_bwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor | None,
    return_dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor | None]:
    batch_size, dim = x.size()[:2]
    width = weight.size(-1)

    if dx is None:
        dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=weight.dtype)
    dbias = torch.zeros_like(bias, dtype=bias.dtype) if bias is not None else None
    dinitial_states = None
    if return_dinitial_states:
        dinitial_states = torch.empty(batch_size, width - 1, dim, device=x.device, dtype=x.dtype).transpose(1, 2)

    _causal_conv1d_bwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        dout=dout,
        seq_idx=seq_idx,
        initial_states=initial_states,
        dfinal_states=dfinal_states,
        dx=dx,
        dweight=dweight,
        dbias=dbias,
        dinitial_states=dinitial_states,
        silu_activation=silu_activation,
    )

    dweight = dweight.type_as(weight)
    if bias is not None: dbias = dbias.type_as(bias)
    return dx, dweight, dbias, dinitial_states


def causal_conv1d_update_function(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    silu_activation: bool,
    cache_seqlens: torch.Tensor | None,
    conv_state_indices: torch.Tensor | None,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _causal_conv1d_update_cpp(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        out=out,
        silu_activation=silu_activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )
    return out
