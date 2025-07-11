import math
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

# from bert_padding import pad_input, unpad_input
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None

index_first_axis = IndexFirstAxis.apply

class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None

index_put_first_axis = IndexPutFirstAxis.apply

def pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[-1]
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

def unpad_input(hidden_states, attention_mask, unused_mask=None):
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch, 
    )


is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

def _get_block_size_n(device, head_dim, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32: return 128
    if head_dim <= 64: return 128
    elif head_dim <= 96: return 64
    elif head_dim <= 128:
        if is_sm8x: return 64 if is_causal else 32
        else:       return 64
    elif head_dim <= 160:
        if is_sm8x: return 64
        else:       return 32
    elif head_dim <= 256: return 64


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    dq_pad_fn = output_pad_fn
    if key_padding_mask is not None:
        dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
    else:
        dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
    return (
        q_unpad.detach().requires_grad_(),
        k_unpad.detach().requires_grad_(),
        v_unpad.detach().requires_grad_(),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q.detach().requires_grad_(),
        k.detach().requires_grad_(),
        v.detach().requires_grad_(),
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q, k, v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
    ],
)
@pytest.mark.parametrize("softcap", [0.0, 50.0])
def test_flash_attn_output(
    seqlen_q, seqlen_k, d, causal, local, deterministic, mha_type, dtype, softcap
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 6 if softcap == 0.0 else 4  # softcap reference impl takes more memory
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if softcap > 0:
        # Ensure the values of qk are at least within softcap range.
        q = q * softcap
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)

    out = flash_attn_func(q, k, v, causal=causal, window_size=window_size, softcap=softcap, deterministic=deterministic)
    out_ref = attention_ref(q, k, v, None, None, None, causal, window_size, softcap)
    out_pt  = attention_ref(q, k, v, None, None, None, causal, window_size, softcap, upcast=False, reorder_ops=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)

    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
    print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
    print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
    assert (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
    assert (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()


@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 147),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
    ],
)
@pytest.mark.parametrize("softcap", [0.0, 50.0])
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, causal, local, deterministic, mha_type, dtype, softcap
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 6 if softcap == 0.0 else 4  # softcap reference impl takes more memory
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if softcap > 0:
        # Ensure the values of qk are at least within softcap range.
        q = q * softcap

    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True)

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        q, k, v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)

    out_unpad = flash_attn_varlen_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        deterministic=deterministic,
    )
    out = output_pad_fn(out_unpad)

    out_ref = attention_ref(q, k, v,
        query_padding_mask, key_padding_mask, None,
        causal=causal, window_size=window_size, softcap=softcap,
    )

    out_pt = attention_ref(q, k, v,
        query_padding_mask, key_padding_mask, None,
        causal=causal, window_size=window_size, softcap=softcap,
        upcast=False, reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
    dk = dk_pad_fn(dk_unpad)
    dv = dk_pad_fn(dv_unpad)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)
    dq = dq_pad_fn(dq_unpad)
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
    print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
    print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
    assert (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
    assert (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
    ],
)
def test_flash_attn_causal(seqlen_q, seqlen_k, swap_sq_sk, d, local, dtype):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    causal = True
    torch.random.manual_seed(0)
    batch_size = 8
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    out_ref = attention_ref(q, k, v, None, None, None, causal, window_size)
    out_pt  = attention_ref(q, k, v, None, None, None, causal, window_size, upcast=False, reorder_ops=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    dq,     dk,     dv     = torch.autograd.grad(out,     (q, k, v), g)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
    dq_pt,  dk_pt,  dv_pt  = torch.autograd.grad(out_pt,  (q, k, v), g)

    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
    print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
    print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + 1e-5
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + 1e-5
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
    ],
)
# TODO: add smaller page sizes when https://github.com/Dao-AILab/flash-attention/pull/824 is merged
@pytest.mark.parametrize("paged_kv_block_size", [None, 256, 512])
def test_flash_attn_varlen_causal(
    seqlen_q, seqlen_k, swap_sq_sk, d, local, paged_kv_block_size, dtype
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    causal = True
    torch.random.manual_seed(0)
    batch_size = 8
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)

    if paged_kv_block_size is None:
        k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
        block_table = None
    else:
        k, v, block_table, k_cache_paged, v_cache_paged, num_blocks = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads, d, device, dtype
        )
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)
    out_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad if paged_kv_block_size is None else k_cache_paged,
        v_unpad if paged_kv_block_size is None else v_cache_paged,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
    )
    out = output_pad_fn(out_unpad)
    out_ref = attention_ref(q, k, v,
        query_padding_mask,
        key_padding_mask,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt = attention_ref(q, k, v,
        query_padding_mask,
        key_padding_mask,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    test_backward = block_table is None
    if test_backward:
        (
            dq_unpad,
            dk_unpad,
            dv_unpad,
        ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    if test_backward:
        assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + 1e-5
        assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + 1e-5
        assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 339),
        (799, 1),
        (3, 1024),
        (64, 800),
        (128, 128),
        (256, 256),
        (2, 255),
    ],
)
def test_flash_attn_splitkv(
    seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, deterministic, dtype
):
    if swap_sq_sk and seqlen_q != seqlen_k:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 1
    nheads = 12
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=causal, window_size=window_size, deterministic=deterministic)
    out_ref = attention_ref(q, k, v, None, None, None, causal, window_size)
    out_pt = attention_ref(q, k, v, None, None, None, causal, window_size, upcast=False, reorder_ops=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
    print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
    print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + 2e-4
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + 2e-4
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + 2e-4

 
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [0])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("rotary_interleaved", [False])
@pytest.mark.parametrize("rotary_fraction", [0.0])
@pytest.mark.parametrize("paged_kv_block_size", [None, 256])
@pytest.mark.parametrize("has_leftpad", [False])
# @pytest.mark.parametrize("has_batch_idx", [False, True])
@pytest.mark.parametrize("has_batch_idx", [False])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        (64, 256),
        (64, 2048),
        (128, 128),
    ],
)
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    has_leftpad,
    paged_kv_block_size,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    mha_type,
    num_splits,
    dtype,
):
    if seqlen_q > seqlen_k and new_kv:                      pytest.skip()
    if not new_kv and rotary_fraction > 0.0:                pytest.skip()
    if has_batch_idx and paged_kv_block_size is not None:   pytest.skip()
    if has_leftpad and paged_kv_block_size is not None:     pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 2
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        block_table = None
    else:
        (
            k_cache,
            v_cache,
            block_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
        )
    cache_seqlens = torch.randint(
        0 if new_kv else 1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (
            (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
            if new_kv
            else (seqlen_k + 1)
        ),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    if has_leftpad:
        cache_leftpad = torch.cat([torch.randint(0, cache_seqlens[i].item(), (1,), dtype=torch.int32, device=device)
                                   if cache_seqlens[i].item() > 0 else torch.zeros(1, dtype=torch.int32, device=device)
                                   for i in range(batch_size)])
    else:
        cache_leftpad = None
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    if has_leftpad:
        key_padding_mask = torch.logical_and(
            key_padding_mask, arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k)
        )
    cache_batch_idx = None if not has_batch_idx else \
        torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[:batch_size]
    attn_bias = None
    if True: # rotary_dim > 0:
        cos, sin = None, None
        q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (
        k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    v_cache_ref = (
        v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    out = flash_attn_with_kvcache(
        q,
        k_cache if paged_kv_block_size is None else k_cache_paged,
        v_cache if paged_kv_block_size is None else v_cache_paged,
        k,
        v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        block_table=block_table,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
    )
    out_ref = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        attn_bias,
        causal=causal,
        window_size=window_size,
        key_leftpad=cache_leftpad,
    )
    out_pt = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        attn_bias,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
        key_leftpad=cache_leftpad,
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        if paged_kv_block_size is None:
            k_cache_select = (
                k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
            )
            v_cache_select = (
                v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
            )
        else:
            k_cache_select = rearrange(
                k_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    mult = 3
    assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5


def _generate_block_kvcache(seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype):
    num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, num_blocks



@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (239, 1),
        (3, 799),
        (799, 3),
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (257, 257),
    ],
)
def test_flash_attn_race_condition(seqlen_q, seqlen_k, d, causal, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    torch.random.manual_seed(42)
    out0 = flash_attn_func(q, k, v, causal=causal)
    g = torch.randn_like(out0)
    dq0, dk0, dv0 = torch.autograd.grad(out0, (q, k, v), g)
    # Numerical error if we just do any arithmetic on dq
    dq_atol = 2 * ((dq0 + 0.3 - 0.3) - dq0).abs().max().item()

    for i in range(250):
        torch.random.manual_seed(42)
        out = flash_attn_func(q, k, v, causal=causal)
        assert torch.equal(out, out0)

        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_equal = torch.allclose(dq, dq0, atol=dq_atol)
        if not dq_equal:
            print(f"Iter {i}, {dq_atol = }, dQ max diff: {(dq - dq0).abs().max().item()}")
        assert torch.equal(dv, dv0)
        assert torch.equal(dk, dk0)
        assert dq_equal


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize("seqlen", [2, 5, 17, 128])
def test_flash_attn_bwd_overflow(seqlen, d, causal, dtype):
    """We previously had a bug where not masking elements beyond seqlen_k caused NaN in dQ,
    in the case where seqlen % 128 != 0.
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 5
    q = torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 5
    k, v = [
        torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 3
        for _ in range(2)
    ]
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out = flash_attn_func(q, k, v, causal=causal)
    g = torch.randn_like(out)
    out.backward(g)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref(q_pt, k_pt, v_pt, causal=causal, upcast=False, reorder_ops=True)
    out_pt.backward(g)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_ref.backward(g)
    print(f"dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= 5 * (q_pt.grad - q_ref.grad).abs().max().item() + 1e-3
    assert (k.grad - k_ref.grad).abs().max().item() <= 5 * (k_pt.grad - k_ref.grad).abs().max().item() + 1e-3
    assert (v.grad - v_ref.grad).abs().max().item() <= 5 * (v_pt.grad - v_ref.grad).abs().max().item() + 1e-3


@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 256])
def test_flash_attn_bwd_transpose(seqlen, d, causal, dtype):
    """We previously had a bug where we were using the wrong strides of dout, which shows up when dout is not contiguous.
    """
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 5
    nheads = 2
    q, k, v = [
        torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    out = rearrange(flash_attn_func(q, k, v, causal=causal), "b s ... -> s b ...")
    # So g is not contiguous
    g = torch.randn(seqlen, 2 * batch_size, nheads, d, dtype=dtype, device="cuda")[:, ::2]
    out.backward(g)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref(q_pt, k_pt, v_pt, causal=causal, upcast=False, reorder_ops=True)
    out_pt = rearrange(out_pt, "b s ... -> s b ...")
    out_pt.backward(g)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_ref = rearrange(out_ref, "b s ... -> s b ...")
    out_ref.backward(g)
    print(f"dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= 2 * (q_pt.grad - q_ref.grad).abs().max().item()
    assert (k.grad - k_ref.grad).abs().max().item() <= 2 * (k_pt.grad - k_ref.grad).abs().max().item()
    assert (v.grad - v_ref.grad).abs().max().item() <= 2 * (v_pt.grad - v_ref.grad).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize('d', [128])
def test_flash_attn_bwd_varlen_overflow(d, causal, dtype):
    """We previously had a bug where not masking elements beyond seqlen_k caused NaN in dQ, in the case where seqlen % 128 != 0 or varlen.
    """
    device = "cuda"
    torch.random.manual_seed(0)
    nheads = 5
    q_cuseqlen = torch.tensor([0, 76, 110, 256], device=device, dtype=torch.int32)
    k_cuseqlen = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    Mq = 256
    Mk = 3

    q = torch.randn([Mq, nheads, d], dtype=dtype, device=device) * 3
    k, v = [torch.randn([Mk, nheads, d], dtype=dtype, device=device) * 3 for _ in range(2)]
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out = flash_attn_varlen_func(q, k, v, q_cuseqlen, k_cuseqlen, Mq, Mk, causal=causal)
    g = torch.randn_like(out)
    out.backward(g)

    assert not q.grad.isnan().any()
    assert not k.grad.isnan().any()
    assert not v.grad.isnan().any()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
    ],
)
def test_flash_attn_deterministic(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk and seqlen_q != seqlen_k:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=causal, window_size=window_size, deterministic=True)

    g = torch.randn_like(out)
    dq0, dk0, dv0 = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
    for _ in range(50):
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
        assert torch.equal(dv, dv0)
        assert torch.equal(dk, dk0)
        assert torch.equal(dq, dq0)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
    ],
)
def test_flash_attn_varlen_deterministic(seqlen_q, seqlen_k, swap_sq_sk, d, causal, local, dtype):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)
    out = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        window_size=window_size,
        deterministic=True,
    )

    g = torch.randn_like(out)
    dq0, dk0, dv0 = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g, retain_graph=True)
    for _ in range(50):
        dq, dk, dv = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g, retain_graph=True)
        assert torch.equal(dv, dv0)
        assert torch.equal(dk, dk0)
        assert torch.equal(dq, dq0)


if __name__ == "__main__":
    test_flash_attn_output(seqlen_q=113, seqlen_k=203, d=128, causal=False, local=True, \
        deterministic=False, mha_type="mha", dtype=torch.bfloat16, softcap=0.0)

    test_flash_attn_varlen_output(seqlen_q=113, seqlen_k=203, d=128, causal=True, local=False, \
        deterministic=False, mha_type="mha", dtype=torch.bfloat16, softcap=0.0)

    test_flash_attn_causal(seqlen_q=3, seqlen_k=799, swap_sq_sk=False, d=128, local=False, dtype=torch.bfloat16)
