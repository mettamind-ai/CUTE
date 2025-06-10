#!/usr/bin/env python3
# Copyright (C) 2024, Tri Dao.

import math

import torch
import torch.nn.functional as F

import pytest
from einops import rearrange

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref
from causal_conv1d.causal_conv1d_interface import causal_conv1d_update, causal_conv1d_update_ref

@pytest.mark.parametrize("return_final_states", [False, True])
# @pytest.mark.parametrize("return_final_states", [True])
@pytest.mark.parametrize("has_initial_states", [False, True])
# @pytest.mark.parametrize("has_initial_states", [False])
@pytest.mark.parametrize("channel_last", [False, True])
# @pytest.mark.parametrize('channel_last', [True])
@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [3])
@pytest.mark.parametrize(
    "seqlen", [1, 2, 8, 16, 32, 64, 128, 129, 130, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize('dim', [64, 4096 + 32])
# @pytest.mark.parametrize('dim', [64])
def test_causal_conv1d(dim, seqlen, width, has_bias, silu_activation, itype, channel_last, has_initial_states, return_final_states):
    if not channel_last and (has_initial_states or return_final_states):
        pytest.skip("Only channel_last support initial_states or return_final_states")
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 2
    # batch = 1
    if not channel_last:
        x = torch.randn(batch, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    if has_initial_states:
        initial_states = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2).requires_grad_()
    else:
        initial_states = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    initial_states_ref = initial_states.detach().clone().requires_grad_() if initial_states is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(x, weight, bias, initial_states=initial_states, return_final_states=return_final_states,
                           activation=activation)
    out_ref = causal_conv1d_ref(x_ref, weight_ref, bias_ref, initial_states=initial_states_ref, return_final_states=return_final_states, activation=activation)
    if return_final_states:
        out, final_states = out
        out_ref, final_states_ref = out_ref
        print(f"Final states max diff: {(final_states - final_states_ref).abs().max().item()}")
        print(f"Final states mean diff: {(final_states - final_states_ref).abs().mean().item()}")
        assert torch.allclose(final_states, final_states_ref, rtol=rtol, atol=atol)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    if return_final_states:
        out += F.sigmoid(final_states).sum(dim=-1, keepdim=True)
        out_ref += F.sigmoid(final_states_ref).sum(dim=-1, keepdim=True)

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")
    if has_initial_states:
        print(f"dinitial_states max diff: {(initial_states.grad - initial_states_ref.grad).abs().max().item()}")

    assert torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    if has_bias:
        assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)
    if has_initial_states:
        assert torch.allclose(initial_states.grad, initial_states_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update(dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1,)).item()
    conv_state = torch.randn(batch, state_len, dim, device=device, dtype=itype).transpose(-1, -2)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = (torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
                     if has_cache_seqlens else None)
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation, cache_seqlens=cache_seqlens)
    out_ref = causal_conv1d_update_ref(x, conv_state_ref, weight, bias, activation=activation, cache_seqlens=cache_seqlens)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update_with_batch_gather(dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1,)).item()

    total_entries = 10 * batch
    conv_state = torch.randn(total_entries, state_len, dim, device=device, dtype=itype).transpose(-1, -2)
    conv_state_indices = torch.randperm(total_entries)[:batch].to(dtype=torch.int32, device=device)

    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = (torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
                     if has_cache_seqlens else None)
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation,
                               cache_seqlens=cache_seqlens, conv_state_indices=conv_state_indices)
    out_ref = causal_conv1d_update_ref(x, conv_state_ref, weight, bias, activation=activation, cache_seqlens=cache_seqlens)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)



# @pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize('channel_last', [True])
# @pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('itype', [torch.bfloat16])
# @pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize('silu_activation', [True])
# @pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize('has_bias', [True])
# @pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize(
    # "seqlen", [8, 16, 32, 64, 128, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
    "seqlen", [2048]
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
def test_causal_conv1d_race_condition(seqlen, width, has_bias, silu_activation, itype, channel_last):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch = 2
    # batch = 1
    dim = 4096 + 32  # Try dim not divisible by 64
    # dim = 64
    if not channel_last:
        x = torch.randn(batch, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    activation = None if not silu_activation else "silu"
    out0 = causal_conv1d_fn(x, weight, bias, activation=activation)
    g = torch.randn_like(out0)
    dx0, dw0, db0 = torch.autograd.grad(out0, (x, weight, bias), g)
    dw_atol = 1e-4
    db_atol = 1e-4

    for i in range(10000):
        out = causal_conv1d_fn(x, weight, bias, activation=activation)
        dx, dw, db = torch.autograd.grad(out, (x, weight, bias), g)
        dw_equal = torch.allclose(dw, dw0, atol=dw_atol)
        # if not dw_equal:
        #     breakpoint()
        if has_bias:
            db_equal = torch.allclose(db, db0, atol=db_atol)
            # if not db_equal:
            #     breakpoint()
        assert torch.equal(out, out0)
        assert torch.equal(dx, dx0)
        assert dw_equal
        if has_bias:
            assert dw_equal
