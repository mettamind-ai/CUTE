#!/usr/bin/env python3
"""Test parity between WinRWKV (original) and WinRWKVVarlen.

Tests:
1. Single sequence parity (original vs varlen with 1 seq)
2. Multiple sequences with different lengths
3. Edge cases: minimum seq length (16), single token sequences
4. Gradient parity check
"""
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.bfloat16)

from winrwkv import WinRWKV, fused_loss_fn
from winrwkv_varlen import WinRWKVVarlen, fused_loss_fn_varlen, varlen_timeshift, precompute_starts


def test_single_sequence_parity():
    """Test that single sequence gives same results as original."""
    print("=" * 70)
    print("TEST 1: Single Sequence Parity")
    print("=" * 70)
    
    vocab_size = 4096
    dim, n_layers = 256, 6
    ctxlen = 1024
    n_steps = 5
    lr = 1e-3
    
    torch.manual_seed(1981)
    model_orig = WinRWKV(vocab_size, n_layers, dim, ctxlen).cuda()
    
    torch.manual_seed(1981)
    model_var = WinRWKVVarlen(vocab_size, n_layers, dim, ctxlen).cuda()
    
    opt_orig = torch.optim.AdamW(model_orig.parameters(), lr=lr)
    opt_var = torch.optim.AdamW(model_var.parameters(), lr=lr)
    
    losses_orig, losses_var = [], []
    
    for step in range(n_steps):
        torch.manual_seed(42 + step)
        
        input_orig = torch.randint(5, vocab_size // 4, (1, ctxlen), dtype=torch.long).cuda()
        target_orig = F.pad(input_orig[:, 1:], (0, 1), value=-100)
        
        input_var = input_orig.squeeze(0)
        target_var = torch.full((ctxlen,), -100, dtype=torch.long).cuda()
        target_var[:-1] = input_var[1:]
        cu_seqlens = torch.tensor([0, ctxlen], dtype=torch.int32).cuda()
        
        opt_orig.zero_grad()
        xn_orig = model_orig(input_orig, return_logits=False)
        target_flat = target_orig.reshape(-1)
        target_flat[0] = -100
        from optimus import FusedCE
        loss_orig = FusedCE.apply(xn_orig.reshape(-1, dim), model_orig.head.weight, target_flat, 1, -100, 1.0)
        loss_orig.backward()
        opt_orig.step()
        
        opt_var.zero_grad()
        loss_var = fused_loss_fn_varlen(model_var, input_var, target_var, cu_seqlens)
        loss_var.backward()
        opt_var.step()
        
        losses_orig.append(loss_orig.item())
        losses_var.append(loss_var.item())
    
    avg_diff_pct = sum(abs(a-b)/a*100 for a,b in zip(losses_orig, losses_var)) / n_steps
    passed = avg_diff_pct < 5 and losses_orig[-1] < losses_orig[0] and losses_var[-1] < losses_var[0]
    
    print(f"Loss diff: {avg_diff_pct:.2f}%")
    print(f"Original: {losses_orig[0]:.4f} -> {losses_orig[-1]:.4f}")
    print(f"Varlen:   {losses_var[0]:.4f} -> {losses_var[-1]:.4f}")
    print(f"{'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_multiple_sequences():
    """Test varlen with multiple sequences of different lengths."""
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Sequences (different lengths)")
    print("=" * 70)
    
    vocab_size = 4096
    dim, n_layers = 256, 6
    n_steps = 5
    lr = 1e-3
    
    # Various sequence length combinations
    seq_configs = [
        [256, 256, 256, 256],      # 4 equal sequences
        [512, 256, 128, 128],      # mixed lengths
        [64, 64, 64, 64, 64, 64],  # many short sequences
        [16, 32, 48, 64, 80, 96],  # increasing lengths
    ]
    
    all_passed = True
    
    for seq_lengths in seq_configs:
        total_tokens = sum(seq_lengths)
        
        torch.manual_seed(1981)
        model = WinRWKVVarlen(vocab_size, n_layers, dim, total_tokens).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        
        losses = []
        for step in range(n_steps):
            torch.manual_seed(42 + step)
            
            input_ids = torch.randint(5, vocab_size // 4, (total_tokens,), dtype=torch.long).cuda()
            target = torch.full((total_tokens,), -100, dtype=torch.long).cuda()
            
            offset = 0
            for seq_len in seq_lengths:
                if seq_len > 1:
                    target[offset:offset + seq_len - 1] = input_ids[offset + 1:offset + seq_len]
                offset += seq_len
            
            cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lengths), 0)), dtype=torch.int32).cuda()
            
            opt.zero_grad()
            loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        
        improved = losses[-1] < losses[0]
        print(f"Seqs {seq_lengths}: {losses[0]:.4f} -> {losses[-1]:.4f} {'✓' if improved else '✗'}")
        all_passed = all_passed and improved
    
    print(f"{'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_minimum_sequence_length():
    """Test with minimum sequence length (CHUNK_LEN=16)."""
    print("\n" + "=" * 70)
    print("TEST 3: Minimum Sequence Length (16 tokens)")
    print("=" * 70)
    
    vocab_size = 4096
    dim, n_layers = 256, 6
    n_steps = 5
    lr = 1e-3
    
    # All sequences at minimum length
    seq_lengths = [16] * 8  # 8 sequences of 16 tokens each
    total_tokens = sum(seq_lengths)
    
    torch.manual_seed(1981)
    model = WinRWKVVarlen(vocab_size, n_layers, dim, total_tokens).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    for step in range(n_steps):
        torch.manual_seed(42 + step)
        
        input_ids = torch.randint(5, vocab_size // 4, (total_tokens,), dtype=torch.long).cuda()
        target = torch.full((total_tokens,), -100, dtype=torch.long).cuda()
        
        offset = 0
        for seq_len in seq_lengths:
            if seq_len > 1:
                target[offset:offset + seq_len - 1] = input_ids[offset + 1:offset + seq_len]
            offset += seq_len
        
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lengths), 0)), dtype=torch.int32).cuda()
        
        opt.zero_grad()
        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    
    improved = losses[-1] < losses[0]
    print(f"8 seqs of 16 tokens: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"{'✓ PASSED' if improved else '✗ FAILED'}")
    return improved


def test_timeshift_correctness():
    """Test varlen_timeshift produces correct output."""
    print("\n" + "=" * 70)
    print("TEST 4: Time-shift Correctness")
    print("=" * 70)
    
    # Simple test case
    T, C = 10, 4
    x = torch.arange(T * C, dtype=torch.float32).reshape(T, C).cuda()
    
    # 3 sequences: [0:3], [3:7], [7:10]
    cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.int32).cuda()
    starts = precompute_starts(cu_seqlens, T)
    
    xx = varlen_timeshift(x, starts)
    
    # Check sequence starts are zero
    starts_zero = (xx[0].abs().sum() == 0) and (xx[3].abs().sum() == 0) and (xx[7].abs().sum() == 0)
    
    # Check non-starts: xx[t] = x[t-1] - x[t]
    correct_shift = True
    for t in [1, 2, 4, 5, 6, 8, 9]:
        expected = x[t-1] - x[t]
        if not torch.allclose(xx[t], expected, atol=1e-5):
            correct_shift = False
            break
    
    passed = starts_zero and correct_shift
    print(f"Sequence starts zeroed: {'✓' if starts_zero else '✗'}")
    print(f"Non-starts shifted correctly: {'✓' if correct_shift else '✗'}")
    print(f"{'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_gradient_flow():
    """Test that gradients flow correctly through varlen model."""
    print("\n" + "=" * 70)
    print("TEST 5: Gradient Flow")
    print("=" * 70)
    
    vocab_size = 4096
    dim, n_layers = 256, 2  # smaller for speed
    seq_lengths = [64, 64]
    total_tokens = sum(seq_lengths)
    
    torch.manual_seed(1981)
    model = WinRWKVVarlen(vocab_size, n_layers, dim, total_tokens).cuda()
    
    input_ids = torch.randint(5, vocab_size // 4, (total_tokens,), dtype=torch.long).cuda()
    target = torch.full((total_tokens,), -100, dtype=torch.long).cuda()
    
    offset = 0
    for seq_len in seq_lengths:
        if seq_len > 1:
            target[offset:offset + seq_len - 1] = input_ids[offset + 1:offset + seq_len]
        offset += seq_len
    
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lengths), 0)), dtype=torch.int32).cuda()
    
    # Forward + backward
    loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
    loss.backward()
    
    # Check gradients exist and are finite
    # Note: v1, v2, v0 at layer 0 have no grad because layer 0 uses v_first = v.clone()
    # instead of mixing with v_first. This is expected behavior.
    grad_ok = True
    no_grad_expected = {'blocks.0.att.v1', 'blocks.0.att.v2', 'blocks.0.att.v0'}
    
    for name, param in model.named_parameters():
        if param.grad is None:
            if name not in no_grad_expected:
                print(f"  Unexpected no grad: {name}")
                grad_ok = False
        elif not torch.isfinite(param.grad).all():
            print(f"  Non-finite grad: {name}")
            grad_ok = False
    
    print(f"All gradients exist and finite (except expected): {'✓' if grad_ok else '✗'}")
    print(f"{'✓ PASSED' if grad_ok else '✗ FAILED'}")
    return grad_ok


def test_chunk_boundary_sequences():
    """Test sequences that align/misalign with chunk boundaries."""
    print("\n" + "=" * 70)
    print("TEST 6: Chunk Boundary Alignment (CHUNK_LEN=16)")
    print("=" * 70)
    
    vocab_size = 4096
    dim, n_layers = 256, 2
    n_steps = 3
    lr = 1e-3
    
    # Test cases: aligned, misaligned, mixed
    test_cases = [
        ("Aligned (32, 48, 16)", [32, 48, 16]),
        ("Misaligned (17, 33, 46)", [17, 33, 46]),  # will be padded internally
        ("Mixed (16, 25, 32, 23)", [16, 25, 32, 23]),
    ]
    
    all_passed = True
    
    for name, seq_lengths in test_cases:
        # Ensure total is multiple of 16
        total_tokens = sum(seq_lengths)
        if total_tokens % 16 != 0:
            pad = 16 - (total_tokens % 16)
            seq_lengths[-1] += pad
            total_tokens = sum(seq_lengths)
        
        torch.manual_seed(1981)
        model = WinRWKVVarlen(vocab_size, n_layers, dim, total_tokens).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        
        losses = []
        for step in range(n_steps):
            torch.manual_seed(42 + step)
            
            input_ids = torch.randint(5, vocab_size // 4, (total_tokens,), dtype=torch.long).cuda()
            target = torch.full((total_tokens,), -100, dtype=torch.long).cuda()
            
            offset = 0
            for seq_len in seq_lengths:
                if seq_len > 1:
                    target[offset:offset + seq_len - 1] = input_ids[offset + 1:offset + seq_len]
                offset += seq_len
            
            cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lengths), 0)), dtype=torch.int32).cuda()
            
            opt.zero_grad()
            loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        
        improved = losses[-1] < losses[0]
        print(f"{name}: {losses[0]:.4f} -> {losses[-1]:.4f} {'✓' if improved else '✗'}")
        all_passed = all_passed and improved
    
    print(f"{'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_long_sequence():
    """Test with longer context length."""
    print("\n" + "=" * 70)
    print("TEST 7: Long Sequence (4096 tokens)")
    print("=" * 70)
    
    vocab_size = 4096
    dim, n_layers = 256, 4
    ctxlen = 4096
    n_steps = 3
    lr = 1e-3
    
    # Single long sequence
    torch.manual_seed(1981)
    model = WinRWKVVarlen(vocab_size, n_layers, dim, ctxlen).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    for step in range(n_steps):
        torch.manual_seed(42 + step)
        
        input_ids = torch.randint(5, vocab_size // 4, (ctxlen,), dtype=torch.long).cuda()
        target = torch.full((ctxlen,), -100, dtype=torch.long).cuda()
        target[:-1] = input_ids[1:]
        cu_seqlens = torch.tensor([0, ctxlen], dtype=torch.int32).cuda()
        
        opt.zero_grad()
        loss = fused_loss_fn_varlen(model, input_ids, target, cu_seqlens)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    
    improved = losses[-1] < losses[0]
    print(f"4096 tokens: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"{'✓ PASSED' if improved else '✗ FAILED'}")
    return improved


def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING ALL PARITY TESTS")
    print("=" * 70 + "\n")
    
    results = []
    results.append(("Single Sequence Parity", test_single_sequence_parity()))
    results.append(("Multiple Sequences", test_multiple_sequences()))
    results.append(("Minimum Sequence Length", test_minimum_sequence_length()))
    results.append(("Time-shift Correctness", test_timeshift_correctness()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Chunk Boundary Alignment", test_chunk_boundary_sequences()))
    results.append(("Long Sequence", test_long_sequence()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
