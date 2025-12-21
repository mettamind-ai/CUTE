#!/usr/bin/env python3
"""
Test suite for RWKV7 varlen kernel correctness.

Strategy:
1. Create multiple sequences with different lengths
2. Run each sequence separately through ORIGINAL kernel (ground truth)
3. Pack all sequences and run through VARLEN kernel
4. Compare outputs - they must match exactly (within numerical tolerance)

This verifies:
- Forward pass correctness (outputs match)
- Backward pass correctness (gradients match)
- No gradient leakage across sequence boundaries
"""
import os
import torch
import torch.nn.functional as F
from torch import Tensor

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(42)

HEAD_SIZE = 64
CHUNK_LEN = 16

# Compile flags
FLAGS = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}",
    "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]

def load_kernels():
    """Load both original and varlen kernels."""
    from torch.utils.cpp_extension import load
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Compiling original kernel (wkv7.cu)...")
    load(name="wind_backstepping", 
         sources=[f'{base_dir}/wkv7.cu'], 
         is_python_module=False, verbose=True, extra_cuda_cflags=FLAGS)
    
    print("\nCompiling varlen kernel (wkv7_varlen.cu)...")
    load(name="wind_backstepping_varlen", 
         sources=[f'{base_dir}/wkv7_varlen.cu'], 
         is_python_module=False, verbose=True, extra_cuda_cflags=FLAGS)
    
    print("\nBoth kernels compiled successfully!")


class OriginalRWKV7(torch.autograd.Function):
    """Original fixed-length RWKV7 kernel wrapper."""
    @staticmethod
    def forward(ctx, w, q, k, v, a, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0, f"T={T} must be multiple of CHUNK_LEN={CHUNK_LEN}"
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, a, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, a, b])
        
        y = torch.empty_like(v)
        s = torch.empty(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, a, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, a, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        # Convert dy to bfloat16 if needed (autograd may pass float32)
        if dy.dtype != torch.bfloat16:
            dy = dy.to(torch.bfloat16)
        dy = dy.contiguous()
        w, q, k, v, a, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, da, db = [torch.empty_like(x) for x in [w, q, k, v, a, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, a, b, dy, s, sa, dw, dq, dk, dv, da, db)
        return dw, dq, dk, dv, da, db


class VarlenRWKV7(torch.autograd.Function):
    """Varlen RWKV7 kernel wrapper."""
    @staticmethod
    def forward(ctx, w, q, k, v, a, b, cu_seqlens):
        # w shape: (total_tokens, H, C)
        total_tokens, H, C = w.shape
        num_seqs = cu_seqlens.shape[0] - 1
        num_chunks = (total_tokens + CHUNK_LEN - 1) // CHUNK_LEN
        
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, a, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, a, b])
        assert cu_seqlens.dtype == torch.int32
        
        y = torch.empty_like(v)
        s_chunk = torch.empty(H, num_chunks, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(total_tokens, H, C, dtype=torch.float32, device=w.device)
        
        torch.ops.wind_backstepping_varlen.forward_varlen(
            w, q, k, v, a, b, cu_seqlens, y, s_chunk, sa
        )
        ctx.save_for_backward(w, q, k, v, a, b, cu_seqlens, s_chunk, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        # Convert dy to bfloat16 if needed (autograd may pass float32)
        if dy.dtype != torch.bfloat16:
            dy = dy.to(torch.bfloat16)
        dy = dy.contiguous()
        w, q, k, v, a, b, cu_seqlens, s_chunk, sa = ctx.saved_tensors
        dw, dq, dk, dv, da, db = [torch.empty_like(x) for x in [w, q, k, v, a, b]]
        torch.ops.wind_backstepping_varlen.backward_varlen(
            w, q, k, v, a, b, dy, cu_seqlens, s_chunk, sa,
            dw, dq, dk, dv, da, db
        )
        return dw, dq, dk, dv, da, db, None  # None for cu_seqlens


def run_original_single_sequence(w, q, k, v, a, b):
    """Run original kernel on a single sequence (B=1)."""
    return OriginalRWKV7.apply(w, q, k, v, a, b)


def create_test_data(seq_lengths, H, C, device='cuda', requires_grad=False):
    """
    Create random test data for multiple sequences.
    
    Returns:
        Packed tensors (total_tokens, H, C) and cu_seqlens
    """
    total_tokens = sum(seq_lengths)
    
    # Scale down random values to avoid numerical overflow in RWKV7 recurrence
    scale = 0.1
    w = torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device=device, requires_grad=requires_grad) * scale
    q = torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device=device, requires_grad=requires_grad) * scale
    k = torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device=device, requires_grad=requires_grad) * scale
    v = torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device=device, requires_grad=requires_grad) * scale
    a = torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device=device, requires_grad=requires_grad) * scale
    b = torch.randn(total_tokens, H, C, dtype=torch.bfloat16, device=device, requires_grad=requires_grad) * scale
    
    cu_seqlens = [0]
    for length in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    
    return w, q, k, v, a, b, cu_seqlens


def run_original_per_sequence(seq_lengths, w, q, k, v, a, b, H, C):
    """
    Run original kernel separately for each sequence (ground truth).
    
    Returns:
        y_packed: packed output (total_tokens, H, C)
    """
    outputs = []
    offset = 0
    
    for seq_len in seq_lengths:
        if seq_len == 0:
            offset += seq_len
            continue
            
        # Pad to multiple of CHUNK_LEN
        padded_len = ((seq_len + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
        
        # Extract this sequence's data
        w_seq = w[offset:offset+seq_len].clone()
        q_seq = q[offset:offset+seq_len].clone()
        k_seq = k[offset:offset+seq_len].clone()
        v_seq = v[offset:offset+seq_len].clone()
        a_seq = a[offset:offset+seq_len].clone()
        b_seq = b[offset:offset+seq_len].clone()
        
        # Pad if needed
        if padded_len > seq_len:
            pad_len = padded_len - seq_len
            w_seq = F.pad(w_seq, (0, 0, 0, 0, 0, pad_len))
            q_seq = F.pad(q_seq, (0, 0, 0, 0, 0, pad_len))
            k_seq = F.pad(k_seq, (0, 0, 0, 0, 0, pad_len))
            v_seq = F.pad(v_seq, (0, 0, 0, 0, 0, pad_len))
            a_seq = F.pad(a_seq, (0, 0, 0, 0, 0, pad_len))
            b_seq = F.pad(b_seq, (0, 0, 0, 0, 0, pad_len))
        
        # Reshape to (1, T, H, C) for original kernel
        w_seq = w_seq.unsqueeze(0).contiguous()  # (1, padded_len, H, C)
        q_seq = q_seq.unsqueeze(0).contiguous()
        k_seq = k_seq.unsqueeze(0).contiguous()
        v_seq = v_seq.unsqueeze(0).contiguous()
        a_seq = a_seq.unsqueeze(0).contiguous()
        b_seq = b_seq.unsqueeze(0).contiguous()
        
        # Run original kernel
        y_seq = run_original_single_sequence(w_seq, q_seq, k_seq, v_seq, a_seq, b_seq)
        
        # Extract only the valid (non-padded) output
        y_valid = y_seq[0, :seq_len]  # (seq_len, H, C)
        outputs.append(y_valid)
        
        offset += seq_len
    
    if len(outputs) == 0:
        return torch.empty(0, H, C, dtype=torch.bfloat16, device=w.device)
    
    y_packed = torch.cat(outputs, dim=0)
    return y_packed


def test_forward_correctness():
    """Test that varlen forward matches original kernel run per-sequence."""
    print("\n" + "="*60)
    print("TEST: Forward Correctness")
    print("="*60)
    
    H = 2  # num heads
    C = HEAD_SIZE
    
    test_cases = [
        ([32, 32], "Two equal sequences, multiple of CHUNK_LEN"),
        ([16, 48], "Different lengths, both multiple of CHUNK_LEN"),
        ([20, 30, 14], "Three sequences, NOT multiples of CHUNK_LEN"),
        ([5, 10, 3], "Short sequences (< CHUNK_LEN)"),
        ([1, 1, 1], "Single token sequences"),
        ([64], "Single long sequence"),
        ([17], "Single sequence, not multiple of CHUNK_LEN"),
        ([16], "Single sequence, exactly CHUNK_LEN"),
        ([15], "Single sequence, CHUNK_LEN - 1"),
        ([32, 1, 32], "Short sequence between long ones"),
    ]
    
    all_passed = True
    
    for seq_lengths, description in test_cases:
        print(f"\n{description}: {seq_lengths}")
        
        # Create test data
        w, q, k, v, a, b, cu_seqlens = create_test_data(seq_lengths, H, C)
        
        # Run original kernel per-sequence (ground truth)
        with torch.no_grad():
            y_expected = run_original_per_sequence(seq_lengths, w, q, k, v, a, b, H, C)
        
        # Run varlen kernel
        with torch.no_grad():
            y_varlen = VarlenRWKV7.apply(w, q, k, v, a, b, cu_seqlens)
        
        # Compare
        max_diff = (y_expected - y_varlen).abs().max().item()
        mean_diff = (y_expected - y_varlen).abs().mean().item()
        
        # Tolerance: bfloat16 has ~3 decimal digits precision
        tolerance = 1e-2
        passed = max_diff < tolerance
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        
        if not passed:
            all_passed = False
            # Debug info
            print(f"  Expected shape: {y_expected.shape}, Varlen shape: {y_varlen.shape}")
            print(f"  Expected[:5]: {y_expected.flatten()[:5]}")
            print(f"  Varlen[:5]: {y_varlen.flatten()[:5]}")
    
    return all_passed


def test_backward_correctness():
    """Test that varlen backward matches original kernel gradients."""
    print("\n" + "="*60)
    print("TEST: Backward Correctness")
    print("="*60)
    
    H = 2
    C = HEAD_SIZE
    
    # Test with single sequences that are multiples of CHUNK_LEN
    # (no padding needed, direct comparison)
    test_cases = [
        (32, "Single sequence, 2 chunks"),
        (48, "Single sequence, 3 chunks"),
        (64, "Single sequence, 4 chunks"),
        (16, "Single sequence, 1 chunk"),
    ]
    
    all_passed = True
    
    for seq_len, description in test_cases:
        print(f"\n{description}: seq_len={seq_len}")
        
        # Create test data (seq_len is multiple of CHUNK_LEN, no padding needed)
        scale = 0.1
        
        # Original kernel format: (B, T, H, C)
        w_orig = torch.randn(1, seq_len, H, C, dtype=torch.bfloat16, device='cuda', requires_grad=True) * scale
        q_orig = torch.randn(1, seq_len, H, C, dtype=torch.bfloat16, device='cuda', requires_grad=True) * scale
        k_orig = torch.randn(1, seq_len, H, C, dtype=torch.bfloat16, device='cuda', requires_grad=True) * scale
        v_orig = torch.randn(1, seq_len, H, C, dtype=torch.bfloat16, device='cuda', requires_grad=True) * scale
        a_orig = torch.randn(1, seq_len, H, C, dtype=torch.bfloat16, device='cuda', requires_grad=True) * scale
        b_orig = torch.randn(1, seq_len, H, C, dtype=torch.bfloat16, device='cuda', requires_grad=True) * scale
        
        # Varlen format: (total_tokens, H, C) - same data, different layout
        w_var = w_orig.detach().clone().reshape(seq_len, H, C).contiguous().requires_grad_(True)
        q_var = q_orig.detach().clone().reshape(seq_len, H, C).contiguous().requires_grad_(True)
        k_var = k_orig.detach().clone().reshape(seq_len, H, C).contiguous().requires_grad_(True)
        v_var = v_orig.detach().clone().reshape(seq_len, H, C).contiguous().requires_grad_(True)
        a_var = a_orig.detach().clone().reshape(seq_len, H, C).contiguous().requires_grad_(True)
        b_var = b_orig.detach().clone().reshape(seq_len, H, C).contiguous().requires_grad_(True)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
        
        # Run original kernel
        y_orig = OriginalRWKV7.apply(w_orig, q_orig, k_orig, v_orig, a_orig, b_orig)
        loss_orig = y_orig.sum()
        loss_orig.backward()
        
        # Run varlen kernel
        y_var = VarlenRWKV7.apply(w_var, q_var, k_var, v_var, a_var, b_var, cu_seqlens)
        loss_var = y_var.sum()
        loss_var.backward()
        
        # Compare gradients
        tolerance = 1e-2
        grad_names = ['dw', 'dq', 'dk', 'dv', 'da', 'db']
        grads_orig = [
            w_orig.grad.reshape(seq_len, H, C),
            q_orig.grad.reshape(seq_len, H, C),
            k_orig.grad.reshape(seq_len, H, C),
            v_orig.grad.reshape(seq_len, H, C),
            a_orig.grad.reshape(seq_len, H, C),
            b_orig.grad.reshape(seq_len, H, C),
        ]
        grads_var = [w_var.grad, q_var.grad, k_var.grad, v_var.grad, a_var.grad, b_var.grad]
        
        case_passed = True
        for name, g_orig, g_var in zip(grad_names, grads_orig, grads_var):
            if g_orig is None or g_var is None:
                print(f"  {name}: SKIP (None gradient)")
                continue
            max_diff = (g_orig - g_var).abs().max().item()
            passed = max_diff < tolerance
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: max_diff = {max_diff:.6e}")
            if not passed:
                case_passed = False
        
        if not case_passed:
            all_passed = False
    
    return all_passed


def test_no_gradient_leakage():
    """
    Test that gradients don't leak across sequence boundaries.
    
    Strategy:
    1. Create sequences A, B, C
    2. Pack them together
    3. Compute loss ONLY on sequence B's output
    4. Backprop
    5. Assert: gradients for A and C inputs are EXACTLY zero
    """
    print("\n" + "="*60)
    print("TEST: No Gradient Leakage")
    print("="*60)
    
    H = 2
    C = HEAD_SIZE
    seq_lengths = [20, 30, 25]  # A, B, C
    
    w, q, k, v, a, b, cu_seqlens = create_test_data(seq_lengths, H, C, requires_grad=True)
    
    # Run varlen kernel
    y = VarlenRWKV7.apply(w, q, k, v, a, b, cu_seqlens)
    
    # Compute loss ONLY on sequence B (indices 20:50)
    start_B = seq_lengths[0]
    end_B = start_B + seq_lengths[1]
    loss = y[start_B:end_B].sum()
    loss.backward()
    
    # Check gradients for A (indices 0:20) and C (indices 50:75) are zero
    start_A, end_A = 0, seq_lengths[0]
    start_C, end_C = end_B, end_B + seq_lengths[2]
    
    all_passed = True
    
    for name, grad in [('w', w.grad), ('q', q.grad), ('k', k.grad), 
                       ('v', v.grad), ('a', a.grad), ('b', b.grad)]:
        if grad is None:
            print(f"  {name}: SKIP (None)")
            continue
        
        # Check A region
        max_A = grad[start_A:end_A].abs().max().item()
        # Check C region  
        max_C = grad[start_C:end_C].abs().max().item()
        # Check B region (should be non-zero)
        max_B = grad[start_B:end_B].abs().max().item()
        
        # A and C should be exactly zero (or very close due to floating point)
        tolerance = 1e-10
        passed_A = max_A < tolerance
        passed_C = max_C < tolerance
        passed_B = max_B > tolerance  # B should have non-zero gradients
        
        status = "✓" if (passed_A and passed_C and passed_B) else "✗"
        print(f"  {status} {name}: A={max_A:.2e}, B={max_B:.2e}, C={max_C:.2e}")
        
        if not (passed_A and passed_C):
            all_passed = False
            print(f"    LEAK DETECTED! Gradients should be zero for A and C")
    
    return all_passed


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("TEST: Edge Cases")
    print("="*60)
    
    H = 2
    C = HEAD_SIZE
    
    test_cases = [
        ([1], "Single token"),
        ([1, 1, 1, 1], "All single tokens"),
        ([3, 7, 2, 5], "All < CHUNK_LEN"),
        ([16, 32, 16], "All multiples of CHUNK_LEN"),
        ([15, 17, 31, 33], "Around CHUNK_LEN boundaries"),
        ([128], "Long sequence"),
    ]
    
    all_passed = True
    
    for seq_lengths, description in test_cases:
        print(f"\n{description}: {seq_lengths}")
        
        try:
            w, q, k, v, a, b, cu_seqlens = create_test_data(seq_lengths, H, C)
            
            with torch.no_grad():
                y_expected = run_original_per_sequence(seq_lengths, w, q, k, v, a, b, H, C)
                y_varlen = VarlenRWKV7.apply(w, q, k, v, a, b, cu_seqlens)
            
            max_diff = (y_expected - y_varlen).abs().max().item()
            tolerance = 1e-2
            passed = max_diff < tolerance
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} | Max diff: {max_diff:.6e}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("="*60)
    print("RWKV7 Varlen Kernel Test Suite")
    print("="*60)
    
    load_kernels()
    
    results = {}
    results['forward'] = test_forward_correctness()
    results['backward'] = test_backward_correctness()
    results['leakage'] = test_no_gradient_leakage()
    results['edge_cases'] = test_edge_cases()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(results.values())
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED!"))
