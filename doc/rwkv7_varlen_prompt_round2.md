# RWKV7 Varlen Testing Strategy - Round 2

## Context

Following your excellent self-rebuttal, we've decided to implement the varlen kernel with:
- **No `s_end`** - use tail forward replay instead (max 15 steps)
- **Global `s_chunk`** checkpoints only
- **Per-sequence blocks** for clean boundary handling
- **Grid 1D** to avoid 65535 limit

Before implementing the kernel, we want to design a comprehensive test suite to verify correctness.

## Goal

Design test cases that will definitively prove the varlen kernel is correct, specifically:
1. Forward outputs match original kernel (run per-sequence)
2. Backward gradients match original kernel
3. No gradient leakage across sequence boundaries
4. Edge cases handled correctly

## Current Test Approach (Draft)

```python
"""
Strategy:
1. Create multiple sequences with different lengths
2. Run each sequence SEPARATELY through ORIGINAL kernel (ground truth)
3. Pack all sequences and run through VARLEN kernel
4. Compare outputs - they must match exactly (within numerical tolerance)
"""

def test_forward_correctness():
    # For each test case:
    # 1. Create packed data (w, q, k, v, a, b) and cu_seqlens
    # 2. Run original kernel on each sequence separately (with padding to CHUNK_LEN multiple)
    # 3. Run varlen kernel on packed data
    # 4. Compare outputs
    
    test_cases = [
        [32, 32],           # Two equal sequences, multiple of CHUNK_LEN
        [16, 48],           # Different lengths, both multiple of CHUNK_LEN
        [20, 30, 14],       # Three sequences, NOT multiples of CHUNK_LEN
        [5, 10, 3],         # Short sequences (< CHUNK_LEN)
        [1, 1, 1],          # Single token sequences
        [64],               # Single long sequence
        [17],               # Single sequence, not multiple of CHUNK_LEN
    ]
```

## Questions for Analysis

### 1. Test Case Coverage

Are the test cases above sufficient? What additional cases should we include?

Specifically:
- What sequence length combinations would stress-test the tail replay logic?
- What combinations would catch boundary bugs?
- Should we test with specific input patterns (e.g., all zeros, all ones, identity-like)?

### 2. Numerical Tolerance

The original kernel uses:
- bfloat16 for inputs/outputs
- float32 for intermediate `sa_` and checkpoints `s_`

What numerical tolerance should we use for comparison?
- Forward output comparison
- Backward gradient comparison

Should we expect **exact** match or allow some epsilon?

### 3. Gradient Leakage Test Design

To prove no gradient leakage, I'm thinking:

```python
def test_no_gradient_leakage():
    # Create sequences A, B, C
    # Pack them together
    # Compute loss ONLY on sequence B's output
    # Backprop
    # Assert: gradients for A and C inputs are EXACTLY zero
```

Is this the right approach? Any edge cases to consider?

### 4. Backward Correctness Test

For backward, should we:
- Compare each gradient tensor (dw, dq, dk, dv, da, db) separately?
- Use `torch.autograd.gradcheck` for numerical gradient verification?
- Both?

### 5. Edge Cases

The self-rebuttal mentioned these edge cases:
- Empty sequence (length = 0)
- Single token sequence (length = 1)
- Sequence length < CHUNK_LEN
- Sequence boundary exactly at chunk boundary
- Very long sequence

For each, what specific behavior should we verify?

### 6. Stress Tests

Should we include:
- Random sequence lengths (many iterations)?
- Very large number of sequences (test grid limit)?
- Memory pressure tests?

### 7. Reference Implementation

For ground truth, I plan to:
1. Pad each sequence to multiple of CHUNK_LEN
2. Run original kernel with batch_size=1
3. Extract only valid (non-padded) outputs

Is this correct? Any issues with this approach?

The padding approach means:
- Sequence of length 17 → padded to 32
- Run original kernel on (1, 32, H, C)
- Take output[:17] as ground truth

Will the padding affect the valid outputs? (I believe no, since state starts at 0 and padding tokens come after valid tokens)

## Deliverables Requested

1. **Complete test case list** with rationale for each
2. **Numerical tolerance recommendations** with justification
3. **Gradient leakage test design** - exact implementation approach
4. **Any gotchas** we might miss

## Constraints

- Original kernel (`wkv7.cu`) must NOT be modified
- Test must work with HEAD_SIZE=64, CHUNK_LEN=16
- bfloat16 dtype throughout
- Must run on CUDA (SM80+)

---

> **Evidence standard**: Only conclude when you have **reliable evidence** from the provided context, or you can **reason it out clearly and defensibly**. If evidence is weak, state uncertainty and propose a conservative fallback.
