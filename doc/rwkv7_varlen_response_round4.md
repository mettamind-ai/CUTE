Dưới đây là 2 phần như bạn yêu cầu:

1. **Code review verdict** cho các fix (theo snippet bạn đưa; mình sẽ nói rõ chỗ nào có/không có đủ “evidence”).
2. **Các test function mới** để append vào `test_wkv7_varlen.py` (không phải full file), tập trung “bắt bug thật”: p0 off-by-one, checkpoint index/layout sai, boundary leakage, missing writes, numeric instability.
   Mình dựa trên cấu trúc test hiện tại của bạn để viết tương thích. 

---

## 1) Code review verdict cho các fix

> Lưu ý về evidence: bạn chỉ paste **key sections** chứ không paste full file. Vì vậy mình **có thể kết luận chắc** về tính đúng/sai của các biểu thức bạn đưa (p0, static_assert, TORCH_CHECK macro), nhưng **không thể xác nhận “đã được áp dụng khắp nơi & không còn chỗ hard-code cũ”** nếu không nhìn full `wkv7_varlen.cu` build đang chạy.

### 1.1 p0 mới: `((p_end + 1) & ~(CHUNK - 1)) - 1` có đúng không?

**Đúng, với điều kiện `CHUNK` là power-of-two và `p_end+1` không âm.**

* Nếu `CHUNK = 2^k` thì mask `~(CHUNK-1)` là bitmask “xoá k bit thấp”.
* Với `x >= 0`, ta có:
  `x & ~(CHUNK-1)` = `floor(x/CHUNK) * CHUNK`.

Bạn dùng `x = p_end + 1`, và trong “valid inputs” thì `p_end = end-1`, `end>start>=0` ⇒ `p_end >= 0` ⇒ `x >= 1`. Nên không dính case âm.

✅ Vậy:
`p0 = ((p_end + 1) & ~(CHUNK - 1)) - 1`
tương đương:
`p0 = floor((p_end + 1)/CHUNK)*CHUNK - 1`
đúng với mọi sequence hợp lệ.

**Edge sanity**: nếu `p_end < CHUNK-1` thì `p0 = -1`, rơi vào nhánh “no checkpoint inside seq” như thiết kế.

---

### 1.2 static_assert CHUNK power-of-two: ổn và cần thiết

`static_assert((CHUNK & (CHUNK - 1)) == 0, "CHUNK must be power of 2");`

* ✅ Đây là cách chuẩn để đảm bảo mọi chỗ bạn dùng `& (CHUNK-1)` thực sự tương đương modulo.
* Gợi ý nhỏ: bạn có thể thêm `static_assert(CHUNK > 0)` cho rõ ràng (dù thực tế define CHUNK không bao giờ 0).

---

### 1.3 TORCH_CHECK validations: đã đi đúng hướng, nhưng **còn thiếu vài check quan trọng**

Các check bạn liệt kê (CUDA/contiguous/dtype) là “phần cứng” quan trọng nhất.

Nhưng để “full validation” theo nghĩa **không thể OOB / shape mismatch** thì bạn nên bổ sung thêm các check sau (nếu chưa có trong phần code bạn không paste):

#### (A) Shape checks bắt buộc (ngăn OOB ngay lập tức)

Kernel compile-time dùng `_C_` và launch block `_C_` threads. Nếu input C dimension không khớp `_C_`, indexing sẽ OOB.

Nên check:

* `TORCH_CHECK(w.dim()==3, "w must be (total_tokens,H,C)")` và tương tự q/k/v/a/b/y/sa
* `TORCH_CHECK(w.size(2) == _C_, "C mismatch with compiled HEAD_SIZE")`
* `TORCH_CHECK(q.sizes()==w.sizes() ... v/a/b/y)` đồng shape
* `TORCH_CHECK(sa.sizes() == w.sizes(), "sa must be (total_tokens,H,C) fp32")`
* `TORCH_CHECK(s_chunk.dim()==4 && s_chunk.size(0)==H && s_chunk.size(2)==_C_ && s_chunk.size(3)==_C_)`
* `TORCH_CHECK(s_chunk.size(1) == (total_tokens + CHUNK - 1)/CHUNK)`

#### (B) cu_seqlens semantic checks (rất nên có)

Nếu thiếu cái này thì vẫn có thể “không crash ngay” nhưng output sai/race:

* `cu_seqlens.dim()==1`
* `cu_seqlens.numel() == num_seqs + 1` (bạn đã implicit qua `num_seqs = size(0)-1`)
* **giá trị**:

  * `cu_seqlens[0] == 0`
  * `cu_seqlens[-1] == total_tokens`
  * monotonic non-decreasing

Cái này muốn check trên host thường phải copy `cu_seqlens` về CPU (vì bạn đang TORCH_CHECK trong C++). Chi phí nhỏ vì `num_seqs` thường nhỏ.

#### (C) dy dtype/contiguous (trong backward)

Trong test wrapper của bạn, bạn convert `dy` về bf16 trước khi gọi op. 
Nhưng nếu ai gọi op trực tiếp, nên có:

* `CHECK_DTYPE_BF16(dy)` (hoặc cho phép FP32 rồi tự cast)
* `CHECK_CONTIGUOUS(dy)`

---

### 1.4 “Reject empty inputs” (`TORCH_CHECK(total_tokens > 0)`)

Đây là **policy quyết định** chứ không phải bugfix.

* ✅ Nếu bạn muốn đơn giản hóa và tránh corner-case `num_chunks=0`, thì reject là hợp lý.
* ⚠️ Nhưng nếu bạn muốn support “empty batch” (ví dụ inference có thể filter hết seq), thì tốt hơn:

  * Nếu `total_tokens==0`: return early ở C++ wrapper và **không launch kernel**.
  * Khi đó vẫn an toàn và ergonomic hơn.

Vì bạn đã chọn reject, mình sẽ viết test đảm bảo nó thật sự raise (để không “half supported” kiểu im lặng).

---

### 1.5 “Round 3 issues” còn sót lại?

Các fix bạn liệt kê đã giải quyết đúng 2 cái “time-bomb” lớn: hard-code 16 và validation đầu vào.

Nhưng vẫn còn 2 nhóm rủi ro (không nhất thiết là bug):

1. **Numerical NaN** cho w_input quá lớn (0 * inf / 1/0) – không được fix (cả original cũng có).
2. **Schedule dependence** (global vs local checkpoint) – bạn đang chấp nhận và muốn test bằng “Ref-Aligned” để so sánh công bằng. 

---

## 2) Các test function mới để append vào `test_wkv7_varlen.py`

### Mục tiêu các test này bắt được gì?

* **Off-by-one p0**: sẽ làm backward load sai checkpoint hoặc replay sai range ⇒ diff lớn / NaN.
* **Checkpoint index/layout sai**: s_chunk bị NaN ở các full-chunk đáng lẽ phải được ghi, hoặc backward đọc nhầm chunk.
* **Boundary leakage**: packed multi-seq + loss chỉ ở một seq ⇒ grads seq khác phải 0 (bạn đã có).
* **Missing writes**: không chỉ y/sa mà cả **s_chunk**.
* **Numeric stability**: w_input cực trị (w≈1, w≈exp(-1), w tiny) không tạo NaN/Inf và vẫn match reference khi aligned.

> Các hàm dưới đây dùng lại class/helper sẵn có trong file của bạn (`OriginalRWKV7`, `VarlenRWKV7`, `create_test_data`, …). 
> Bạn chỉ cần **copy-paste append**. Sau đó thêm vào `__main__` nếu muốn auto-run.

---

```python
def _ref_aligned_original_seq_grads(
    start_offset: int,
    w_seq: Tensor, q_seq: Tensor, k_seq: Tensor, v_seq: Tensor, a_seq: Tensor, b_seq: Tensor,
    weight_seq: Tensor | None = None,
    noop_w_input: float = -100.0,
):
    """
    Run ORIGINAL kernel on a single sequence, but with a prefix of no-op tokens of length (start_offset % CHUNK_LEN)
    so that ORIGINAL's LOCAL checkpoint schedule aligns with VARLEN's GLOBAL checkpoint schedule.

    Returns grads for the REAL tokens only (shape: (L, H, C)) for each of (w,q,k,v,a,b).
    """
    assert w_seq.dim() == 3, "w_seq must be (L,H,C)"
    device = w_seq.device
    L, H, C = w_seq.shape
    pad_pre = start_offset % CHUNK_LEN

    # total length before final pad
    T0 = pad_pre + L
    # original kernel requires T multiple of CHUNK_LEN
    T_pad = ((T0 + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
    # pad_post = T_pad - T0  # not used explicitly

    def make_pad():
        return torch.zeros((1, T_pad, H, C), dtype=torch.bfloat16, device=device)

    # Build padded inputs (leaf tensors)
    w_pad = make_pad()
    q_pad = make_pad()
    k_pad = make_pad()
    v_pad = make_pad()
    a_pad = make_pad()
    b_pad = make_pad()

    # Prefix no-op tokens: w_input very negative => exp(w_input) ~ 0 => w ~= exp(-0)=1
    if pad_pre > 0:
        w_pad[0, :pad_pre].fill_(noop_w_input)
        # q/k/v/a/b are already zeros => no-op exactly if state starts at 0

    # Real tokens placed after prefix
    sl = slice(pad_pre, pad_pre + L)
    w_pad[0, sl] = w_seq
    q_pad[0, sl] = q_seq
    k_pad[0, sl] = k_seq
    v_pad[0, sl] = v_seq
    a_pad[0, sl] = a_seq
    b_pad[0, sl] = b_seq

    # Mark as leafs with grad
    w_pad.requires_grad_(True)
    q_pad.requires_grad_(True)
    k_pad.requires_grad_(True)
    v_pad.requires_grad_(True)
    a_pad.requires_grad_(True)
    b_pad.requires_grad_(True)

    # Run original
    y_pad = OriginalRWKV7.apply(w_pad.contiguous(), q_pad.contiguous(), k_pad.contiguous(),
                               v_pad.contiguous(), a_pad.contiguous(), b_pad.contiguous())

    y_real = y_pad[0, sl]  # (L,H,C)

    if weight_seq is None:
        loss = y_real.sum()
    else:
        # Weight in fp32 for stable dot product; dy will be cast to bf16 in backward wrapper
        loss = (y_real.to(torch.float32) * weight_seq.to(torch.float32)).sum()

    loss.backward()

    # Extract grads for real tokens only
    dw = w_pad.grad[0, sl].contiguous()
    dq = q_pad.grad[0, sl].contiguous()
    dk = k_pad.grad[0, sl].contiguous()
    dv = v_pad.grad[0, sl].contiguous()
    da = a_pad.grad[0, sl].contiguous()
    db = b_pad.grad[0, sl].contiguous()

    return dw, dq, dk, dv, da, db


def test_s_chunk_checkpoint_writes_and_last_partial_chunk_nan():
    """
    Catches:
    - checkpoint condition bug ((p+1)%CHUNK) wrong
    - chunk indexing bug
    - missing writes to s_chunk
    - accidental writes into last partial chunk

    Strategy:
    - prefill s_chunk with NaNs
    - run varlen forward
    - assert:
      * all FULL chunks [0:num_full_chunks) are finite
      * last PARTIAL chunk (if any) remains NaN
    """
    print("\n" + "="*60)
    print("TEST: s_chunk Checkpoint Writes + Last Partial Chunk NaN")
    print("="*60)

    H = 2
    C = HEAD_SIZE

    # Choose total_tokens NOT multiple of CHUNK_LEN
    seq_lengths = [17, 29, 3, 64, 18]  # total=131, 131%16=3
    total_tokens = sum(seq_lengths)
    num_chunks = (total_tokens + CHUNK_LEN - 1) // CHUNK_LEN
    num_full_chunks = total_tokens // CHUNK_LEN  # floor

    w, q, k, v, a, b, cu_seqlens = create_test_data(seq_lengths, H, C, requires_grad=False)

    y = torch.full((total_tokens, H, C), float('nan'), dtype=torch.bfloat16, device='cuda')
    s_chunk = torch.full((H, num_chunks, C, C), float('nan'), dtype=torch.float32, device='cuda')
    sa = torch.full((total_tokens, H, C), float('nan'), dtype=torch.float32, device='cuda')

    torch.ops.wind_backstepping_varlen.forward_varlen(w, q, k, v, a, b, cu_seqlens, y, s_chunk, sa)

    # Basic finite checks
    y_ok = torch.isfinite(y).all().item()
    sa_ok = torch.isfinite(sa).all().item()

    # Check full chunks written
    full_ok = True
    if num_full_chunks > 0:
        full_ok = torch.isfinite(s_chunk[:, :num_full_chunks]).all().item()

    # Check last partial chunk remains NaN (only if partial exists)
    partial_ok = True
    if total_tokens % CHUNK_LEN != 0:
        # There is a partial chunk at index num_full_chunks == num_chunks-1
        partial = s_chunk[:, num_full_chunks:]
        # Expect untouched => still NaN everywhere
        partial_ok = torch.isnan(partial).all().item()

    passed = y_ok and sa_ok and full_ok and partial_ok
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | y finite={y_ok}, sa finite={sa_ok}, full_chunks_finite={full_ok}, partial_chunk_nan={partial_ok}")

    return passed


def test_backward_ref_aligned_multi_seq_misaligned_starts():
    """
    Main missing test from round 3:
    - Multiple sequences packed together
    - Start offsets NOT multiples of CHUNK_LEN
    - Lengths NOT multiples of CHUNK_LEN
    - total_tokens NOT multiple of CHUNK_LEN (exercises last partial chunk path)
    - Uses Ref-Aligned original reference (prefix no-op padding)

    Catches:
    - p0 off-by-one
    - wrong checkpoint chunk index
    - wrong transpose layout between forward store / backward load
    - subtle boundary errors that don't show up in single-seq aligned tests
    """
    print("\n" + "="*60)
    print("TEST: Backward Ref-Aligned | Multi-Seq + Misaligned Starts")
    print("="*60)

    H = 2
    C = HEAD_SIZE
    seq_lengths = [17, 29, 3, 64, 18]  # misaligned + total not multiple of 16
    total_tokens = sum(seq_lengths)

    # Create packed varlen inputs (leaf tensors)
    w, q, k, v, a, b, cu_seqlens = create_test_data(seq_lengths, H, C, requires_grad=True)

    # Random weights to avoid accidental cancellation
    torch.manual_seed(123)
    weight = torch.randn((total_tokens, H, C), dtype=torch.float32, device='cuda')

    # Varlen forward/backward
    y_var = VarlenRWKV7.apply(w, q, k, v, a, b, cu_seqlens)
    loss_var = (y_var.to(torch.float32) * weight).sum()
    loss_var.backward()

    grads_var = [w.grad, q.grad, k.grad, v.grad, a.grad, b.grad]
    grad_names = ["dw", "dq", "dk", "dv", "da", "db"]

    # Reference: per-sequence original with ref-aligned prefix no-op
    tol = 1e-2
    all_passed = True

    # Walk sequences
    for s in range(len(seq_lengths)):
        start = int(cu_seqlens[s].item())
        end = int(cu_seqlens[s + 1].item())
        L = end - start
        if L <= 0:
            continue

        # Slices of values (detach so ref run independent)
        w_s = w.detach()[start:end].clone()
        q_s = q.detach()[start:end].clone()
        k_s = k.detach()[start:end].clone()
        v_s = v.detach()[start:end].clone()
        a_s = a.detach()[start:end].clone()
        b_s = b.detach()[start:end].clone()
        wt_s = weight[start:end].clone()

        ref_dw, ref_dq, ref_dk, ref_dv, ref_da, ref_db = _ref_aligned_original_seq_grads(
            start_offset=start,
            w_seq=w_s, q_seq=q_s, k_seq=k_s, v_seq=v_s, a_seq=a_s, b_seq=b_s,
            weight_seq=wt_s,
        )
        refs = [ref_dw, ref_dq, ref_dk, ref_dv, ref_da, ref_db]

        print(f"\n  Sequence {s}: start={start}, len={L}, start%CHUNK={start % CHUNK_LEN}")

        # Compare each grad tensor on REAL token region
        for name, g_var, g_ref in zip(grad_names, grads_var, refs):
            if g_var is None:
                print(f"    {name}: SKIP (var grad None)")
                continue
            gv = g_var[start:end].contiguous()
            gr = g_ref.contiguous()

            # Finite check first (catches NaN from wrong checkpoint load)
            finite_ok = torch.isfinite(gv).all().item() and torch.isfinite(gr).all().item()
            max_diff = (gv - gr).abs().max().item()
            passed = finite_ok and (max_diff < tol)
            status = "✓" if passed else "✗"
            print(f"    {status} {name}: max_diff={max_diff:.6e}, finite_ok={finite_ok}")
            if not passed:
                all_passed = False

    return all_passed


def test_backward_ref_aligned_boundary_endcases():
    """
    Targeted p0/off-by-one catcher:
    Create a misaligned sequence whose p_end is EXACTLY a global checkpoint boundary (p_end+1 % CHUNK == 0).

    If p0 math is wrong, backward may:
    - load wrong chunk (possibly NaN)
    - or replay wrong range
    => large diff / NaN vs ref-aligned reference.
    """
    print("\n" + "="*60)
    print("TEST: Backward Ref-Aligned | p_end at Global Boundary (Misaligned Start)")
    print("="*60)

    H = 2
    C = HEAD_SIZE

    # seq0 length 17 => seq1 starts at 17 (misaligned, 17%16=1)
    # seq1 length 15 => seq1 ends at global p_end=31 which is a checkpoint boundary (31+1=32)
    seq_lengths = [17, 15]
    total_tokens = sum(seq_lengths)

    w, q, k, v, a, b, cu_seqlens = create_test_data(seq_lengths, H, C, requires_grad=True)

    torch.manual_seed(456)
    weight = torch.randn((total_tokens, H, C), dtype=torch.float32, device='cuda')

    y_var = VarlenRWKV7.apply(w, q, k, v, a, b, cu_seqlens)
    loss_var = (y_var.to(torch.float32) * weight).sum()
    loss_var.backward()

    tol = 1e-2
    all_passed = True
    grad_names = ["dw", "dq", "dk", "dv", "da", "db"]
    grads_var = [w.grad, q.grad, k.grad, v.grad, a.grad, b.grad]

    for s in range(len(seq_lengths)):
        start = int(cu_seqlens[s].item())
        end = int(cu_seqlens[s + 1].item())
        L = end - start
        if L <= 0:
            continue

        w_s = w.detach()[start:end].clone()
        q_s = q.detach()[start:end].clone()
        k_s = k.detach()[start:end].clone()
        v_s = v.detach()[start:end].clone()
        a_s = a.detach()[start:end].clone()
        b_s = b.detach()[start:end].clone()
        wt_s = weight[start:end].clone()

        ref = _ref_aligned_original_seq_grads(
            start_offset=start,
            w_seq=w_s, q_seq=q_s, k_seq=k_s, v_seq=v_s, a_seq=a_s, b_seq=b_s,
            weight_seq=wt_s,
        )

        print(f"\n  Sequence {s}: start={start}, len={L}, p_end={end-1}, (p_end+1)%CHUNK={(end)%CHUNK_LEN}")

        for name, g_var, g_ref in zip(grad_names, grads_var, ref):
            gv = g_var[start:end].contiguous()
            gr = g_ref.contiguous()
            finite_ok = torch.isfinite(gv).all().item() and torch.isfinite(gr).all().item()
            max_diff = (gv - gr).abs().max().item()
            passed = finite_ok and (max_diff < tol)
            status = "✓" if passed else "✗"
            print(f"    {status} {name}: max_diff={max_diff:.6e}, finite_ok={finite_ok}")
            if not passed:
                all_passed = False

    return all_passed


def test_w_input_extremes_stability_and_match():
    """
    Stress numeric extremes of w_input:
    - very negative => w ~ 1
    - around 0 => w ~ exp(-1)
    - moderately positive => w tiny but nonzero (avoid underflow-to-0)

    Uses a single aligned sequence (start=0, length multiple of CHUNK) so schedule matches directly,
    expecting close match vs original for both forward and backward.
    """
    print("\n" + "="*60)
    print("TEST: Numerical Extremes of w_input (Forward+Backward Match + Finite)")
    print("="*60)

    H = 2
    C = HEAD_SIZE
    T = 32  # multiple of 16

    scale = 0.1
    device = 'cuda'

    # Build base random inputs
    q = (torch.randn(T, H, C, dtype=torch.bfloat16, device=device) * scale).detach().requires_grad_(True)
    k = (torch.randn(T, H, C, dtype=torch.bfloat16, device=device) * scale).detach().requires_grad_(True)
    v = (torch.randn(T, H, C, dtype=torch.bfloat16, device=device) * scale).detach().requires_grad_(True)
    a = (torch.randn(T, H, C, dtype=torch.bfloat16, device=device) * scale).detach().requires_grad_(True)
    b = (torch.randn(T, H, C, dtype=torch.bfloat16, device=device) * scale).detach().requires_grad_(True)

    # Construct w_input with channel-wise extremes
    w_fp32 = torch.empty((T, H, C), dtype=torch.float32, device=device)
    q1 = C // 4
    q2 = C // 2
    q3 = (3 * C) // 4
    w_fp32[..., :q1] = -100.0   # w≈1
    w_fp32[..., q1:q2] = 0.0    # w≈exp(-1)
    w_fp32[..., q2:q3] = 2.0    # w=exp(-exp(2)) ~ exp(-7.389) ~ 6.17e-4
    w_fp32[..., q3:] = 4.0      # w=exp(-exp(4)) ~ exp(-54.6) ~ 1.8e-24 (tiny but nonzero in fp32)
    w = w_fp32.to(torch.bfloat16).detach().requires_grad_(True)

    # Build ORIGINAL format tensors (B=1,T,H,C)
    w_orig = w.detach().clone().reshape(1, T, H, C).contiguous().requires_grad_(True)
    q_orig = q.detach().clone().reshape(1, T, H, C).contiguous().requires_grad_(True)
    k_orig = k.detach().clone().reshape(1, T, H, C).contiguous().requires_grad_(True)
    v_orig = v.detach().clone().reshape(1, T, H, C).contiguous().requires_grad_(True)
    a_orig = a.detach().clone().reshape(1, T, H, C).contiguous().requires_grad_(True)
    b_orig = b.detach().clone().reshape(1, T, H, C).contiguous().requires_grad_(True)

    # VARLEN format already (T,H,C)
    w_var = w.detach().clone().contiguous().requires_grad_(True)
    q_var = q.detach().clone().contiguous().requires_grad_(True)
    k_var = k.detach().clone().contiguous().requires_grad_(True)
    v_var = v.detach().clone().contiguous().requires_grad_(True)
    a_var = a.detach().clone().contiguous().requires_grad_(True)
    b_var = b.detach().clone().contiguous().requires_grad_(True)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    # Same weighted loss to avoid cancellation
    torch.manual_seed(789)
    weight = torch.randn((T, H, C), dtype=torch.float32, device=device)

    # Forward
    y_o = OriginalRWKV7.apply(w_orig, q_orig, k_orig, v_orig, a_orig, b_orig)[0]
    y_v = VarlenRWKV7.apply(w_var, q_var, k_var, v_var, a_var, b_var, cu_seqlens)

    # Forward compare
    max_diff_y = (y_o - y_v).abs().max().item()
    y_finite = torch.isfinite(y_o).all().item() and torch.isfinite(y_v).all().item()

    # Backward
    loss_o = (y_o.to(torch.float32) * weight).sum()
    loss_v = (y_v.to(torch.float32) * weight).sum()
    loss_o.backward()
    loss_v.backward()

    tol = 1e-2
    grads_ok = True
    pairs = [
        ("dw", w_orig.grad.reshape(T, H, C), w_var.grad),
        ("dq", q_orig.grad.reshape(T, H, C), q_var.grad),
        ("dk", k_orig.grad.reshape(T, H, C), k_var.grad),
        ("dv", v_orig.grad.reshape(T, H, C), v_var.grad),
        ("da", a_orig.grad.reshape(T, H, C), a_var.grad),
        ("db", b_orig.grad.reshape(T, H, C), b_var.grad),
    ]

    print(f"  Forward: finite={y_finite}, max_diff={max_diff_y:.6e}")
    if (not y_finite) or (max_diff_y >= tol):
        grads_ok = False

    for name, go, gv in pairs:
        finite_ok = torch.isfinite(go).all().item() and torch.isfinite(gv).all().item()
        max_diff = (go - gv).abs().max().item()
        passed = finite_ok and (max_diff < tol)
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: max_diff={max_diff:.6e}, finite_ok={finite_ok}")
        if not passed:
            grads_ok = False

    return grads_ok


def test_torch_check_validations_expected_errors():
    """
    Optional but useful: verify TORCH_CHECK validations really trigger.
    This will FAIL if those checks are missing (which is good feedback).

    Covers:
    - empty total_tokens rejection (total_tokens==0)
    - cu_seqlens wrong dtype
    - non-contiguous input
    """
    print("\n" + "="*60)
    print("TEST: TORCH_CHECK Validations (Expected Errors)")
    print("="*60)

    device = "cuda"
    H = 2
    C = HEAD_SIZE
    ok = True

    # Case 1: total_tokens == 0
    try:
        w = torch.empty((0, H, C), dtype=torch.bfloat16, device=device)
        q = torch.empty_like(w)
        k = torch.empty_like(w)
        v = torch.empty_like(w)
        a = torch.empty_like(w)
        b = torch.empty_like(w)
        cu = torch.tensor([0, 0], dtype=torch.int32, device=device)  # num_seqs=1 but empty
        y = torch.empty_like(w)
        s_chunk = torch.empty((H, 0, C, C), dtype=torch.float32, device=device)
        sa = torch.empty((0, H, C), dtype=torch.float32, device=device)

        torch.ops.wind_backstepping_varlen.forward_varlen(w, q, k, v, a, b, cu, y, s_chunk, sa)
        print("  ✗ FAIL: expected error for total_tokens==0, but no error raised")
        ok = False
    except Exception as e:
        print(f"  ✓ PASS: total_tokens==0 raised error ({type(e).__name__})")

    # Case 2: cu_seqlens wrong dtype (int64)
    try:
        seq_lengths = [16]
        w, q, k, v, a, b, cu = create_test_data(seq_lengths, H, C, requires_grad=False)
        cu_bad = cu.to(torch.int64)
        total_tokens = sum(seq_lengths)
        num_chunks = (total_tokens + CHUNK_LEN - 1) // CHUNK_LEN
        y = torch.empty_like(w)
        s_chunk = torch.empty((H, num_chunks, C, C), dtype=torch.float32, device=device)
        sa = torch.empty((total_tokens, H, C), dtype=torch.float32, device=device)

        torch.ops.wind_backstepping_varlen.forward_varlen(w, q, k, v, a, b, cu_bad, y, s_chunk, sa)
        print("  ✗ FAIL: expected error for cu_seqlens int64, but no error raised")
        ok = False
    except Exception as e:
        print(f"  ✓ PASS: cu_seqlens int64 raised error ({type(e).__name__})")

    # Case 3: non-contiguous w
    try:
        total_tokens = 16
        # make non-contiguous but correct shape (total_tokens, H, C)
        w_nc = (torch.randn(H, total_tokens, C, dtype=torch.bfloat16, device=device) * 0.1).permute(1, 0, 2)  # non-contig
        q = torch.randn((total_tokens, H, C), dtype=torch.bfloat16, device=device) * 0.1
        k = torch.randn((total_tokens, H, C), dtype=torch.bfloat16, device=device) * 0.1
        v = torch.randn((total_tokens, H, C), dtype=torch.bfloat16, device=device) * 0.1
        a = torch.randn((total_tokens, H, C), dtype=torch.bfloat16, device=device) * 0.1
        b = torch.randn((total_tokens, H, C), dtype=torch.bfloat16, device=device) * 0.1
        cu = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
        num_chunks = (total_tokens + CHUNK_LEN - 1) // CHUNK_LEN
        y = torch.empty((total_tokens, H, C), dtype=torch.bfloat16, device=device)
        s_chunk = torch.empty((H, num_chunks, C, C), dtype=torch.float32, device=device)
        sa = torch.empty((total_tokens, H, C), dtype=torch.float32, device=device)

        torch.ops.wind_backstepping_varlen.forward_varlen(w_nc, q.contiguous(), k.contiguous(), v.contiguous(),
                                                         a.contiguous(), b.contiguous(), cu, y, s_chunk, sa)
        print("  ✗ FAIL: expected error for non-contiguous w, but no error raised")
        ok = False
    except Exception as e:
        print(f"  ✓ PASS: non-contiguous w raised error ({type(e).__name__})")

    return ok
```

---

### Gợi ý tích hợp vào `__main__`

Bạn chỉ cần thêm vào phần cuối (không bắt buộc, nhưng để auto-run):

```python
results['s_chunk_writes'] = test_s_chunk_checkpoint_writes_and_last_partial_chunk_nan()
results['backward_ref_aligned_misaligned'] = test_backward_ref_aligned_multi_seq_misaligned_starts()
results['backward_ref_aligned_boundary'] = test_backward_ref_aligned_boundary_endcases()
results['w_input_extremes'] = test_w_input_extremes_stability_and_match()
results['torch_checks'] = test_torch_check_validations_expected_errors()
```

---

## Tóm tắt ưu tiên

* ✅ p0 bitmask mới + static_assert: **đúng và cần thiết** (giải quyết “time-bomb” lớn nhất).
* ✅ TORCH_CHECK dtype/contiguous/CUDA: **rất tốt**, nhưng để “full validation” thì nên **thêm shape checks + cu_seqlens semantic checks** như mình liệt kê.
* ⚠️ Reject `total_tokens==0`: tùy policy; test mình viết sẽ confirm đúng behavior bạn muốn.

Các test mới ở trên sẽ cho bạn “evidence mạnh” đúng cái còn thiếu: **backward packed + misaligned + ref-aligned**, và **s_chunk missing writes/partial-chunk**—đây là nơi bug indexing/off-by-one hay lẩn trốn nhất. 
