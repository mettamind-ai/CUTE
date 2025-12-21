Below is a **deliberately conservative** test strategy designed to catch the *specific* failure modes your varlen design is vulnerable to:

* global checkpoint alignment (misaligned vs per-seq)
* tail forward replay off-by-one
* boundary leakage (state / gradient carry)
* wrong indexing (seq/head/token stride)
* missing writes / uninitialized outputs
* edge cases (empty seq, tiny seq, huge num_seqs)

I’ll also call out one critical gotcha: **your proposed “pad to CHUNK_LEN and slice” ground truth is forward-correct, but can be backward-mismatched** unless you control padding and checkpoint alignment.

---

## 0) The single biggest gotcha (fix your “ground truth” plan)

### Why your current padding ground truth can fail (even if varlen is correct)

Your varlen kernel uses **global** chunk checkpoints at packed positions where `(p+1) % 16 == 0`.

If you run the original kernel per sequence, its checkpoints happen at local `t=15,31,...` (aligned to seq start).
For sequences that start at a packed offset `start % 16 != 0`, **checkpoint boundaries inside the sequence differ** between:

* varlen (global): checkpoint when `(start + t + 1) % 16 == 0`
* per-seq original: checkpoint when `(t + 1) % 16 == 0`

This difference can change reconstruction drift patterns and lead to small gradient deltas even if both are “mathematically correct.”

### Fix: make the per-sequence reference emulate packed-global chunk alignment

For each sequence `s` with packed start index `start = cu_seqlens[s]`:

1. **Prefix-pad** by `prefix = start % 16` tokens
2. Place real tokens after that prefix
3. **Suffix-pad** to the next multiple of 16
4. Make BOTH prefix and suffix tokens be **no-op tokens** (so they do not change the state)

This makes the original kernel’s local chunk boundaries match the packed-global ones in the real region.

### No-op token recipe (works with your kernel math)

Choose padding token inputs so that the state update becomes identity:

We want, per padding token: `S_t = S_{t-1}`.

Sufficient conditions:

* `w = 1` exactly
* `k = 0`
* `b = 0`
* `v = 0` (optional if `k=0`)

Concrete:

* `w_input = -10000` (bf16) → `exp(w_input)` underflows to 0 → `w = exp(-0)=1` exactly
* `k[:] = 0`, `b[:] = 0`, `v[:] = 0`
* set `q[:] = 0`, `a[:] = 0` too (not required but simplifies)

Then:

* forward update uses only `S_prev * 1 + 0 + 0`
* backward reconstruction across padding is exact and harmless

**This eliminates the “padding affects backward due to reconstruction error” worry.**

---

## 1) Complete test case list (with rationale)

I’d structure your suite into 6 buckets: **smoke**, **tail replay**, **alignment**, **boundary**, **edge**, **stress/fuzz**.

### Bucket A — Minimal smoke tests (fast, catches 80% of indexing bugs)

1. **Single empty sequence**

* lengths: `[0]`
* verifies: kernel handles L=0 without illegal memory access, writes nothing, gradients are empty/zero

2. **Single token**

* lengths: `[1]`
* verifies: base case; state init/reset; backward terminal condition; replay path when no chunk exists

3. **Max short (<CHUNK_LEN)**

* lengths: `[15]`
* verifies: tail replay path that starts from state=0 and replays whole seq

4. **Exactly CHUNK_LEN**

* lengths: `[16]`
* verifies: checkpoint boundary exact; no tail replay required; no off-by-one at `(p+1)&15==0`

5. **Just over CHUNK_LEN**

* lengths: `[17]`
* verifies: tail replay length = 1 (if end not aligned), first internal checkpoint behavior

6. **Two sequences small**

* lengths: `[3, 2]`
* verifies: boundary handling at small sizes, no leakage of state

7. **Your canonical example**

* lengths: `[3, 2, 4]`
* verifies: multiple boundaries, backward boundary reset is correct

---

### Bucket B — Tail replay sweep (directly targets replay logic, 0..15 steps)

You want to guarantee you hit every tail replay length `r ∈ {0..15}` where:

[
r = (p_{end} + 1) \bmod 16
]

**Construct test where each sequence ends at a different remainder.**

Practical deterministic construction:

* Make one big packed list of sequences such that you can predict `start` and `end` mod 16.
* Easiest: start with a “filler” prefix of no-op sequences to achieve desired start offsets.

Recommended test case:

* lengths: `[1]*16 + [32 + k for k in range(16)]`

  * First 16 sequences of length 1 create starts at offsets 0..15.
  * Next 16 longer sequences ensure internal checkpoints exist and end remainders vary.

What it catches:

* replay off-by-one (`p0` computation)
* replay loop bounds (`p0+1..p_end`)
* using `sa_t` from wrong timestep
* wrong indexing of `s_chunk` (chunk id computed from global `p`)

---

### Bucket C — Start-offset / global checkpoint alignment (critical because you use global `s_chunk`)

These tests force sequences to start at nasty offsets:

1. **Sequence starts exactly at global chunk end**

* lengths: `[15, 20]` → second sequence starts at `p=15`, which is chunk end (`p+1` multiple of 16)
* catches: checkpoint load/store when first token is chunk boundary; chunk-id math

2. **Sequence starts at every offset 0..15 with length > 32**

* e.g. lengths: `[1]*16 + [64]*4`
* ensures multiple sequences with different start offsets traverse multiple chunk boundaries

3. **Sequence ends exactly at global chunk end**

* choose lengths so `p_end % 16 == 15`
* catches: replay length 0 path vs >0 path

---

### Bucket D — Boundary bug traps (forward + backward “no cross-seq influence”)

These are crafted to catch “accidental carry” even if basic comparisons pass.

1. **Alternating tiny and long**

* lengths: `[1, 64, 1, 64, 1, 64]`
* catches: any persistent state or gradient reuse across blocks/seqs

2. **Many short sequences**

* lengths: `[1]*128`
* catches: off-by-one in cu_seqlens, writing outside bounds, missing writes

3. **Include empty sequences between non-empty**

* lengths: `[8, 0, 9, 0, 1, 0, 17]`
* catches: cu_seqlens repeated values; skipping logic; start/end correctness

---

### Bucket E — Input pattern tests (make debugging easier and catch “silent” mistakes)

Use same lengths as some bucket above, but swap input distributions:

1. **All zeros** (except `w_input` in a safe range, see below)

* expected: state updates depend mostly on `k*v` and `sa*b`; with zeros these collapse
* catches: uninitialized memory usage; non-determinism

2. **Constant ones** (bounded)

* catches: sign mistakes, indexing mistakes that random noise might hide

3. **One-hot / structured**

* Example: set `k` one-hot per token, `v` one-hot per row
* catches: transpose/stride mistakes (wrong dimension treated as head/channel)

4. **Random but range-controlled** (most important)

* Use distributions that avoid `w≈0` division blow-ups:

  * suggest `w_input ~ Uniform(-6, 2)` in float32 then cast to bf16
  * `q,k,v,a,b ~ Normal(0, 0.5)` or Uniform(-1,1)

---

### Bucket F — Stress & fuzz (find rare off-by-one / indexing corruption)

1. **Random lengths fuzz**

* for N iterations (e.g., 200):

  * num_seqs random in [1, 64]
  * lengths random in [0, 128] with a bias toward small values (0..20)
* This is where “almost correct” kernels fail.

2. **Grid limit test (>65535 sequences) without huge memory**

* Use **mostly empty sequences** so total_tokens stays small:

  * lengths: `[0]*70000 + [16]`
* Verifies:

  * your 1D grid mapping works
  * kernels early-return safely for empty sequences
  * wrapper / dispatch doesn’t use grid.y

3. **Large total_tokens sanity**

* lengths: e.g. 256 sequences of avg length 256 (total 65536)
* mainly a “does it run and match?” test; also catches overflow in indexing math

---

## 2) Numerical tolerance recommendations (what to expect, and what to assert)

### Best-case (recommended): expect **bitwise exact bf16 match**

If you:

* compare varlen against **reference with prefix/suffix no-op padding** (alignment fix above)
* and your varlen code uses the *same math* (`__expf`, same loop order)

Then you should aim for:

* **Forward:** `torch.equal(y_varlen, y_ref_packed)`
* **Backward:** `torch.equal(dw_varlen, dw_ref_packed)` etc for all grads

This is the strongest possible test.

### Practical fallback (if exact match is not stable across GPUs / compiler)

If exact equality fails but you believe math is same, use bf16-aware tolerances.

Because outputs and grads are bf16, a reasonable general tolerance is:

* **Forward (bf16):** `atol=2e-2`, `rtol=1e-2`
* **Backward grads (bf16):** `atol=5e-2`, `rtol=2e-2`

Justification (defensible, not hand-wavy):

* bf16 has ~7-bit mantissa → relative quantization ~0.78% near magnitude 1
* many ops + exp + recurrence can amplify error
* grads can be larger magnitude and noisier than forward outputs

### Strong recommendation: report both

Even if you assert with allclose, always log:

* max abs diff
* max rel diff
* count of elements exceeding tolerance

This makes failures actionable.

---

## 3) Gradient leakage test design (your idea is right; here’s how to make it bulletproof)

Your sketch is correct. To make it “definitive”:

### Design goals

* Loss depends **only** on sequence B tokens
* Gradients for A and C must be **exactly zero** (not just small)
* Ensure B actually has nonzero gradients (avoid vacuous pass)

### A robust implementation pattern

1. Choose lengths that stress boundaries + replay:

* Example: `[15, 17, 14]`

  * B starts at global offset 15 (nasty)
  * B length 17 requires tail replay
  * C short stresses replay-from-zero

2. Generate random inputs (range-controlled).

3. Run varlen forward → y packed.

4. Define loss using **only B slice**:

```python
startB = cu[1]; endB = cu[2]
loss = (y[startB:endB].float() ** 2).sum()
```

Square ensures gradient is proportional to y, almost surely nonzero.

5. Backward.

6. Assertions:

* **Non-vacuous:** ensure B grads are nonzero

```python
assert (dq[startB:endB].abs().sum() > 0)
# or any of dw/dk/dv/da/db
```

* **No leakage:** A and C grads exactly zero

```python
assert (dq[:startB] == 0).all()
assert (dq[endB:] == 0).all()
# Repeat for dw, dk, dv, da, db
```

Why “exact zero” is fair here:

* If sequences are truly disjoint in your kernel, there is literally no computational path.
* Any nonzero means either:

  * wrong indexing/writes
  * accidental cross-seq processing
  * memory aliasing bug

### Extra leakage tests (high value)

* Loss on **first** sequence only and verify others zero.
* Loss on **last** sequence only and verify others zero.
* Loss on a **single token** inside B (e.g., middle token) to catch off-by-one boundaries.

---

## 4) Backward correctness tests (what to compare and what NOT to rely on)

### Compare each gradient tensor separately

Yes. Do all of:

* `dw, dq, dk, dv, da, db`

And compare packed slices exactly aligned to real tokens.

Also compare:

* gradients for padding/no-op prefix/suffix are zero in reference runs (sanity)

### About `torch.autograd.gradcheck`

I would **not** make gradcheck your main proof because:

* your op is bf16-only
* `gradcheck` expects float64 stability and uses finite differences
* exp + recurrence + bf16 quantization makes finite differences noisy

**Better alternative (optional, but strong):** a small float64 reference for tiny T/H
For example:

* H=1
* T up to 8
* compute forward in float64 using the *matrix* formulation
* let PyTorch autograd produce gradients in float64
* compare kernels (cast to float32/bf16) with looser tolerances

This catches “both kernels share same bug” (unlikely, but this is how you get confidence beyond A/B diff).

---

## 5) Edge cases: what exactly to verify

### Empty sequence (L=0)

Verify:

* forward output has zero tokens for that seq
* backward does not crash
* cu_seqlens with repeats doesn’t cause illegal memory access
* leakage test still holds

### Single token (L=1)

Verify:

* forward matches reference
* backward matches reference
* tail replay starts from state=0 and replays exactly 1 step (if no checkpoint)

### L < CHUNK_LEN

Verify:

* replay-from-zero path is used
* gradients correct
* no hidden assumption about chunk checkpoint presence

### Boundary exactly at chunk boundary

Construct packed so `p_end % 16 == 15` for some seq.
Verify:

* tail replay length 0 path correct
* state initialization and checkpoint load paths correct

### Very long sequence

Verify:

* periodic checkpoint reload path (inside backward loop) triggers many times
* no drift explosion
* matches reference

---

## 6) Stress tests: yes, but keep them “orthogonal”

I recommend including:

1. **Random fuzz** (correctness)

* catches rare off-by-one, especially in p0 computation and chunk indexing

2. **Many sequences** (launch/indexing)

* especially `num_seqs > 65535` with mostly empty seqs

3. **Memory pressure** (stability, not correctness)

* run a large case and ensure:

  * no OOM
  * no NaNs
  * outputs and grads match reference within tolerance

Keep memory/perf tests separate so correctness failures aren’t buried.

---

## 7) Answer to your ground truth question: “Will padding affect valid outputs?”

### Forward

You were right: **padding after valid tokens does not affect forward outputs** for valid tokens (causal recurrence).

### Backward (the subtle part)

Mathematically, padding with loss ignored **should not** affect gradients for valid tokens.

But because your backward uses **finite-precision state reconstruction**, random padding tokens can introduce reconstruction error when you backtrack across them. That error can perturb the reconstructed state at the last valid token and thus perturb gradients.

✅ The fix is what I described in Section 0:

* use **no-op padding tokens**
* and (if you want to match varlen’s global checkpoint scheme) use **prefix no-op padding** of `start % 16`

With that, your “pad and slice” ground truth becomes solid.

---

## 8) Summary checklist (things you should not miss)

**Must-have in the test harness:**

* ✅ reference per-seq run uses **prefix padding = start%16** (no-op) to emulate global chunk alignment
* ✅ suffix padding to multiple of 16 is **no-op**
* ✅ loss uses only real tokens (ensures dy=0 on padding)
* ✅ compare packed outputs/grads assembled in packed order
* ✅ include a gradient leakage test with “loss on B only” and assert A/C grads **exactly 0**
* ✅ fuzz random lengths biased to small lengths and include empty sequences
* ✅ include a `num_seqs > 65535` launch test (mostly empty)

If you follow this, you’ll have a suite that:

* catches off-by-one replay bugs
* catches boundary leaks
* catches chunk-index errors
* avoids false failures due to checkpoint alignment mismatch

If you want, I can turn this into a concrete pytest-style skeleton (helper functions for no-op tokens, reference runner with prefix/suffix, pack/unpack comparisons, and a parameterized list of cases).

----

# tự phản biện kết luận trên. mục đích là tìm ra những kẽ hở và sửa chữa

Dưới đây là **tự phản biện** (self‑rebuttal) cho kế hoạch test mình vừa đề xuất, mục tiêu là chỉ ra **kẽ hở**, vì sao nó nguy hiểm, và **cách sửa** sao cho test suite vừa “đánh trúng bug”, vừa tránh false fail / false pass.

---

# 1) Kẽ hở số 1: Mình đã “ngầm đổi định nghĩa ground truth” bằng prefix padding

Trong câu trả lời trước, mình nói “ground truth per‑sequence” nhưng rồi lại đề xuất **prefix no‑op padding = start%16** để “align global checkpoints”.

## Vì sao đây là kẽ hở?

Có **2 định nghĩa đúng** khác nhau:

### (A) “Đúng theo original kernel run per‑sequence”

* Bạn chạy original kernel trên từng sequence riêng (t=0..L-1), checkpoint ở local `(t+1)%16==0`.
* Đây là “baseline semantics” mà bạn nói muốn match.

### (B) “Đúng theo varlen implementation với global checkpoint schedule”

* Checkpoint/reload ở vị trí **global** `(p+1)%16==0`.
* Với sequence bắt đầu lệch pha, schedule checkpoint bên trong sequence **khác** (A).
* Về toán học đều hợp lệ (checkpoint chỉ là “numerical stabilization”), nhưng **kết quả bf16 gradient có thể khác** vì drift khi reconstruct phụ thuộc checkpoint schedule.

Khi mình thêm **prefix padding**, mình đã biến reference (A) thành một reference kiểu (B) (tức là “reference đã bị uốn để giống varlen”), nên:

* Nếu bạn muốn “match original per‑sequence baseline”, test đó **không còn chứng minh điều này** nữa.
* Nó vẫn rất hữu ích để debug, nhưng **không đủ** để chứng minh “varlen == original”.

## Sửa thế nào cho chặt?

Bạn nên duy trì **2 reference** song song:

1. **Ref‑Local (baseline):**
   chạy original kernel per-seq, **chỉ suffix pad** lên bội 16 (no-op), *không prefix pad*.
   → Đây là “ground truth theo original”.

2. **Ref‑Aligned (instrumentation):**
   chạy original kernel per-seq, **prefix pad = start%16 no-op** + suffix pad no-op.
   → Đây là “ground truth theo checkpoint schedule của varlen” (giúp bắt bug indexing/replay, kỳ vọng match rất sát/đúng bit).

**Tiêu chí pass đề xuất:**

* Varlen vs Ref‑Aligned: **rất chặt** (lý tưởng là exact bf16).
* Varlen vs Ref‑Local: **cho phép sai số nhỏ** (vì khác schedule), nhưng **phải bị chặn** bằng một bound hợp lý (xem mục 3).

Điểm hay: nếu varlen sai logic, nó thường fail Ref‑Aligned **ngay lập tức**; nếu varlen đúng logic nhưng khác schedule, Ref‑Aligned pass còn Ref‑Local chỉ lệch rất nhỏ.

---

# 2) Kẽ hở số 2: “Prefix align để tránh false fail” có thể trở thành **false pass** nếu mục tiêu là baseline

Ngay cả khi varlen có bug khiến kết quả lệch so với baseline, nhưng bug đó lại “gần” với Ref‑Aligned do mình “align reference”, bạn có thể bị **ảo tưởng** rằng “varlen đúng”.

Ví dụ dễ xảy ra:

* Varlen backward reload checkpoint theo global `(p+1)%16==0` (đúng theo thiết kế).
* Nhưng bạn thực sự muốn match original per‑seq local schedule `(t+1)%16==0`.
* Ref‑Aligned giúp varlen pass, nhưng baseline vẫn lệch (và có thể lệch đủ lớn để training khác).

➡️ **Sửa:** luôn có test so với Ref‑Local. Ref‑Aligned chỉ dùng để *pinpoint bug*, không được thay thế baseline.

---

# 3) Kẽ hở số 3: Tolerance mình đưa ra hơi “võ đoán” và có thể che bug

Mình đưa `atol/rtol` kiểu 2e-2, 5e-2… Đây có 2 vấn đề:

1. Nếu tolerance **quá rộng**, bạn có thể pass dù có bug nhẹ (off-by-one replay, sai stride head…) tạo sai số “không quá lớn”.
2. Nếu tolerance **quá chặt**, bạn có thể fail vì sai số do schedule drift (không phải bug).

## Sửa: dùng tolerance “data-driven” từ chính 2 reference

Cách chặt nhất (và rất hợp với “evidence standard”) là:

* Tính **độ lệch do schedule**:
  [
  \Delta_{\text{schedule}} = | \text{Ref-Local} - \text{Ref-Aligned} |
  ]
  cho forward và từng gradient tensor.

* Khi đó bạn đặt bound cho varlen vs Ref‑Local kiểu:
  [
  | \text{Varlen} - \text{Ref-Local} | \le \alpha \cdot \Delta_{\text{schedule}} + \epsilon
  ]
  với (\alpha) khoảng 1.2–2.0 (bảo thủ), (\epsilon) là slack nhỏ.

Điều này biến tolerance từ “ước lượng” thành **được chứng minh bằng thực nghiệm nội bộ**: “varlen lệch không được lớn hơn lệch mà chỉ riêng schedule đã gây ra.”

### Gợi ý cụ thể

* Dùng metric `max_abs` và `max_rel` theo tensor:

  * `max_abs = (x-y).abs().max()`
  * `max_rel = ((x-y).abs() / (y.abs()+1e-6)).max()`
* Và `count_exceed` (số phần tử vượt một threshold)

**Pass logic:**

* Varlen vs Ref‑Aligned: yêu cầu `max_abs == 0` (hoặc cực nhỏ).
* Varlen vs Ref‑Local: yêu cầu `max_abs <= 2 * max_abs_schedule + margin`.

---

# 4) Kẽ hở số 4: No-op padding token dựa vào underflow expf — gần như chắc đúng, nhưng vẫn nên “test chính no-op”

Mình dùng `w_input=-10000` để ép `exp(-10000)→0`, nên `w=exp(-0)=1`. Đây **rất hợp lý** vì xa giới hạn float32.

Nhưng “kẽ hở” là: mình giả định không-op token tạo identity update **trong cả forward lẫn backward** mà không chứng minh bằng test.

## Sửa: thêm unit test “no-op thật sự là no-op”

Một test rất rẻ nhưng cực hữu ích:

* Tạo 1 sequence real length L.
* Chạy original kernel trên:

  * `T = L` (suffix pad no-op để T multiple 16)
  * `T = prefix_noop + L` (prefix no-op)
* So sánh:

  * output của phần real tokens phải **y hệt** (exact bf16)
  * gradients của phần real tokens phải **y hệt** (exact bf16)

Nếu fail thì có 2 khả năng:

* no-op token không thực sự identity (bug trong cách bạn set no-op)
* hoặc original kernel có hành vi phụ thuộc vào “số bước” (rất ít khả năng, nhưng test sẽ phát hiện)

---

# 5) Kẽ hở số 5: Test “no gradient leakage” có thể fail vì lý do **không phải leakage** (NaN/Inf)

Ý tưởng của bạn (loss chỉ trên sequence B, grads A/C phải 0) là đúng để bắt leak.

Nhưng có một tình huống làm test fail “giả”:

* Nếu `w` quá nhỏ → reconstruct chia cho `w` → stateT có thể thành `inf`
* Rồi trong dq tính `stateT * dy`; nếu `dy=0` mà `stateT=inf` ⇒ `inf * 0 = NaN`
* Kết quả: dq không phải 0, nhưng **không phải do leak** (nó là numeric pathology)

## Sửa:

* Trong leakage tests, **khóa distribution** của `w_input` để tránh `w` siêu nhỏ.

  * Ví dụ: `w_input ∈ [-6, 2]` như mình nói trước (w tối thiểu ~6e-4)
* Và thêm assertion “không NaN/Inf” trước khi check zeros:

  ```python
  assert torch.isfinite(dq).all()
  ```

Ngoài ra, để tránh “pass rỗng”:

* luôn check B grads **nonzero**:

  ```python
  assert dq[startB:endB].abs().sum() > 0
  ```

---

# 6) Kẽ hở số 6: Stress “num_seqs > 65535” dễ biến thành test cực chậm vì số block = num_seqs * H

Mình đề xuất `[0]*70000 + [16]`. Nhưng nếu H lớn (ví dụ 64), grid = 70k*64 = 4.48M blocks — launch overhead rất lớn, test có thể chậm vô lý.

## Sửa:

* Grid-limit test nên dùng **H=1** (HC=64):

  * vẫn test mapping seq-id đúng, vẫn vượt 65535
  * block count ~70k, hợp lý
* Và dùng mostly empty sequences để compute nhỏ.

---

# 7) Kẽ hở số 7: Mình thiếu một test quan trọng: **invariance theo vị trí pack (start offset)**

Vì bạn dùng **global checkpoints**, backward có thể phụ thuộc vào `start % 16` (về mặt numerical drift).
Ngay cả nếu bạn chấp nhận drift nhỏ, bạn vẫn muốn chắc rằng:

* forward **không bao giờ** phụ thuộc start offset (phải exact)
* backward **ít nhất** không “nhảy loạn” khi start offset thay đổi

## Sửa: thêm “offset invariance test”

Thiết kế:

* Chọn một sequence X cố định (inputs random).
* Tạo nhiều packed batches mà X xuất hiện ở các start offset khác nhau bằng cách chèn trước nó các sequences no-op với tổng length thay đổi.
* So sánh output của X:

  * forward: phải **exact match**
* So sánh gradients của X (loss chỉ trên X):

  * phải match Ref‑Local trong tolerance chặt
  * đồng thời theo dõi độ dao động theo offset; nếu dao động tăng mạnh ở offset nào đó, thường là bug replay/p0.

Test này bắt:

* sai công thức p0 trong tail replay
* bug dùng sai sa index trong replay
* bug chunk-id global

---

# 8) Kẽ hở số 8: “So sánh output/grads” có thể bỏ sót bug “không ghi đủ phần tử” nếu tensor ban đầu tình cờ là 0

Nếu output buffer được `torch.empty` thì có thể chứa rác; nếu kernel không ghi hết, diff sẽ lộ. Nhưng nếu bạn dùng `torch.zeros`, kernel thiếu ghi có thể vẫn “tình cờ pass”.

## Sửa: dùng NaN prefill để bắt “missing writes”

Trước khi gọi varlen:

* fill y/dw/dq/... bằng NaN (float32) rồi cast? (bf16 cũng có NaN)
* chạy kernel
* assert mọi phần tử “đáng lẽ phải ghi” đều finite:

  ```python
  assert torch.isfinite(y).all()
  ```

Cực hữu ích để bắt bug “return sớm”, “cu_seqlens loop sai”, “lỗi grid indexing”.

---

# 9) Kẽ hở số 9: Test suite hiện tại vẫn phụ thuộc 100% vào original kernel → nguy cơ “cả hai cùng sai theo cùng một cách”

Bạn có thể chấp nhận điều này vì mục tiêu là “match original”, nhưng nếu bạn muốn “definitively correct” hơn, cần một reference độc lập.

## Sửa (bảo thủ, không quá tốn): thêm CPU float64 reference cho small cases

* H=1, T≤8, số sequences nhỏ
* implement state matrix 64×64 trong torch float64 (CPU)
* forward + autograd để lấy grads float64
* so sánh với varlen/original (cast sang float32/bf16) với tolerance

Điều này giúp bắt lỗi kiểu:

* nhầm transpose state / nhầm dimension r/c
* nhầm công thức w = exp(-exp(x))
  mà nếu original kernel vô tình cùng sai, test A/B sẽ không phát hiện.

---

# 10) Kết luận đã “sửa” test strategy (phiên bản chặt hơn)

## Những thay đổi quan trọng so với kế hoạch trước

1. **Không dùng Ref‑Aligned để thay Ref‑Local.**
   Dùng **cả hai**:

   * Ref‑Aligned: bắt bug logic/indexing (kỳ vọng exact)
   * Ref‑Local: đảm bảo tương thích baseline (kỳ vọng close, bound theo Δ_schedule)

2. **Tolerance không “đoán”, mà calibrate bằng Δ_schedule = Ref‑Local vs Ref‑Aligned.**

3. **Thêm tests mới bắt lỗi mà trước đó bỏ sót:**

   * no-op token thật sự no-op (unit test)
   * offset invariance (pack vị trí khác nhau)
   * NaN prefill để bắt missing writes
   * grid-limit test với H=1
   * optional CPU float64 independent reference cho tiny cases

---

Nếu bạn muốn, mình có thể viết luôn một “test harness skeleton” (pytest) theo đúng cấu trúc:

* `run_ref_local(seq_inputs)`
* `run_ref_aligned(seq_inputs, start_offset)`
* `run_varlen(packed_inputs, cu_seqlens)`
* `compare_with_dual_reference(...)`
* `leakage_test(...)`
* `offset_invariance_test(...)`

…và chỉ ra rõ mỗi test nên dùng strict equality hay calibrated tolerance.

