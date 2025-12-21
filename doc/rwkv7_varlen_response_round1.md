Below is a “from first principles” treatment that matches what your kernel is *actually* doing (matrix state), then extends it to varlen packing with a boundary‑correct backward, including checkpoints.

I’m going to be very explicit about indices because that’s where varlen bugs happen.

---

# 0. What the kernel really computes (important for correctness)

Even though the “math formulation” you wrote shows `state[j]` as a vector, **the kernel’s per-thread `state[C]` implies the full per-head state is a `C×C` matrix**.

Let:

* `C = HEAD_SIZE = 64`
* For a fixed `(sequence, head)` and time `t`,

  * `S_t ∈ ℝ^{C×C}` is the state **matrix**
  * row index `r` corresponds to the **output/value channel** (thread index in forward)
  * column index `c` corresponds to the **w/k/q/a/b channel**

Then, in your forward kernel:

* each thread `i` owns **one row** `r=i` of `S_t` as a length‑`C` array `state[j] = S_t[r=i, c=j]`.

The shared vectors are:

* `w[c]`, `k[c]`, `q[c]`, `a[c]`, `b[c]` for `c=0..C-1`
* `v[r]` is *not* in shared in forward; each thread loads its own scalar `v_i`, so row `r` gets `v_t[r]`

Define the per-row “state-attention” scalar:

[
\mathrm{sa}*t[r] = \sum*{c=0}^{C-1} a_t[c];S_{t-1}[r,c].
]

Then the state update in code

```cpp
s = s*w[j] + sa*b[j] + k[j]*v;
```

is exactly:

[
S_t[r,c] = S_{t-1}[r,c]; w_t[c] + \mathrm{sa}_t[r]; b_t[c] + v_t[r]; k_t[c].
]

And the output per row `r`:

[
y_t[r] = \sum_{c=0}^{C-1} S_t[r,c]; q_t[c].
]

Finally, **checkpoint layout**: your forward stores `S_t` *transposed* so backward can load contiguous columns:

* forward writes `s_[row=c, col=r] = S_t[r,c]`
* backward thread `i` loads contiguous `s_[row=i, col=*]`, which gives the **column** `S_t[*, i]`.

That “transpose trick” is intentional and consistent with the backward math (as we’ll see).

---

# 1. Backward derivation for a single sequence (step-by-step)

Ignore batching and heads for the derivation; everything below is per `(seq, head)`.

## 1.1 Definitions

Let loss be (L). You are given:

* (\mathrm{d}y_t[r] = \frac{\partial L}{\partial y_t[r]})

Define the state gradient matrix:

[
G_t[r,c] \equiv \frac{\partial L}{\partial S_t[r,c]}.
]

We will derive:

* ( \mathrm{d}q_t, \mathrm{d}w_t, \mathrm{d}k_t, \mathrm{d}v_t, \mathrm{d}a_t, \mathrm{d}b_t)
* recurrence to propagate (G_t \to G_{t-1})
* and then show boundary behavior for varlen.

---

## 1.2 Gradients from the output (y_t = S_t q_t)

Elementwise:

[
y_t[r] = \sum_c S_t[r,c]; q_t[c].
]

### Gradient w.r.t. (q_t)

[
\frac{\partial y_t[r]}{\partial q_t[c]} = S_t[r,c]
\quad\Rightarrow\quad
\mathrm{d}q_t[c]
= \sum_r \mathrm{d}y_t[r]; S_t[r,c].
]

In vector/matrix form:

[
\mathrm{d}q_t = S_t^\top; \mathrm{d}y_t.
]

### Gradient contribution to (G_t)

[
\frac{\partial y_t[r]}{\partial S_t[r,c]} = q_t[c]
\quad\Rightarrow\quad
G_t[r,c] \mathrel{+}= \mathrm{d}y_t[r]; q_t[c].
]

So:

[
G_t \mathrel{+}= \mathrm{d}y_t; q_t^\top
\quad\text{(outer product)}.
]

**This is exactly what your code accumulates as:**

* `dstate[j] += dyi * q[j]` (row `r=i` of `G_t`)
* `dstateT[j] += qi * dy[j]` (column `c=i` of `G_t`)

---

## 1.3 Gradients through the state update

State update:

[
S_t[r,c] = S_{t-1}[r,c]; w_t[c]

* \mathrm{sa}_t[r]; b_t[c]
* v_t[r]; k_t[c]
  ]

where

[
\mathrm{sa}*t[r] = \sum_u a_t[u]; S*{t-1}[r,u].
]

We proceed term by term.

---

### 1.3.1 Gradients for (w_t)

Since (S_t[r,c]) depends on (w_t[c]) only through (S_{t-1}[r,c];w_t[c]):

[
\frac{\partial S_t[r,c]}{\partial w_t[c]} = S_{t-1}[r,c].
]

Thus:

[
\mathrm{d}w_t[c] = \sum_r G_t[r,c]; S_{t-1}[r,c].
]

This is a dot product of the **column gradient** (G_t[:,c]) with the **previous state column** (S_{t-1}[:,c]).

**Matches your code:**

* backward reconstructs (S_{t-1}[:,i]) into `stateT[j]`
* `dw += dstateT[j] * stateT[j]`
* then chain rule through (w = \exp(-\exp(x)))

Chain rule:

Let (x = w_input), (w = \exp(-\exp(x))).

[
\frac{dw}{dx} = w \cdot (-\exp(x)).
]

Your code:

* `wi_fac = -exp(x)`
* `wi = exp(wi_fac) = w`
* stores `dw * wi * wi_fac` ✅

---

### 1.3.2 Gradients for (k_t) and (v_t)

From the term (v_t[r];k_t[c]):

* For (k_t[c]):

[
\frac{\partial S_t[r,c]}{\partial k_t[c]} = v_t[r]
\Rightarrow
\mathrm{d}k_t[c] = \sum_r G_t[r,c]; v_t[r].
]

* For (v_t[r]):

[
\frac{\partial S_t[r,c]}{\partial v_t[r]} = k_t[c]
\Rightarrow
\mathrm{d}v_t[r] = \sum_c G_t[r,c]; k_t[c].
]

**Matches your code:**

* `dk += dstateT[j] * v[j]` computes (\sum_r G_t[r,i] v[r])
* `dv += dstate[j] * k[j]` computes (\sum_c G_t[i,c] k[c])

---

### 1.3.3 Gradients for (b_t) and for (\mathrm{sa}_t)

From the term (\mathrm{sa}_t[r]; b_t[c]):

* For (b_t[c]):

[
\frac{\partial S_t[r,c]}{\partial b_t[c]} = \mathrm{sa}_t[r]
\Rightarrow
\mathrm{d}b_t[c] = \sum_r G_t[r,c]; \mathrm{sa}_t[r].
]

* For (\mathrm{sa}_t[r]):

[
\frac{\partial S_t[r,c]}{\partial \mathrm{sa}_t[r]} = b_t[c]
\Rightarrow
\mathrm{d}\mathrm{sa}_t[r] = \sum_c G_t[r,c]; b_t[c].
]

**Matches your code:**

* `db += dstateT[j] * sa[j]` computes (\mathrm{d}b_t[i])
* `dSb += dstate[j] * b[j]` computes (\mathrm{d}\mathrm{sa}_t[i]) (per row)
* stores `dSb_shared[i] = dsa[i]`

---

### 1.3.4 Gradients for (a_t) via (\mathrm{sa}*t = S*{t-1} a_t)

[
\mathrm{sa}*t[r] = \sum_u S*{t-1}[r,u]; a_t[u].
]

* For (a_t[u]):

[
\frac{\partial \mathrm{sa}*t[r]}{\partial a_t[u]} = S*{t-1}[r,u]
\Rightarrow
\mathrm{d}a_t[u] = \sum_r \mathrm{d}\mathrm{sa}*t[r]; S*{t-1}[r,u].
]

That is exactly:

[
\mathrm{d}a_t = S_{t-1}^\top; \mathrm{d}\mathrm{sa}_t.
]

**Matches your code:**

* thread `i` corresponds to column (u=i)
* it holds `stateT[j] = S_{t-1}[j, i]` after reconstruction
* it sees `dSb_shared[j] = dsa[j]`
* computes `da = Σ_j stateT[j] * dSb_shared[j]` ✅

---

### 1.3.5 Propagate state gradients (G_t \to G_{t-1})

This is the critical recurrence that determines whether gradients “leak”.

Two dependency paths from (S_{t-1}) into (S_t):

1. direct: (S_t[r,c] \supset S_{t-1}[r,c];w_t[c])

So:

[
\frac{\partial S_t[r,c]}{\partial S_{t-1}[r,c]} = w_t[c]
\Rightarrow
G_{t-1}[r,c] \mathrel{+}= G_t[r,c]; w_t[c].
]

2. indirect via (\mathrm{sa}_t[r]):

Since (\mathrm{sa}*t[r] = \sum_u a_t[u];S*{t-1}[r,u]),

[
\frac{\partial \mathrm{sa}*t[r]}{\partial S*{t-1}[r,c]} = a_t[c].
]

And (S_t[r,c] \supset \mathrm{sa}_t[r];b_t[c]), so the total gradient to (\mathrm{sa}_t[r]) is (\mathrm{d}\mathrm{sa}_t[r]).

Thus:

[
G_{t-1}[r,c] \mathrel{+}= \mathrm{d}\mathrm{sa}_t[r]; a_t[c].
]

Combine:

[
\boxed{
G_{t-1}[r,c] = G_t[r,c]; w_t[c] + \mathrm{d}\mathrm{sa}_t[r]; a_t[c].
}
]

**Matches your code updates:**

* row view (thread = row (r=i)): `dstate[c] = dstate[c]*w[c] + dsa[r]*a[c]`
* column view (thread = col (c=i)): `dstateT[r] = dstateT[r]*w[i] + a[i]*dsa[r]`

That second line is just selecting the ((r,c=i)) elements of the boxed recurrence.

---

# 2. What varlen boundaries mean mathematically (and what backward must do)

## 2.1 Formalizing packed sequences

Let there be (S) sequences with lengths (L_0, \dots, L_{S-1}).

`cu_seqlens` is:

[
\mathrm{cu}[0]=0,\quad
\mathrm{cu}[s+1]=\sum_{u=0}^{s} L_u.
]

Total tokens:

[
T_{\text{tot}} = \mathrm{cu}[S].
]

Global token index (p \in [0, T_{\text{tot}})) maps to sequence (s) and local time (t) such that:

[
\mathrm{cu}[s] \le p < \mathrm{cu}[s+1],
\quad
t = p - \mathrm{cu}[s].
]

### Varlen forward rule

For each sequence (s):

[
S^{(s)}*{-1} = 0
]
[
S^{(s)}*{t} = f(S^{(s)}*{t-1}, x^{(s)}*{t})
\quad t=0..L_s-1
]

There is **no** dependence between different (s).

Packing just stores tensors in global order (p), but the computation graph is a disjoint union of per-sequence graphs.

---

## 2.2 Rigorous “no gradient leak” proof

Take two different sequences (s \ne s'). Consider any token parameters (x^{(s)}*t) and any output (y^{(s')}*{t'}).

Because the recurrence for (S^{(s')}) only uses inputs from sequence (s') and initial state (0), the function (y^{(s')}_{t'}) is **independent** of (x^{(s)}_t):

[
\frac{\partial y^{(s')}_{t'}}{\partial x^{(s)}_t} = 0.
]

Thus, by chain rule for any loss (L) depending on all outputs:

[
\frac{\partial L}{\partial x^{(s)}*t}
= \sum*{s',t'} \frac{\partial L}{\partial y^{(s')}*{t'}};
\frac{\partial y^{(s')}*{t'}}{\partial x^{(s)}*t}
= \sum*{t'} \frac{\partial L}{\partial y^{(s)}*{t'}};
\frac{\partial y^{(s)}*{t'}}{\partial x^{(s)}_t}.
]

All terms with (s' \ne s) vanish.

**Therefore gradients cannot cross sequence boundaries.**
Any implementation that allows `dstate` (or any state-gradient accumulator) to carry information from sequence (s) into sequence (s-1) is wrong.

---

## 2.3 Boundary conditions for backward (what to reset)

For a *single* sequence (s) of length (L):

* There is no token after the last one. So the state after the last token has no influence on the loss.

That is:

[
G^{(s)}_{L} = 0 \quad \text{(terminal condition)}.
]

* The initial state is constant (S^{(s)}*{-1} = 0). Even if backward computes (G^{(s)}*{-1}), it must **not** propagate further.

So in code terms:

* At the **end of each sequence**, initialize:

  * `dstate[:] = 0`  (row gradients)
  * `dstateT[:] = 0` (column gradients)

This enforces (G_L=0).

* When you cross from sequence (s) to (s-1) in a global backward scan, you must prevent the just-computed gradient to the initial state of (s) from being used as the “future gradient” for (s-1). The cleanest way is: **reset at sequence end when you start processing each sequence**.

---

# 3. Answering your three “CRITICAL” boundary questions

I’ll use your example boundary “t=5 going to t=4” (crossing from C to B).

Assume global packed indices and your example:

* A: p=0,1,2
* B: p=3,4
* C: p=5,6,7,8

### Q1) At boundary (e.g., t=5 going to t=4), `stateT[t=4]` should be what value?

It must be the **true forward state at token 4 in sequence B**, i.e.

[
S^{(B)}_{t=1} \quad \text{(because B has length 2, token p=4 is local t=1)}.
]

In other words:

* `stateT` used when computing gradients at global token `p=4` must equal (S_{p=4}) (post-update state for that token in B), **not** 0 and not anything derived from C.

Why it cannot be reconstructed from token 5:

* token `p=5` is the first token of C and has previous state (0).
  Reconstructing “previous” from C0 gives (S^{(C)}*{-1} = 0), which is **not** (S^{(B)}*{1}).

So you must **load** the state for token 4 from a checkpoint that belongs to B (most naturally, a “sequence end checkpoint”).

---

### Q2) At boundary, `dstate` and `dstateT` should be reset to what?

They must be reset to **all zeros** at the start of the backward processing for a sequence end:

[
dstate[:] = 0,\quad dstateT[:] = 0.
]

Reason (step-by-step):

* The only way gradients from later tokens influence earlier tokens is through the recurrence (G_{t-1} = G_t \cdot (\cdots)).
* At a sequence end, there is no later token in that sequence, so (G_{\text{after end}} = 0).
* Therefore the “incoming” gradient to the last token must start at zero. If you keep non-zero accumulators from a different sequence, you violate the block-diagonal (independent) structure proven above.

---

### Q3) How does the checkpoint at `t=chunk_end` interact with sequence boundaries?

Key fact: a checkpoint is only a way to get an accurate value of (S_t) at some time (t). It is **mathematically valid** to load a checkpoint at time (t) if (and only if) it corresponds to the state of the *same sequence* at that token.

So boundary interaction reduces to: **don’t use a checkpoint from sequence C to initialize reconstruction for sequence B.**

Two robust ways:

1. **Per-sequence processing (recommended):** each CUDA block handles one sequence; then “chunk checkpoints crossing boundaries” is impossible.

2. **Global processing:** if you do a single backward scan over all packed tokens, then:

   * you may load a chunk checkpoint at a global chunk end that lies inside sequence C,
   * but when you reach `p=4` (sequence B end), you must override with B’s end checkpoint and reset gradients.
     Any reconstruction state that “crosses” the boundary is discarded as soon as you hit the next sequence end.

---

# 4. Concrete example walkthrough: lengths [3,2,4], cu_seqlens=[0,3,5,9]

I’ll do this symbolically (since you didn’t provide numeric tensors), but it’s still “exact” in terms of the inputs.

## 4.1 Forward: states around boundaries

Let token parameters at global index (p) be (w_p,q_p,k_p,v_p,a_p,b_p).

Define (M_p = D(w_p) + a_p b_p^\top) (as derived earlier).

Sequence A (p=0,1,2), with (S^{A}_{-1}=0):

* p=0 (A0):

  * (\mathrm{sa}*0 = S*{-1} a_0 = 0)
  * (S_0 = 0\cdot M_0 + v_0 k_0^\top = v_0 k_0^\top)
* p=1 (A1):

  * (\mathrm{sa}_1 = S_0 a_1)
  * (S_1 = S_0 M_1 + v_1 k_1^\top)
* p=2 (A2):

  * (\mathrm{sa}_2 = S_1 a_2)
  * (S_2 = S_1 M_2 + v_2 k_2^\top)

Sequence B (p=3,4), reset (S^{B}_{-1}=0) at p=3:

* p=3 (B0):

  * (\mathrm{sa}_3 = 0)
  * (S_3 = v_3 k_3^\top)
* p=4 (B1):

  * (\mathrm{sa}_4 = S_3 a_4)
  * (S_4 = S_3 M_4 + v_4 k_4^\top)

Sequence C (p=5,6,7,8), reset at p=5:

* p=5 (C0):

  * (\mathrm{sa}_5 = 0)
  * (S_5 = v_5 k_5^\top)
* p=6:

  * (S_6 = S_5 M_6 + v_6 k_6^\top)
* p=7:

  * (S_7 = S_6 M_7 + v_7 k_7^\top)
* p=8:

  * (S_8 = S_7 M_8 + v_8 k_8^\top)

**Boundary observation:** there is no formula connecting (S_5) to (S_4); they live in different sequences and have separate initial conditions.

---

## 4.2 Backward: what must happen at boundaries

For each sequence, backward is the same recurrence derived in §1, with terminal condition (G_{\text{after end}} = 0).

### Start with sequence C (end at p=8)

Initialize at sequence C end:

* “future state gradient” is zero:
  [
  G_{9} = 0
  ]
* At p=8:
  [
  G_8 = \underbrace{0}_{\text{future}} + \mathrm{d}y_8 q_8^\top
  ]
* Then propagate:
  [
  G_7 = G_8 D(w_8) + \mathrm{d}\mathrm{sa}_8 a_8^\top
  ]
  etc, down to p=5.

At p=5 (first token of C), after computing gradients, you might compute (G_{4}^{(\text{as if})}) via recurrence, but that object corresponds to gradient wrt C’s initial state (S^{C}_{-1}), not B’s last state. It must not be used for p=4.

### Now crossing boundary to sequence B end (p=4)

Correct boundary condition for sequence B:

* There is **no** future token in B after p=4 (because p=5 is start of C and independent).
* Therefore for B, we must restart with:
  [
  G_{5}^{(B)} = 0
  ]
  i.e. the “incoming future gradient” at B’s end is zero.

So at p=4:
[
G_4 = 0 + \mathrm{d}y_4 q_4^\top
]
with **no contribution from C**.

This is exactly the operational meaning of:

* reset `dstate`/`dstateT` to 0 when you start a new sequence in backward (i.e., at that sequence’s end token),
* and load `stateT` from B’s end checkpoint (not from C reconstruction).

Then you process p=3 similarly, and then jump to A end p=2 with another reset, etc.

That’s the rigorous “no leak” behavior.

---

# 5. Checkpoint strategy: options, tradeoffs, and a recommendation

Your current fixed-length backward relies on: **“the last token is always a chunk boundary”** (`assert(T%CHUNK_LEN==0)`) so you can load `S_{T-1}` from `s_` and reconstruct backwards.

Varlen breaks this: sequence ends are arbitrary.

## Option A: Global checkpoints every CHUNK_LEN tokens (in packed space)

**Idea:** checkpoint at global packed positions (p) where ((p+1)\bmod 16=0).

**Problem:** sequence ends are usually not at those positions, and backward needs (S_{\text{end}}) for each sequence to compute:

* (dq) at the last token (needs (S_{end}))
* and to start reconstruction

So **A alone is insufficient** unless you also ensure every sequence end is checkpointed (by padding or extra storage).

### Pros

* Fixed predictable checkpoint count: (\lceil T_{\text{tot}}/16\rceil)
* Memory scales with total tokens, not max length

### Cons

* Still need something for per-sequence ends (or padding)

---

## Option B: Per-sequence checkpoints (aligned to each sequence start)

Checkpoint within each sequence at local steps where ((t+1)\bmod 16=0).

### Pros

* Conceptually clean
* No chunk cuts across boundaries
* Reconstruction distance ≤ 16 within each sequence

### Cons

* Memory/indexing becomes variable
* If you allocate as `(num_seqs, max_chunks, ...)`, memory scales with `num_seqs * max_seqlen`, which can destroy the packing memory benefit unless you *also* pack checkpoints with another prefix sum.

---

## Option C: No checkpoints (recompute forward inside backward)

### Pros

* Simplest correctness-wise for varlen
* No checkpoint/boundary headaches

### Cons

* ~2× compute (or worse depending on how much you recompute)
* defeats much of the point of this kernel design

---

## Option D (recommended): Hybrid = global chunk checkpoints + per-sequence end checkpoints

This is the minimum change that preserves your current reconstruction method and fixes varlen correctness.

### What to store

1. **Global “chunk” checkpoints** at packed indices where ((p+1)\bmod 16=0)
   Store (S_p) (transposed layout like today).

2. **Per-sequence end checkpoints** at each sequence’s last token
   For sequence (s), end token is (p = \mathrm{cu}[s+1]-1). Store (S_p) there as well.

### Why this works

* Within a sequence, between any two global chunk checkpoints, distance ≤ 16.
* For sequences shorter than 16, you still have the end checkpoint.
* Backward can always start each sequence by loading its end checkpoint.
* Boundary correctness is handled by resetting gradients at each sequence end (see §2–§4).

### Memory

Let:

* total tokens (T_{\text{tot}})
* sequences (S)
* heads (H)
* checkpoint matrix size (C^2) float32

Checkpoint memory:

* global chunk checkpoints: (\lceil T_{\text{tot}}/16\rceil \cdot H \cdot C^2)
* end checkpoints: (S \cdot H \cdot C^2)

That’s **O(total_tokens/16 + num_seqs)**, which is what you want for packing.

### Implementation simplicity

* Minimal modifications to forward/backward loops
* No need to pack variable number of chunk checkpoints per sequence
* No need for per-token binary search if you process per sequence (recommended)

---

# 6. Edge cases: explicit handling

I’ll state exactly what the forward and backward should do.

## 6.1 Sequence length < CHUNK_LEN (e.g., 1..15)

Forward:

* no global chunk checkpoint inside the sequence
* must still store **end checkpoint** at the last token

Backward:

* load end checkpoint at the last token and reconstruct backward for the whole sequence
* resets ensure no leak to other sequences

✅ Works with Option D.

---

## 6.2 Sequence boundary exactly at chunk boundary

That means:

* end token (p) satisfies ((p+1)\bmod 16=0)

Forward:

* global chunk checkpoint already captures that end state
* end checkpoint is redundant but harmless

Backward:

* if you code “if last token: load end checkpoint”, you’ll load the redundant one
* gradients remain correct

✅ Works.

---

## 6.3 Empty sequence (length = 0)

cu_seqlens has repeated entries: `cu[s] == cu[s+1]`.

Forward:

* skip entirely (no tokens)
* do not write end checkpoint (there is no end token)

Backward:

* skip entirely

Important: your launch grid should still include this sequence id if you keep `num_seqs` blocks; inside the kernel:

```cpp
if (start == end) return;
```

✅ Works.

---

## 6.4 Single token sequence (length = 1)

Forward:

* initial state is 0
* `sa = 0`
* state after token is (S = v k^\top)
* store end checkpoint at that token

Backward:

* load end checkpoint
* compute gradients for that token
* reconstruct to (S_{-1}=0) and exit

✅ Works.

---

## 6.5 Very long sequence (≫ CHUNK_LEN)

Forward:

* stores global chunk checkpoints every 16 packed tokens inside it
* store end checkpoint

Backward:

* loads end checkpoint to start
* periodically refreshes state from global chunk checkpoints (reduces reconstruction drift)
* normal recurrence

✅ Works.

---

# 7. Modified backward kernel pseudo-code with boundary-correct varlen handling

I’m going to give the version that avoids the “scan across all sequences in one loop” entirely, because that design is both slower (no seq parallelism) and harder to reason about.

## 7.1 Recommended grid mapping

Launch:

* `grid.x = H` (heads)
* `grid.y = num_seqs` (one block per sequence per head)
* `block.x = C` (64 threads)

This preserves your current parallelism structure.

## 7.2 Required tensors for varlen mode

Packed tensors (bf16), shape: `(total_tokens, H, C)`:

* `w, q, k, v, a, b`  (rename as you wish)

Saved for backward:

* `sa` float32, shape `(total_tokens, H, C)`

Checkpoints:

* `s_chunk` float32, shape `(H, num_chunks, C, C)`
  where `num_chunks = ceil(total_tokens / CHUNK_LEN)`
* `s_end` float32, shape `(num_seqs, H, C, C)`
  storing the end state for each sequence

Both `s_chunk` and `s_end` store the **transposed** matrix layout like your existing code.

## 7.3 Backward pseudo-code

```cuda
__global__ void backward_kernel_varlen(
    int total_tokens,
    int H,
    const int* __restrict__ cu_seqlens, // (num_seqs+1)
    int num_seqs,

    // packed inputs: (total_tokens, H, C)
    bf* __restrict__ w_,
    bf* __restrict__ q_,
    bf* __restrict__ k_,
    bf* __restrict__ v_,
    bf* __restrict__ a_,
    bf* __restrict__ b_,
    bf* __restrict__ dy_,

    // saved intermediates
    float* __restrict__ sa_,            // (total_tokens, H, C)

    // checkpoints
    float* __restrict__ s_chunk_,       // (H, num_chunks, C, C) transposed
    float* __restrict__ s_end_,         // (num_seqs, H, C, C) transposed

    // outputs
    bf* __restrict__ dw_,
    bf* __restrict__ dq_,
    bf* __restrict__ dk_,
    bf* __restrict__ dv_,
    bf* __restrict__ da_,
    bf* __restrict__ db_
){
    constexpr int C = 64;
    int hh  = blockIdx.x;      // head
    int seq = blockIdx.y;      // sequence id
    int i   = threadIdx.x;     // 0..63 (dual role: row and col index)

    int start = cu_seqlens[seq];
    int end   = cu_seqlens[seq + 1];
    int L     = end - start;
    if (L <= 0) return;

    // Per-thread persistent vectors across time:
    // stateT[j] will represent S_{t}[row=j, col=i] for the CURRENT time t
    // (i.e. column i of S_t).
    float stateT[C]  = {0};

    // dstate[j]  = G_t[row=i, col=j] (row i of G_t)
    // dstateT[j] = G_t[row=j, col=i] (col i of G_t)
    float dstate[C]  = {0};
    float dstateT[C] = {0};

    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C];
    __shared__ float dy[C], sa[C];
    __shared__ float dsa_shared[C]; // dsa per row (your dSb_shared)

    for (int tl = L - 1; tl >= 0; --tl) {
        int p = start + tl; // global packed token index

        // ----- load token vectors into shared -----
        int ind = (p * H + hh) * C + i;

        __syncthreads();
        q[i]  = to_float(q_[ind]);
        k[i]  = to_float(k_[ind]);
        a[i]  = to_float(a_[ind]);
        b[i]  = to_float(b_[ind]);
        v[i]  = to_float(v_[ind]);
        dy[i] = to_float(dy_[ind]);
        sa[i] = sa_[ind]; // saved sa for this (p,hh,row=i)
        // w is stored as input x, but backward needs both w and exp(x):
        float x = to_float(w_[ind]);
        float wi_fac = -__expf(x);
        w[i] = __expf(wi_fac); // w in (0,1)
        __syncthreads();

        float wi = w[i];       // scalar w for column i
        float ki = k[i];
        float bi = b[i];
        float ai = a[i];
        float qi = q[i];
        float dyi = dy[i];

        // ----- checkpoint load (critical for correctness and stability) -----
        if (tl == L - 1) {
            // Start of backward for this sequence:
            // Must load S_{end} and set incoming state-gradients to 0.
            int base_end = ((seq * H + hh) * C * C) + i * C;
            #pragma unroll
            for (int j = 0; j < C; ++j) stateT[j] = s_end_[base_end + j];

            // Terminal condition: no future tokens in this sequence.
            #pragma unroll
            for (int j = 0; j < C; ++j) { dstate[j] = 0.0f; dstateT[j] = 0.0f; }
        }
        else if ( ((p + 1) & (CHUNK_LEN - 1)) == 0 ) {
            // Optional refresh from global chunk checkpoint
            int chunk = p / CHUNK_LEN;
            int base_chunk = ((hh * num_chunks + chunk) * C * C) + i * C;
            #pragma unroll
            for (int j = 0; j < C; ++j) stateT[j] = s_chunk_[base_chunk + j];
        }

        // ----- dq for q_i uses column i of S_t -----
        // dq_i = sum_r S_t[r,i] * dy[r]
        float dq_i = 0.0f;
        #pragma unroll
        for (int r = 0; r < C; ++r) dq_i += stateT[r] * dy[r];
        dq_[ind] = to_bf(dq_i);

        // ----- reconstruct S_{t-1}[:,i] from S_t[:,i] -----
        // S_t[r,i] = S_{t-1}[r,i]*w_i + sa[r]*b_i + v[r]*k_i
        // => S_{t-1}[r,i] = (S_t[r,i] - sa[r]*b_i - v[r]*k_i)/w_i
        float inv_wi = 1.0f / wi;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            stateT[r] = (stateT[r] - ki * v[r] - bi * sa[r]) * inv_wi;
        }

        // ----- add output gradient contribution to G_t -----
        // G_t[r,c] += dy[r] * q[c]
        // We maintain row i and column i slices:
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            dstate[j]  += dyi * q[j];   // row i: G_t[i,j]
            dstateT[j] += qi  * dy[j];  // col i: G_t[j,i]
        }

        // ----- compute grads for w_i,k_i,b_i using column i of G_t and S_{t-1}[:,i] -----
        float dw_i = 0.0f, dk_i = 0.0f, db_i = 0.0f;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            dw_i += dstateT[r] * stateT[r]; // sum_r G_t[r,i] * S_{t-1}[r,i]
            dk_i += dstateT[r] * v[r];      // sum_r G_t[r,i] * v[r]
            db_i += dstateT[r] * sa[r];     // sum_r G_t[r,i] * sa[r]
        }

        // Chain rule through w = exp(-exp(x))
        // dw_input = dw * w * (-exp(x))  (wi_fac already = -exp(x))
        dw_[ind] = to_bf(dw_i * wi * wi_fac);
        dk_[ind] = to_bf(dk_i);
        db_[ind] = to_bf(db_i);

        // ----- compute dv_i and dsa_i from row i of G_t -----
        float dv_i = 0.0f;
        float dsa_i = 0.0f; // gradient wrt sa[row=i]
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            dv_i  += dstate[c] * k[c]; // dv[row=i] = sum_c G_t[i,c] * k[c]
            dsa_i += dstate[c] * b[c]; // dsa[row=i] = sum_c G_t[i,c] * b[c]
        }
        dv_[ind] = to_bf(dv_i);

        // Share dsa across threads so each column thread can compute da_i
        __syncthreads();
        dsa_shared[i] = dsa_i;
        __syncthreads();

        // ----- compute da_i using S_{t-1} and dsa -----
        // da[i] = sum_r S_{t-1}[r,i] * dsa[r]
        float da_i = 0.0f;
        #pragma unroll
        for (int r = 0; r < C; ++r) da_i += stateT[r] * dsa_shared[r];
        da_[ind] = to_bf(da_i);

        // ----- propagate G_{t-1} slices -----
        // G_{t-1}[r,c] = G_t[r,c]*w[c] + dsa[r]*a[c]
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            dstate[c] = dstate[c] * w[c] + dsa_i * a[c]; // row i
        }
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            dstateT[r] = dstateT[r] * wi + ai * dsa_shared[r]; // col i
        }
    }
}
```

### Why this pseudo-code is boundary-correct

* It **never crosses sequences** in the time loop: `tl` runs only within `[start,end)`.
* It explicitly enforces the terminal condition (G_L=0) by zeroing `dstate`/`dstateT` at `tl==L-1`.
* It initializes the state for backward at the end of the sequence by loading `s_end`.
* Therefore gradients cannot leak across sequences.

This is the most conservative and defensible approach.

---

# 8. Performance implications and mitigations

## 8.1 Branch divergence

* The only branches inside the loop are:

  * `if (tl==L-1)` (same for all threads)
  * `else if ((p+1)&15==0)` (same for all threads)

So **no warp divergence** (all 64 threads in block take the same path).

Avoid `%` and use bitmask since `CHUNK_LEN=16`:

* `((p+1) & 15) == 0` is cheaper than `(p+1)%16==0`.

## 8.2 Memory access pattern

Packed layout `(total_tokens, H, C)`:

* For a fixed `(p, hh)`, threads `i=0..63` read contiguous `C` bf16 values.
* That is perfectly coalesced.

`sa` is float32 `(total_tokens, H, C)`:

* also coalesced.

Checkpoint loads/stores:

* `s_chunk`/`s_end` are float32 and are read/written in contiguous segments of length `C` per thread (your transposed layout), which is reasonably cache-friendly.

## 8.3 Shared memory usage

Same as current (a few `C`-length float arrays). No change.

## 8.4 Parallelism

Using `grid.y = num_seqs` preserves parallelism across sequences, unlike a “single pass over all tokens” design which would only have `H` blocks.

That’s a large performance win.

---

# 9. Alternative approaches (and why I wouldn’t pick them first)

## Padding + mask (do nothing in CUDA)

* Easy, but wastes compute on padding (the thing you want to avoid).
* Doesn’t reduce checkpoint/sa memory either.

## Process sequences separately (no packing)

* Simple correctness.
* Loses some launch amortization and data locality; but since you already have one block per `(seq, head)`, this is basically what the recommended kernel does anyway—just reading from a packed tensor.

## Recompute-forward-in-backward (Option C)

* Correct and simple.
* Likely too expensive for training, since your kernel already does heavy per-token work.

---

# 10. Summary of the “correct varlen recipe”

**Mathematical requirements (proved above):**

1. Each sequence must start with (S_{-1}=0).
2. Backward for each sequence must start with terminal condition (G_{L}=0).
3. No state or gradient accumulator may carry across boundaries.

**Engineering requirements (because your backward reconstructs state):**
4. You must have a way to obtain (S_{\text{end}}) for each sequence in backward:

* either store it (recommended) or recompute it (expensive) or pad so it lands on a stored boundary (wastes compute).

**Recommended implementation:**

* Use one block per `(seq, head)` and loop only within that sequence.
* Store:

  * global chunk checkpoints every 16 packed tokens (optional but good for stability)
  * **per-sequence end checkpoint** (required unless you pad/recompute)
* In backward:

  * load end checkpoint at `tl=L-1`
  * zero `dstate` / `dstateT` at `tl=L-1`
  * never process tokens outside the sequence in that block

This gives you FlashAttn-style varlen packing without gradient leakage, while staying very close to your existing kernel structure.

---

If you want, I can also sketch the **forward varlen kernel** and the **Python packing/unpacking API** (including how to allocate `s_chunk`/`s_end`/`sa` tensors and keep it backward-compatible when `cu_seqlens is None`) in the same style—still self-contained and consistent with your current op.

---

TỰ PHẢN BIỆN
------------

Dưới đây là “tự phản biện” (self‑rebuttal) cho kết luận trước, với mục tiêu **đào ra kẽ hở** và **đề xuất sửa** một cách bảo thủ nhưng hiệu quả. Mình sẽ nói thẳng: phần **đúng toán** mình vẫn giữ, nhưng phần **chiến lược checkpoint (đặc biệt `s_end`)** trong kết luận trước có một kẽ hở rất lớn về **bộ nhớ** và có thể sửa đẹp hơn.

---

# 1) Kẽ hở lớn nhất: `s_end` có thể “đốt” VRAM khi nhiều sequence ngắn

Trong kết luận trước mình “recommend” Option D: **global chunk checkpoints + per‑sequence end checkpoint `s_end`**.

Về đúng/sai toán: đúng.
Nhưng về **hệ quả hệ thống**: có thể rất tệ khi bạn pack **nhiều sequence ngắn** (đúng use-case varlen).

## 1.1 Vì sao `s_end` có thể cực đắt?

Mỗi end checkpoint phải lưu **state C×C float32 cho mỗi head**.

* `C = 64` ⇒ `C*C = 4096` float32
* 1 float32 = 4 bytes ⇒ **16 KB / head / sequence**
* nếu `H = 64` heads ⇒ **~1 MB / sequence** chỉ cho `s_end`

Nếu bạn pack kiểu “nhiều sequence rất ngắn” (VD hàng nghìn seq), `s_end` có thể thành hàng GB, dù tổng token không nhiều.

So sánh nhanh:

* Global chunk checkpoint memory ~ (\lceil T_{tot}/16\rceil \cdot H \cdot C^2)
* End checkpoint memory ~ (S \cdot H \cdot C^2)

Tỉ lệ:

[
\frac{\text{s_end}}{\text{s_chunk}} \approx \frac{S}{T_{tot}/16} = \frac{16}{L_{avg}}
]

* Nếu (L_{avg}=256): overhead ~ 6.25% (ổn)
* Nếu (L_{avg}=8): overhead ~ 200% (gấp **3 lần** tổng checkpoint!) → **ngược mục tiêu varlen**

➡️ **Kẽ hở**: mình đã “recommend” một phương án mà trong đúng tình huống varlen hay gặp (nhiều seq ngắn) có thể phá vỡ lợi ích.

---

# 2) Kẽ hở thứ hai: mình bỏ lỡ một trick quan trọng — bạn KHÔNG cần `s_end` nếu đã lưu `sa`

Điểm then chốt: trong kernel gốc, bạn đã lưu `sa_t[r]` float32 cho mọi token và mọi row.

Nhờ `sa` đã lưu, bạn có thể **tính lại state ở cuối sequence** từ checkpoint chunk gần nhất với **tối đa 15 bước forward**, mà **không cần** lưu `s_end`.

Đây là sửa chữa quan trọng nhất.

## 2.1 Chứng minh “không cần `s_end`”

Ta cần state cuối sequence: (S_{p_{end}}) (trạng thái *sau* token cuối).

Giả sử ta có checkpoint ở một vị trí (p_0) (thường là “chunk end” toàn cục) sao cho:

* (p_0 \le p_{end})
* và (S_{p_0}) đã được lưu trong `s_chunk`

Ta có công thức forward update (đúng như kernel):

[
S_p[r,c] = S_{p-1}[r,c]; w_p[c] + sa_p[r]; b_p[c] + v_p[r]; k_p[c]
]

**Quan trọng:** công thức này chỉ cần:

* (S_{p-1}) (state trước đó)
* (w_p, k_p, b_p, v_p)
* **và (sa_p)**

Nó **không cần** tự tính lại (sa_p) bằng dot(a, state) vì `sa_p` đã lưu sẵn từ forward.

➡️ Vậy: nếu có (S_{p_0}), ta có thể chạy forward từ (p_0+1) đến (p_{end}) để lấy (S_{p_{end}}).

## 2.2 Tại sao chỉ tối đa 15 bước?

Vì chunk end toàn cục ở các vị trí (p = 15,31,47,...) (điều kiện ((p+1)\bmod 16=0)).

Với mọi (p_{end}), chunk end gần nhất không vượt quá (p_{end}) là:

[
p_0 = \bigl(\lfloor (p_{end}+1)/16 \rfloor \cdot 16\bigr) - 1
]

Khoảng cách (p_{end}-p_0) chính là ((p_{end}+1)\bmod 16), luôn nằm trong ([0,15]).

* Nếu (p_{end}) đúng chunk end ⇒ 0 bước.
* Nếu không ⇒ tối đa 15 bước.

Nếu sequence quá ngắn không chứa chunk end nào, ta đặt (p_0 = start-1) và (S_{p_0}=0). Khi đó số bước forward = (L), mà (L<16) với seq ngắn.

➡️ **Sửa chữa**: bỏ `s_end`, thay bằng **tail forward replay** tối đa 15 bước nhờ `sa`.

---

# 3) Kẽ hở thứ ba: mình đánh giá “Option D là tối thiểu thay đổi”, nhưng thực ra “bỏ `s_end`” còn tối thiểu hơn

* Không cần thêm tensor `s_end`
* Không cần forward lưu `s_end`
* Chỉ cần thêm một đoạn **prelude** trong backward để dựng (S_{end})

Đây vừa **giảm VRAM**, vừa **giảm code phức tạp**.

---

# 4) Kẽ hở thứ tư: grid.y giới hạn 65535 (có thể vấp nếu num_seqs rất lớn)

Mình đề xuất `grid.y = num_seqs`. Trên CUDA, `gridDim.y` thường bị giới hạn 65535.

Nếu bạn pack kiểu “tons of short sequences”, có thể vượt giới hạn. Cách sửa:

* Dùng grid 1D: `blockIdx.x` chạy từ `0 .. num_seqs*H-1`

  * `hh = idx % H`
  * `seq = idx / H`

Như vậy không bị trần grid.y.

---

# 5) Kẽ hở thứ năm: phần walkthrough ví dụ trước còn “symbolic” quá, chưa khoá chặt logic “không leak” quanh boundary theo đúng kernel

Toán “không leak” vẫn đúng, nhưng mình có thể làm rõ hơn bằng cách mô tả **cơ chế reset** theo đúng flow kernel:

* Nếu bạn chạy **1 block / (seq, head)** thì “không leak” là *hiển nhiên về mặt code* (block không đọc token ngoài range).
* Nếu bạn chạy “global scan all tokens” (như pseudo code bạn đưa), bạn bắt buộc phải:

  * phát hiện boundary
  * reset `state`, `dstate`, `dstateT`
  * và “nhảy” đến checkpoint đúng của sequence trước

Kết luận trước của mình nghiêng về per‑seq block, nhưng mình chưa chỉ ra rõ rằng **per‑seq block là cách làm triệt leak ở mức kiến trúc kernel** (ít rủi ro hơn global scan).

---

# 6) Sửa chữa đề xuất: Checkpoint Strategy mới (mạnh hơn, ít VRAM hơn)

## 6.1 Strategy mới (đề xuất ưu tiên)

**Chỉ dùng global `s_chunk`** (chunk end toàn cục) + `sa_` như hiện tại.
**Không dùng `s_end`.**

Backward mỗi sequence sẽ:

1. **Tính state cuối sequence** bằng “tail forward replay” từ checkpoint chunk gần nhất:

   * start từ `S_{p0}` (load từ `s_chunk`) hoặc 0
   * chạy forward update tối đa 15 bước để ra `S_{p_end}`

2. Từ đó chạy backward đúng như kernel gốc (reconstruct từng bước), và vẫn reload checkpoint ở chunk end để giảm drift.

### Tradeoff

* **VRAM**: giảm mạnh khi nhiều seq ngắn
* **Compute thêm**: tối đa 15 bước forward / sequence (không tính gradient) → thường rẻ hơn đánh đổi VRAM

## 6.2 Khi nào vẫn nên giữ `s_end`?

Nếu bạn biết batch thường gồm **ít sequence nhưng rất dài** (Lavg ≫ 16), thì `s_end` chỉ overhead nhỏ (≈16/Lavg), và giúp:

* giảm chút compute (không cần tail replay)

Nhưng để code “một đường” và tránh VRAM blow-up, mình vẫn khuyên: **mặc định bỏ `s_end`**.

---

# 7) Backward pseudo-code đã sửa (không cần `s_end`)

Mình chỉ viết phần “khác” so với pseudo-code trước: **đoạn dựng `stateT = S_end`** trước khi vào loop backward.

Giữ mapping như kernel gốc: mỗi thread `i` nắm **cột i** của state (vì checkpoint lưu `S^T`).

### Prelude: dựng state cuối sequence

```cuda
// Inputs: start, end (packed indices), p_end = end-1
// stateT[r] will hold S_current[r, i] (column i of S_current)

int p_end = end - 1;

// last global chunk-end <= p_end:
int p0 = (((p_end + 1) & ~15) - 1);  // bit trick for CHUNK_LEN=16
// Explanation: floor((p_end+1)/16)*16 - 1

if (p0 >= start) {
    // load S_{p0} from s_chunk (stored transposed)
    int chunk = p0 >> 4;  // /16
    int base = ((hh * num_chunks + chunk) * C * C) + i * C;
    #pragma unroll
    for (int r=0; r<C; ++r) stateT[r] = s_chunk_[base + r];
} else {
    // No checkpoint inside this sequence prefix => start from zero state
    #pragma unroll
    for (int r=0; r<C; ++r) stateT[r] = 0.0f;
    p0 = start - 1;
}

// Now replay forward from p0+1 .. p_end
for (int p = p0 + 1; p <= p_end; ++p) {

    // load vectors for token p into shared: w[i], k[i], b[i], v[r], sa[r]
    // (same indexing as main loop)

    // Forward update for COLUMN i:
    // S_p[:,i] = S_{p-1}[:,i] * w_p[i] + sa_p[:] * b_p[i] + v_p[:] * k_p[i]
    float wi = w[i], ki = k[i], bi = b[i];
    #pragma unroll
    for (int r=0; r<C; ++r) {
        stateT[r] = stateT[r] * wi + sa[r] * bi + v[r] * ki;
    }
}

// After this, stateT == column i of S_{p_end}, exactly what backward needs.
```

### Sau đó mới vào backward loop như cũ

* Khởi tạo `dstate[:] = 0`, `dstateT[:] = 0`
* Loop `p = p_end .. start`

  * (tuỳ) reload checkpoint nếu `(p+1)&15==0` và p != p_end (để “snap” về forward saved)
  * compute dq
  * reconstruct state_{p-1} bằng chia cho w
  * compute gradients
  * propagate dstate/dstateT

**Điểm quan trọng:** không còn `tl==L-1` load `s_end` nữa, vì `stateT` đã dựng sẵn từ replay.

---

# 8) Verify lại bằng ví dụ [3,2,4] (cu=[0,3,5,9]) và CHUNK_LEN=16

Vì `total_tokens=9 < 16` ⇒ **không có chunk end toàn cục nào**, tức là `s_chunk` không có checkpoint thực sự.

Theo “sửa mới”:

* Seq A: start=0, end=3, p_end=2

  * p0 = (((2+1)&~15)-1) = (-1) < start ⇒ state=0, p0=start-1=-1
  * replay forward p=0..2 (3 bước) ⇒ có S_2
  * backward p=2..0 ⇒ đúng

* Seq B: start=3, end=5, p_end=4

  * p0=-1 < start ⇒ state=0, p0=2
  * replay forward p=3..4 (2 bước) ⇒ có S_4
  * backward p=4..3 ⇒ đúng
  * **Không có bất kỳ dữ liệu nào từ seq C tham gia**, vì block chỉ chạy trong [3,5)

* Seq C: start=5, end=9, p_end=8

  * p0=-1 < start ⇒ state=0, p0=4
  * replay forward p=5..8 (4 bước) ⇒ có S_8
  * backward p=8..5 ⇒ đúng

➡️ Với ví dụ này, “s_end” nếu dùng sẽ cực lãng phí (3 sequences ngắn), trong khi replay chỉ tốn 3+2+4 = 9 bước forward—rất rẻ.

---

# 9) Tác động hiệu năng của sửa chữa

## 9.1 Compute overhead

* Mỗi sequence thêm ≤ 15 bước forward replay (hoặc ≤ L nếu L<16)
* Mỗi bước replay rẻ hơn backward step (không có gradient, không có dstate update)

## 9.2 Bộ nhớ

* Bỏ hẳn `s_end` ⇒ giảm VRAM đáng kể khi nhiều seq ngắn
* `s_chunk` vẫn như cũ: ~ total_tokens/16

## 9.3 Divergence

Replay loop theo token, cả block đi cùng p ⇒ không warp divergence.

## 9.4 Load imbalance

Không thay đổi bản chất: sequence dài vẫn chạy lâu hơn.

---

# 10) Checklist các giả định cần “đóng đinh” (cũng là kẽ hở tiềm tàng)

Để tránh bug “âm thầm”:

1. `cu_seqlens` phải:

   * monotonic non-decreasing
   * `cu[0]=0`, `cu[num_seqs]=total_tokens`
   * dtype int32 trên GPU

2. Mỗi sequence block:

   * nếu `start==end` return ngay

3. Index math:

   * packed index `ind = (p*H + hh)*C + i`

4. Backward-compat:

   * nếu `cu_seqlens == NULL` thì chạy kernel fixed-length như hiện tại (host code chọn kernel), **không nên** nhét `if (cu==nullptr)` trong kernel vì sẽ làm code khó tối ưu.

---

## Kết luận “đã phản biện và sửa”

* Phần **đúng toán** (reset gradient ở boundary, không leak) vẫn đúng và còn “an toàn” nhất khi dùng **1 block/(seq, head)**.
* Phần **checkpoint recommendation** trước đây có kẽ hở VRAM lớn vì `s_end`.
* Sửa tốt hơn: **không cần `s_end`** nếu bạn dùng `sa_` đã lưu để **replay forward tối đa 15 bước** để dựng state cuối sequence trước khi backward.

Nếu bạn muốn, mình có thể viết tiếp:

* pseudo-code forward varlen tương ứng (store `sa_`, store `s_chunk` theo global chunk end),
* và cách thiết kế API PyTorch giống flash-attn (packed tensors + cu_seqlens + max_seqlen) nhưng vẫn backward-compatible với đường fixed-length.
