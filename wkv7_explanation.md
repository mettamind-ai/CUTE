# Understanding the WKV7 CUDA Kernel - High School Edition 🎓

## What is CUDA?

Think of CUDA like having **thousands of workers** in a factory (your GPU) that can all work at the same time. Instead of doing things one-by-one (like a CPU), we can do many things in parallel!

## The Big Picture

This kernel implements **RWKV7**, which is like a special type of neural network that processes sequences (like sentences) one word at a time, but remembers what came before.

## Thread Organization

Let's say we have:
- **B = 2** batches (2 different sequences)
- **T = 4** time steps (4 words/tokens per sequence)
- **H = 2** heads (2 parallel "attention" mechanisms)
- **C = 4** channels (4 numbers per head - let's keep it small!)

The kernel launches:
- **H × B = 2 × 2 = 4 blocks** (one for each head-batch combination)
- **C = 4 threads per block** (one thread per channel)

```
Block (head=0, batch=0): Threads [0, 1, 2, 3]
Block (head=0, batch=1): Threads [0, 1, 2, 3]
Block (head=1, batch=0): Threads [0, 1, 2, 3]
Block (head=1, batch=1): Threads [0, 1, 2, 3]
```

Each thread handles **one channel** (one number) across all time steps.

---

## Forward Kernel - Step by Step Dry Run

Let's follow **Thread 0** in **Block (head=0, batch=0)** through the first 2 time steps.

### Setup
```c
float state[4] = {0, 0, 0, 0};  // Memory that carries forward
```

### Time Step t=0

**Step 1: Load inputs into shared memory**
```c
// Thread 0 loads its channel's values
q[0] = 0.5   // query value
w[0] = 0.8   // decay weight (after exp(-exp(...)))
k[0] = 0.3   // key value
a[0] = 0.2   // mixing parameter a
b[0] = 0.4   // mixing parameter b
v = 0.6      // value (for this thread)
```

**Step 2: Compute "sa" (state attention)**
```c
sa = 0
for j in [0,1,2,3]:
    sa += a[j] * state[j]

// With our example values:
sa = a[0]*state[0] + a[1]*state[1] + a[2]*state[2] + a[3]*state[3]
sa = 0.2*0 + 0.2*0 + 0.2*0 + 0.2*0  // All states are 0 initially
sa = 0.0
```

**Step 3: Update state and compute output**
```c
y = 0
for j in [0,1,2,3]:
    // Update state[j]
    state[j] = state[j] * w[j] + sa * b[j] + k[j] * v

    // Compute output contribution
    y += state[j] * q[j]
```

Let's trace through with example values:

**For j=0 (Thread 0's own channel):**
```c
state[0] = 0 * 0.8 + 0.0 * 0.4 + 0.3 * 0.6
state[0] = 0 + 0 + 0.18
state[0] = 0.18

y += 0.18 * 0.5 = 0.09
```

**For j=1,2,3 (other threads' channels - we see them via shared memory):**
Let's say:
- state[1] = 0 * 0.7 + 0.0 * 0.3 + 0.4 * 0.6 = 0.24
- state[2] = 0 * 0.9 + 0.0 * 0.5 + 0.2 * 0.6 = 0.12
- state[3] = 0 * 0.6 + 0.0 * 0.4 + 0.5 * 0.6 = 0.30

```c
y += state[1] * q[1]  // Let's say q[1]=0.4
y += 0.24 * 0.4 = 0.096

y += state[2] * q[2]  // Let's say q[2]=0.3
y += 0.12 * 0.3 = 0.036

y += state[3] * q[3]  // Let's say q[3]=0.6
y += 0.30 * 0.6 = 0.18

// Total y for Thread 0:
y = 0.09 + 0.096 + 0.036 + 0.18 = 0.402
```

**Output:** `y_[ind] = 0.402`

**Current state:** `state = [0.18, 0.24, 0.12, 0.30]`

---

### Time Step t=1

**Step 1: Load new inputs**
```c
q[0] = 0.7   // new query
w[0] = 0.8   // same decay
k[0] = 0.4   // new key
a[0] = 0.2   // same
b[0] = 0.4   // same
v = 0.5      // new value
```

**Step 2: Compute sa (now state has values!)**
```c
sa = a[0]*state[0] + a[1]*state[1] + a[2]*state[2] + a[3]*state[3]
sa = 0.2*0.18 + 0.2*0.24 + 0.2*0.12 + 0.2*0.30
sa = 0.036 + 0.048 + 0.024 + 0.060
sa = 0.168
```

**Step 3: Update state (this is the key part!)**
```c
// For j=0:
state[0] = state[0] * w[0] + sa * b[0] + k[0] * v
state[0] = 0.18 * 0.8 + 0.168 * 0.4 + 0.4 * 0.5
state[0] = 0.144 + 0.0672 + 0.2
state[0] = 0.4112
```

Notice how:
- **Old state decays**: `0.18 * 0.8 = 0.144` (80% of old value remains)
- **Attention adds**: `0.168 * 0.4 = 0.0672` (mixing with other channels)
- **New input adds**: `0.4 * 0.5 = 0.2` (current key-value product)

This is the **recurrence** - each step builds on the previous state!

**Step 4: Compute output**
```c
y = state[0]*q[0] + state[1]*q[1] + state[2]*q[2] + state[3]*q[3]
// (similar calculation as before)
```

---

## The Magic Formula

The core update is:
```
new_state = old_state × decay + attention_mix + new_input
```

Where:
- **decay (w)**: How much of old state to keep (0.8 = keep 80%)
- **attention_mix (sa × b)**: How other channels influence this one
- **new_input (k × v)**: The current time step's contribution

## Why Save Chunks?

Every `_CHUNK_LEN_` steps (e.g., every 16 steps), we save the state:
```c
if ((t+1) % 16 == 0) {
    // Save state to s_[...]
}
```

This is for **backward pass** (gradient computation). Instead of recomputing everything backwards, we can "jump" to saved checkpoints and work backwards from there - much faster!

---

## Key Concepts

1. **Shared Memory (`__shared__`)**: All threads in a block can see this. Like a whiteboard everyone can read/write.

2. **State Array**: Each thread maintains its own `state[C]` array. This is the "memory" that carries information forward.

3. **Synchronization (`__syncthreads()`)**: "Wait here until everyone finishes loading data" - ensures all threads see the same shared values.

4. **Parallelism**: While Thread 0 processes channel 0, Thread 1 processes channel 1, Thread 2 processes channel 2, etc. - all at the same time!

---

## Visual Summary

```
Time:  t=0          t=1          t=2          t=3
       │            │            │            │
State: [0,0,0,0] → [0.18,0.24,0.12,0.30] → [0.41,0.52,0.28,0.65] → ...
       │            │            │            │
       Load q,k,v   Load q,k,v   Load q,k,v   Load q,k,v
       Compute sa   Compute sa   Compute sa   Compute sa
       Update state Update state Update state Update state
       Output y     Output y     Output y     Output y
```

Each step:
1. Takes new input (q, k, v)
2. Mixes with old state (via w, a, b)
3. Produces output (y)
4. Updates state for next step

---

## Why This is Fast

- **Parallel**: All channels processed simultaneously
- **Efficient memory**: Shared memory is super fast (like L1 cache)
- **No wasted work**: Each thread does exactly one channel
- **GPU-friendly**: Lots of simple math operations (GPUs love this!)

This is why RWKV can be faster than traditional transformers - it processes sequences step-by-step with efficient state updates, rather than computing attention over all previous tokens at once!
