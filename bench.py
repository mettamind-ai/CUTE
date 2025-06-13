#!/usr/bin/env python3
# ncu --set full --metrics --target-processes ./bench.py 
# ------------------------------------------------------------
# Triton benchmark: INT8 (A·B -> FP32) with row/col scaling
# ------------------------------------------------------------
import math, time, torch, triton, triton.language as tl
from triton.testing import do_bench
from optimus import scaled_mm

torch.manual_seed(0)
DEVICE = "cuda"

def make_data(M, N, K, dtype=torch.int8):
    A  = torch.randint(-128, 127, (M, K), dtype=dtype, device=DEVICE)
    B  = torch.randint(-128, 127, (K, N), dtype=dtype, device=DEVICE)
    As = torch.rand(M, 1,  device=DEVICE, dtype=torch.float32)
    Bs = torch.rand(1, N,  device=DEVICE, dtype=torch.float32)
    return A, B, As, Bs

def bench(M, N, K, warmup=20, rep=100):
    A, B, As, Bs = make_data(M, N, K)
    # warm‑up
    for _ in range(warmup):
        _ = scaled_mm(A, B, As, Bs)
    torch.cuda.synchronize()
    # timed
    ms = do_bench(lambda: scaled_mm(A, B, As, Bs), rep=rep) * 1e3
    tflops = 2*M*N*K / (ms * 1e9)
    print(f"Triton dp4a: {ms:6.2f} ms {tflops:6.2f} TFLOP/s")

if __name__ == "__main__":
    bench(M=4096, N=4096, K=4096)
