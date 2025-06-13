#!/usr/bin/env python3
# sudo nvidia-smi -pm 1
# ncu --set full --metrics --target-processes ./bench.py 
# ------------------------------------------------------------
# Triton benchmark: INT8 (A·B -> FP32) with row/col scaling
# ------------------------------------------------------------
import math, time, torch, triton, triton.language as tl
from triton.testing import do_bench
from optimus import scaled_mm

torch.manual_seed(0)
DEVICE = "cuda"

import ctypes, torch, time
lib = ctypes.cdll.LoadLibrary("./liboptimus.so")
f = lib.launch_scaled_int8
f.restype = None
f.argtypes = [
    ctypes.c_void_p,  # A pointer
    ctypes.c_void_p,  # B pointer
    ctypes.c_void_p,  # C pointer
    ctypes.c_void_p,  # As pointer
    ctypes.c_void_p,  # Bs pointer
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int      # K
]


def bench(M, N, K, warmup=10, rep=20):
    A  = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=DEVICE)
    B  = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=DEVICE)
    As = torch.rand(M, 1,  device=DEVICE, dtype=torch.float32)
    Bs = torch.rand(1, N,  device=DEVICE, dtype=torch.float32)
    C  = torch.empty((M,N),device=DEVICE, dtype=torch.float32)

    # warm‑up and debug info
    print(f"A shape: {A.shape}, contiguous: {A.is_contiguous()}, size: {A.numel()}")
    print(f"B shape: {B.shape}, contiguous: {B.is_contiguous()}, size: {B.numel()}")
    print(f"As shape: {As.shape}, contiguous: {As.is_contiguous()}")
    print(f"Bs shape: {Bs.shape}, contiguous: {Bs.is_contiguous()}")
    print(f"C shape: {C.shape}, contiguous: {C.is_contiguous()}, size: {C.numel()}")
    
    # Try to run Triton kernel first to verify input matrices are valid
    print("Running Triton kernel...")
    x = scaled_mm(A, B, As, Bs)
    torch.cuda.synchronize()
    print("Triton kernel succeeded. Running CUDA kernel...")
    
    # Now try the CUDA kernel
    f(A.data_ptr(),B.data_ptr(),C.data_ptr(),As.data_ptr(),Bs.data_ptr(), M,N,K)
    torch.cuda.synchronize()

    # --- Correctness check ---
    print(f"Comparing Triton and CUDA kernel outputs...")
    # NOTE: high tolerance due to different order of operations
    print(f"Are they close? {torch.allclose(x, C, atol=1e-1, rtol=1e-2)}")
    max_diff = (x - C).abs().max().item()
    print(f"Max absolute difference: {max_diff}")
    
    # Debug information
    print(f"Triton output range: [{x.min().item():.2f}, {x.max().item():.2f}]")
    print(f"CUDA output range: [{C.min().item():.2f}, {C.max().item():.2f}]")
    
    # Check for NaN or Inf values
    print(f"Triton has NaN: {torch.isnan(x).any().item()}, Inf: {torch.isinf(x).any().item()}")
    print(f"CUDA has NaN: {torch.isnan(C).any().item()}, Inf: {torch.isinf(C).any().item()}\n")
    # -------------------------

    # timed
    ms = do_bench(lambda: scaled_mm(A, B, As, Bs), rep=rep) * 1e3
    tflops = 2*M*N*K / (ms * 1e9)
    print(f"Triton dp4a: {ms:6.2f} ms {tflops:6.2f} TFLOP/s")

    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    ed = torch.cuda.Event(True)
    st.record()
    for _ in range(rep):
        f(A.data_ptr(),B.data_ptr(),C.data_ptr(),As.data_ptr(),Bs.data_ptr(), M,N,K)
        torch.cuda.synchronize()  # Ensure kernel execution completes before next iteration
    ed.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(ed)/rep
    # Prevent division by zero if timing is too small
    tflops = 2*M*N*K / (max(ms, 0.001) * 1e9)
    print(f"CUDA‑TC kernel: {ms:6.2f} ms", f"{tflops:7.2f} TFLOP/s")


if __name__ == "__main__":
    # Final benchmark with gradually increasing sizes
    print("\nMatrix Size | Triton (ms) | Triton (TFLOP/s) | CUDA (ms) | CUDA (TFLOP/s) | Speedup")
    print("-" * 85)
    
    for size in [512, 1024, 2048, 4096]:
        M, N, K = size, size, size
        print(f"\nBenchmarking {M}x{N}x{K} matrices...")
        A  = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=DEVICE)
        B  = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=DEVICE)
        As = torch.rand(M, 1,  device=DEVICE, dtype=torch.float32)
        Bs = torch.rand(1, N,  device=DEVICE, dtype=torch.float32)
        C  = torch.empty((M,N),device=DEVICE, dtype=torch.float32)
        
        # Run Triton kernel
        x = scaled_mm(A, B, As, Bs)
        triton_ms = do_bench(lambda: scaled_mm(A, B, As, Bs), rep=20) * 1e3
        triton_tflops = 2*M*N*K / (triton_ms * 1e9)
        
        # Run CUDA kernel
        torch.cuda.synchronize()
        st = torch.cuda.Event(True)
        ed = torch.cuda.Event(True)
        st.record()
        for _ in range(20):
            f(A.data_ptr(),B.data_ptr(),C.data_ptr(),As.data_ptr(),Bs.data_ptr(), M,N,K)
            torch.cuda.synchronize()
        ed.record()
        torch.cuda.synchronize()
        cuda_ms = st.elapsed_time(ed)/20
        cuda_tflops = 2*M*N*K / (max(cuda_ms, 0.001) * 1e9)
        
        # Calculate speedup
        speedup = triton_ms / cuda_ms
        
        # Check correctness
        correct = torch.allclose(x, C, atol=1e-1, rtol=1e-2)
        
        # Print results
        print(f"{M}x{N}x{K} | {triton_ms:9.2f} | {triton_tflops:14.2f} | {cuda_ms:8.2f} | {cuda_tflops:13.2f} | {speedup:6.1f}x | {'✓' if correct else '✗'}")
