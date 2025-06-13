#!/usr/bin/env python3
# Triton vs CUDA benchmark for INT8 GEMM with row/col scaling
import torch, triton, ctypes
from triton.testing import do_bench
from optimus import scaled_mm

torch.manual_seed(0)
DEVICE = "cuda"

# Load CUDA kernel
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

def run_benchmark(sizes=[512, 1024, 2048, 4096], rep=20):
    print("\nMatrix Size | Triton (ms) | CUDA (ms) | Speedup | TFLOP/s | Match")
    print("-" * 70)
    
    for size in sizes:
        M, N, K = size, size, size
        ops = 2 * M * N * K  # FLOPs per matrix multiply
        
        # Create test data
        A  = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=DEVICE)
        B  = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=DEVICE)
        As = torch.rand(M, 1, device=DEVICE, dtype=torch.float32)
        Bs = torch.rand(1, N, device=DEVICE, dtype=torch.float32)
        C  = torch.empty((M,N), device=DEVICE, dtype=torch.float32)
        
        # Run Triton kernel and measure performance
        x = scaled_mm(A, B, As, Bs)  # warm-up and reference result
        triton_ms = do_bench(lambda: scaled_mm(A, B, As, Bs), rep=rep) * 1e3
        
        # Run CUDA kernel and measure performance
        torch.cuda.synchronize()
        st = torch.cuda.Event(True)
        ed = torch.cuda.Event(True)
        st.record()
        for _ in range(rep):
            f(A.data_ptr(), B.data_ptr(), C.data_ptr(), As.data_ptr(), Bs.data_ptr(), M, N, K)
            torch.cuda.synchronize()
        ed.record()
        torch.cuda.synchronize()
        cuda_ms = st.elapsed_time(ed)/rep
        
        # Calculate performance metrics
        speedup = triton_ms / cuda_ms
        cuda_tflops = ops / (cuda_ms * 1e6)  # TFLOPs/second
        correct = torch.allclose(x, C, atol=1e-1, rtol=1e-2)
        
        # Print results
        print(f"{M}x{N}x{K} | {triton_ms:9.2f} | {cuda_ms:8.2f} | {speedup:6.1f}x | {cuda_tflops:6.2f} | {'✓' if correct else '✗'}")

if __name__ == "__main__":
    run_benchmark()