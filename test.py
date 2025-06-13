#!/usr/bin/env python3
# Simple test for the CUDA kernel
import torch
import ctypes

# Load the CUDA library
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

# Create small tensors for testing
M, N, K = 32, 32, 32
A  = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
B  = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="cuda")
As = torch.ones(M, 1, device="cuda", dtype=torch.float32)  # Set all scales to 1
Bs = torch.ones(1, N, device="cuda", dtype=torch.float32)  # Set all scales to 1
C  = torch.zeros((M, N), device="cuda", dtype=torch.float32)

# Create CPU reference result
A_cpu = A.cpu().to(torch.float32)
B_cpu = B.cpu().to(torch.float32)
ref = A_cpu @ B_cpu

# Run the kernel
print("Running CUDA kernel...")
try:
    f(A.data_ptr(), B.data_ptr(), C.data_ptr(), As.data_ptr(), Bs.data_ptr(), M, N, K)
    torch.cuda.synchronize()
    print("CUDA kernel executed successfully")
    
    # Check results
    print(f"CUDA output range: [{C.min().item():.2f}, {C.max().item():.2f}]")
    print(f"Reference output range: [{ref.min().item():.2f}, {ref.max().item():.2f}]")
    
    # Compare
    max_diff = (C - ref.cuda()).abs().max().item()
    print(f"Max difference: {max_diff:.2f}")
    print(f"Close match: {torch.allclose(C, ref.cuda(), atol=1.0)}")
    
except Exception as e:
    print(f"Error running CUDA kernel: {e}")