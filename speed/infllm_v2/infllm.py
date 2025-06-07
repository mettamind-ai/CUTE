import torch, os, warnings, psutil, math
import torch.utils.cpp_extension

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6;8.9"  # 3050ti, 4090
os.environ['MAX_JOBS'] = "4"

free_memory_gb = int(psutil.virtual_memory().available) // (1024 ** 3)
max_jobs = math.ceil(free_memory_gb / 9)  # each JOB peak memory cost is ~9GB? when threads = 4
print(f"free_memory_gb {free_memory_gb}, max_jobs {max_jobs}")
os.environ["MAX_JOBS"] = str(max_jobs)

NVCC_FLAGS = [
	"-O3", "-std=c++17",
	"-U__CUDA_NO_HALF_OPERATORS__",
	"-U__CUDA_NO_HALF_CONVERSIONS__",
	"-U__CUDA_NO_HALF2_OPERATORS__",
	"-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--use_fast_math", "-lineinfo", "--threads=8",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-Xptxas=-v", "-diag-suppress=174", # suppress the specific warning
]
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += ["-gencode", f"arch=compute_{80},code=sm_{80}"]

_infllm_v2 = torch.utils.cpp_extension.load(
    "infllm_v2.CUTE",
    sources=[
	    "entry.cu",
	    "flash_api.cpp",
	    "flash_attn/flash_fwd_hdim128_bf16_sm80.cu",
	    "flash_attn/flash_bwd_hdim128_bf16_sm80.cu",
	    "flash_attn/flash_fwd_split_hdim128_bf16_sm80.cu",
	    "flash_attn/flash_fwd_block_hdim128_bf16_sm80.cu",
	    "flash_attn/flash_bwd_block_hdim128_bf16_sm80.cu",
	    "flash_attn/flash_fwd_splitkv_block_hdim128_bf16_sm80.cu",
    ],
    extra_cuda_cflags=NVCC_FLAGS,
    extra_include_paths=[
    	"flash_attn",
        "cutlass/include",
    ],
)
