import sys, warnings, os, re, ast, platform, subprocess, torch
from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version

def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None: return
    warnings.warn( # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary.
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        if not os.environ.get("MAX_JOBS"):
            import psutil
            max_num_jobs_cores = max(1, os.cpu_count() // 2)
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~9GB? when threads = 4
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)
        super().__init__(*args, **kwargs)

cmdclass = {}
ext_modules = []

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

check_if_cuda_home_none("infllm_v2")
assert CUDA_HOME is not None
cc_flag = ["-gencode", "arch=compute_80,code=sm_80"]
_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_metal_version < Version("11.6"): raise RuntimeError("Only supported on CUDA 11.6 and above.")
# if bare_metal_version >= Version("11.8"): cc_flag += ["-gencode", "arch=compute_90,code=sm_90"] # h100

ext_modules.append(
    CUDAExtension(
        name="infllm_v2.C",
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
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            "nvcc": generator_flag + cc_flag + [
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-O3", "-std=c++17", "--use_fast_math",
                "-lineinfo", "--threads", "8",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",]
        },
        include_dirs=[
            Path(this_dir) / "flash_attn",
            Path(this_dir) / "cutlass" / "include",
        ],
    )
)
setup(
    name='infllm_v2',
    version='0.0.0',
    author_email="acha131441373@gmail.com",
    description="infllm_v2 cuda implementation with flash attention and cutlass",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension} if ext_modules else {},
    python_requires=">=3.7",
    install_requires=["torch", "packaging", "psutil",],
) 
