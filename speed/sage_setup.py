"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
from packaging.version import parse, Version
from typing import List, Set
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"8.6", "8.9"}

# Compiler flags.
CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8",
    "-Xptxas=-v",
    "-diag-suppress=174", # suppress the specific warning
]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

# Iterate over all GPUs on the current machine. Also you can modify this part to specify the architecture if you want to build for specific GPU architectures.
compute_capabilities = set()
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 8:
        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
        continue
    compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    raise RuntimeError("No GPUs found. Please specify the target GPU architectures or build on a machine with GPUs.")
else: print(f"Detect GPUs with compute capabilities: {compute_capabilities}")

# Validate the NVCC CUDA version.
if nvcc_cuda_version < Version("12.0"):
    raise RuntimeError("CUDA 12.0 or higher is required to build the package.")

if nvcc_cuda_version < Version("12.3") and any(cc.startswith("9.0") for cc in compute_capabilities):
    raise RuntimeError("CUDA 12.3 or higher is required for compute capability 9.0.")

# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    assert capability.startswith("8.9")
    num = "89"
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

ext_modules = []

qattn_extension = CUDAExtension(
    name="sageattention._qattn_sm89",
    sources=[
        "csrc/qattn/pybind_sm89.cpp",
        "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(qattn_extension)

# Fused kernels.
fused_extension = CUDAExtension(
    name="sageattention._fused",
    sources=[
        "csrc/fused/pybind.cpp",
        "fcsrc/used/fused.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_extension)

setup(
    name='sageattention', 
    version='2.1.1',  
    author='SageAttention team',
    license='Apache 2.0 License',  
    description='Accurate and efficient plug-and-play low-bit attention.',  
    long_description=open('README.md', encoding='utf-8').read(),  
    long_description_content_type='text/markdown', 
    url='https://github.com/thu-ml/SageAttention', 
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)