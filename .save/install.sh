## CUDA Toolkit 12.8
# wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
# sudo sh cuda_12.8.0_570.86.10_linux.run

## Torch 2.8, fastest
# pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 -U

## Torch 2.6 can install flash-attn & mamba-ssm
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
# pip3 install flash-attn mamba-ssm[causal-conv1d] --no-build-isolation --no-cache-dir


pip install torch numpy wandb
pip install liger_kernel # nếu dùng fused loss (optional)

## FLA
# pip install datasets einops ninja
# pip install causal-conv1d>=1.4.0

#####################################################
# torch 2.7+ không cài được causal-conv1d, flash-attn
# => không dùng được mamba và nsa-triton
##################################################### 

## Mamba
# pip install causal-conv1d
# pip install mamba-ssm
# pip install einops opt_einsum

## flash-attn
# MAX_JOBS=6 pip install flash-attn --no-build-isolation
