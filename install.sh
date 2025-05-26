## CUDA Toolkit 12.8
# wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
# sudo sh cuda_12.8.0_570.86.10_linux.run
# pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 -U

## Torch 2.6 can install flash-attn, causal-conv1d and mamba-ssm
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 numpy wandb 				# cơ bản
# MAX_JOBS=6 pip install flash-attn causal-conv1d --no-build-isolation --no-cache-dir 	# nâng cao
# MAX_JOBS=6 pip install mamba-ssm --no-build-isolation --no-cache-dir 					# cho vui
