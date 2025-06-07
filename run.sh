###################################################################### 
##  CUDA Toolkit 12.8 and latest torch
######################################################################
# wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
# sudo sh cuda_12.8.0_570.86.10_linux.run
# pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 -U
# NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

######################################################################
##  Torch 2.6 can install flash-attn, causal-conv1d and mamba-ssm
######################################################################
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --user
MAX_JOBS=6 pip install flash-attn --no-build-isolation --no-cache-dir
# MAX_JOBS=6 pip install mamba-ssm causal-conv1d --no-build-isolation --no-cache-dir
# pip install --no-build-isolation transformer_engine[pytorch]

######################################################################
## Others, data, test run ...
######################################################################
# pip install numpy wandb einops torch==2.7.1 -U --user

git clone https://github.com/NVIDIA/cutlass.git flash/cutlass
cd flash/cutlass; git checkout a75b4ac483166189a45290783cb0a18af5ff0ea5; cd ../..

if [ ! -f data6400.bin ]; then
    wget https://huggingface.co/datasets/Symonsters/MiniTinyStories/resolve/main/data6400.bin.xz
    xz -d data6400.bin.xz
fi

./wingpt.py
# ./pretrain.py --M --vocab 6400 --ohmai 2028
