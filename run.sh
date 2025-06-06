###################################################################### 
##  CUDA Toolkit 12.8 and latest torch with flash-attn
######################################################################
# wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
# sudo sh cuda_12.8.0_570.86.10_linux.run
# pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 -U
# pip3 install torch==2.7.1 -U --user
# sudo apt install libcusparselt0 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention
# python3 setup.py install
# NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

######################################################################
##  Torch 2.6 can install flash-attn, causal-conv1d and mamba-ssm
######################################################################
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
MAX_JOBS=6 pip install flash-attn --no-build-isolation --no-cache-dir
## MAX_JOBS=6 pip install mamba-ssm causal-conv1d --no-build-isolation --no-cache-dir
## pip install --no-build-isolation transformer_engine[pytorch]

######################################################################
## Others, data, test run ...
######################################################################
pip install numpy wandb einops
if [ ! -f data6400.bin ]; then
    wget https://huggingface.co/datasets/Symonsters/MiniTinyStories/resolve/main/data6400.bin.xz
    xz -d data6400.bin.xz
fi
./pretrain.py --M --vocab 6400 --ohmai 2028
