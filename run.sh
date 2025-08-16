############################################## 
## CUDA Toolkit 12.9, CuTE DSL
## https://developer.nvidia.com/cuda-downloads
##############################################

# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# bash miniconda.sh -b -u -p ~/miniconda3
# wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
# sudo sh cuda_12.9.1_575.57.08_linux.run
# conda create -n cute python=3.12; conda activate cute
# pip install nvidia-cutlass-dsl==4.1.0
# python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129 -U

#############################
## Others, data, test run ...
#############################
# sudo apt-get install build-essential cmake ninja-build

python -m pip install git+https://github.com/fla-org/flash-linear-attention.git -U
pip install numpy tqdm wandb einops ninja huggingface_hub optree omegaconf psutil torch==2.8.0 -U --user
git clone https://github.com/NVIDIA/cutlass.git flash/cutlass
cd flash/cutlass; git checkout c506e16788cb08416a4a57e11a9067beeee29420;  cd ../.. # flash_attn 2.7.3

./wingpt.py
python3 data/cached_fineweb10B.py 1
./pretrain.py --bs 1
