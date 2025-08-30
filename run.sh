############################################## 
## CUDA Toolkit 12.9, CuTE DSL
## https://developer.nvidia.com/cuda-downloads
##############################################

# wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
# sudo sh cuda_12.9.1_575.57.08_linux.run

curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13; # uv venv --python 3.13; source .venv/bin/activate
python -m pip install nvidia-cutlass-dsl==4.1.0

#############################
## Others, data, test run ...
#############################
sudo apt-get install build-essential cmake ninja-build
pip install numpy tqdm wandb einops ninja huggingface_hub optree omegaconf psutil torch==2.8.0 -U --user
# python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129 -U

## Cần thiết để biên dịch flash attn 2 on-the-fly
git clone https://github.com/NVIDIA/cutlass.git flash/cutlass
cd flash/cutlass; git checkout c506e16788cb08416a4a57e11a9067beeee29420;  cd ../.. # flash_attn 2.7.3

uv run python wingpt.py
uv run python data/cached_fineweb10B.py 1
uv run python pretrain.py --bs 1
