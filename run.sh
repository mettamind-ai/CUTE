# wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run
# sudo sh cuda_13.1.0_590.44.01_linux.run
# sudo apt-get install build-essential cmake ninja-build

# curl -LsSf https://astral.sh/uv/install.sh | sh
# uv python install 3.13

python3 -m pip install numpy tqdm wandb einops ninja huggingface_hub optree omegaconf psutil torch==2.9.1

## Cần thiết để biên dịch flash attn 2 on-the-fly
git clone https://github.com/NVIDIA/cutlass.git flash/cutlass
cd flash/cutlass; git checkout c506e16788cb08416a4a57e11a9067beeee29420;  cd ../.. # flash_attn 2.7.3

python3 wingpt.py
python3 data/cached_fineweb10B.py 1
python3 pretrain_gpt.py --bs 1
