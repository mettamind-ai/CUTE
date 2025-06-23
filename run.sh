######################################################################## 
## CUDA Toolkit 12.9, CuTE DSL
## https://docs.nvidia.com/cutlass/media/docs/pythonDSL/quick_start.html
########################################################################
# wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
# sudo sh cuda_12.9.1_575.57.08_linux.run
# conda create -n cute python=3.12; conda activate cute
# pip install nvidia-cutlass-dsl==4.0.0

######################################################################
## Others, data, test run ...
######################################################################
# pip install --pre torch==2.8.0.dev20250622 --index-url https://download.pytorch.org/whl/nightly/cu128 -U
pip install numpy tqdm wandb einops ninja torch==2.6.0 -U --user
git clone https://github.com/NVIDIA/cutlass.git flash/attn/cutlass
cd flash/attn/cutlass; git checkout c506e16788cb08416a4a57e11a9067beeee29420;  cd ../../.. # flash_attn 2.7.3

./wingpt.py
if [ ! -f data6400.bin ]; then
    wget https://huggingface.co/datasets/Symonsters/MiniTinyStories/resolve/main/data6400.bin.xz
    xz -d data6400.bin.xz
fi
./pretrain.py --bs 1

# git clone https://github.com/NVIDIA/cutlass.git flash/infllmv2/cutlass
# cd flash/infllmv2/cutlass; git checkout 4c42f73fdab5787e3bb57717f35a8cb1b3c0dc6d;  cd ../../.. # infllmv2
# cd flash; ./bench.py; cd ..
