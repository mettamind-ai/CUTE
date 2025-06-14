###################################################################### 
##  CUDA Toolkit 12.8 and latest torch
######################################################################
# wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
# sudo sh cuda_12.8.0_570.86.10_linux.run
# pip install --pre torch==2.8.0.dev20250613 --index-url https://download.pytorch.org/whl/nightly/cu128 -U
# NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# MAX_JOBS=6 pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir --use-pep517

######################################################################
## Others, data, test run ...
######################################################################
pip install numpy wandb einops helion==0.0.6 torch==2.7.1 -U --user
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
