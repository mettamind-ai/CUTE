## For cutlass_mm.cu
mkdir -p third-party; cd third-party; git clone https://github.com/NVIDIA/cutlass.git

## SageAttn
python3 sage_setup.py build
# cp build/lib.linux-x86_64-cpython-310/sageattention/* .

## Flash Attention 3
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention
git checkout b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7 # 2.7.2 release
cd hopper
MAX_JOBS=4 python3 setup.py install
