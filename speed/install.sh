## Cài fast-hadamard-transform qua pip thì ko được, compile thì OK
## pip install fast-hadamard-transform --no-build-isolation --no-cache-dir
# git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
# cd fast-hadamard-transform
# python3 setup.py install

## flash_attn 3 cài từ source trên 4090 ko được
# git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention/hopper
# MAX_JOBS=4 python3 setup.py install # cho RAM <= 32G

pip install https://github.com/IST-DASLab/gemm-int8/releases/download/v1.0.0/gemm_int8-1.0.0-py3-none-manylinux_2_24_x86_64.whl

## qllmt from https://github.com/IST-DASLab/HALO
cd third-party; git clone https://github.com/NVIDIA/cutlass.git
cd ../qllmt; ln -s ../third-party .; python3 setup.py install; rm -rf CMakeFiles CMakeCache.txt CPackConfig.cmake CPackSourceConfig.cmake Makefile cmake_install.cmake ctest/ build/ dist/ *.egg-info/ bin/
