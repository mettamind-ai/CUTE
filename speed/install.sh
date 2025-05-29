## Cài fast-hadamard-transform qua pip thì ko được, compile thì OK
## pip install fast-hadamard-transform --no-build-isolation --no-cache-dir
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
python3 setup.py install

## flash_attn 3 cài từ source trên 4090 ko được
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
MAX_JOBS=4 python3 setup.py install # cho RAM <= 32G
