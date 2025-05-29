# pip install fast-hadamard-transform --no-build-isolation --no-cache-dir
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
python3 setup.py install

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
MAX_JOBS=4 python3 setup.py install # cho RAM <= 32G

pip install schedulefree transformers wandb datasets tqdm==4.67.1
# zstandard scipy