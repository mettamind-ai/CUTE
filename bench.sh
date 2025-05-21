# ./pretrain.py --S --steps 2000 --bs 12  # 36k
./pretrain.py --S --steps 1500 --bs 16    # 48k
./pretrain.py --S --steps  750 --bs 32    # 96k
./pretrain.py --M --steps 1125 --bs 16    # 64k

# ./pretrain.py --S --steps  500 --bs 48    # 144k OOM
# ./pretrain.py --M --steps  750 --bs 24    # 96k OOM
# ./pretrain.py --L --steps 1125 --bs 16    # 64k OOM
