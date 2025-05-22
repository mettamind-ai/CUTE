./pretrain.py --S --steps 1125 --bs 8 --te 999
./pretrain.py --S --steps 1125 --bs 8

# ./pretrain.py --M --steps 1125 --bs 16
# ./pretrain.py --M --steps 1125 --bs 16 --future 10
# ./pretrain.py --M --steps 1125 --bs 16 --future 20
# ./pretrain.py --M --steps 1125 --bs 16 --future 30

# ./pretrain.py --S --steps 1125 --bs 16    #  64k
# ./pretrain.py --S --steps  563 --bs 32    # 128k
# ./pretrain.py --M --steps 1125 --bs 16    #  64k

# ./pretrain.py --M --steps  750 --bs 24    #  96k OOM
# ./pretrain.py --L --steps 1125 --bs 16 --exits 1