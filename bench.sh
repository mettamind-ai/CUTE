
./pretrain.py --S --bs 16 --exits 1                 # minimal
./pretrain.py --S --bs 16 --exits 1 --te 99 --ve 99 # max-inp
./pretrain.py --S --bs 16 --exits 3 --future 10     # max-out

# ./pretrain.py --M --steps 1125 --bs 16    #  64k
# ./pretrain.py --M --steps  750 --bs 24    #  96k OOM
# ./pretrain.py --L --steps 1125 --bs 16 --exits 1
