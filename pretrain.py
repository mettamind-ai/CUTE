#!/usr/bin/env python3
import os, sys, types
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse, json, time
from datetime import datetime
from pathlib import Path

import torch, wandb
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch import Tensor, nn

from optimus import Muon
from train_utils import LRSchedule, print_model_stats, get_grad_norm

# 98m@46.3min 16x4k 64k/step 36.8k/s => lợi cả computing và loss
# 98m@44.5min 24*4k 96k/step 35.3k/s
DEFAULT_BS, DEFAULT_CTX = 16, 1024*4 # 64k tokens/step works best!
parser = argparse.ArgumentParser()  # bs chia hết cho 16 giúp perf ổn định hơn ???
parser.add_argument("--bs", type=int, default=DEFAULT_BS)  
parser.add_argument("--ctx", type=int, default=DEFAULT_CTX)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--vocab", type=int, default=6400)
parser.add_argument("--minloss", type=float, default=0)
parser.add_argument("--int8rd", type=str, default="abit", choices="abit half full hack".split())
parser.add_argument("--funloss", type=str, default="simple", choices="simple fused".split())
parser.add_argument("--schedule", type=json.loads, default={"warmup":0.05,"decay":0.15})
parser.add_argument("--muonlr", type=float, default=0.030) # default 0.02, modded gpt 0.025
parser.add_argument("--adamlr", type=float, default=0.003) # 3e-4
parser.add_argument("--wd", type=float, default=0.01)      # std=0.01 (1e-2)
parser.add_argument("--exits", type=int, default=3)
for x in "bf16  test  compile  S L M  future".split():
    parser.add_argument(f"--{x}", action="store_true")
args = parser.parse_args()
INT8 = COMPILE = False

if args.test:               # test trên GPU laptop 4G vram
    args.bf16 = True        # luôn dùng bf16 vì ko chạy đc int8
    args.compile = False    # luôn tắt compile để không phải đợi
    args.steps = 100        # thử nhỏ cho vui
    args.funloss = "fused"  # bf16 + fused loss tiết kiệm vram
    args.bs = 1
elif args.bf16:  # Khởi động nhanh để dò số tokens / step hơn lý
    args.funloss = "fused"  # fused loss lúc compile sẽ báo warning
    torch.set_float32_matmul_precision('high') # tăng tốc bf16
    torch.backends.cuda.matmul.allow_tf32  = True
else: # int8, need for speed mode!
    INT8 = True             # giúp 1.5x speedup
    args.compile = True     # compile cho tốc độ tối đa
    os.environ['INT8_MIXED_SR'] = args.int8rd
    from optimus import convert_int8_mixed_precision

if args.compile: # Bật biến env trước khi load wingpt.py
    os.environ['COMPILE_LOSS_METHOD'] = "1"

rank = 0
is_dist = False
world_size = 1
is_master = (rank == 0)
def print0(msg): is_master and print(msg)

#############################
## Init model for pretraining
#############################
from wingpt import WinGPT
if  args.L: # (L)arge @ 4090 => 999m, 48k
    model = WinGPT(
        ve=5, dim=2048, n_layers=32,
        num_heads=8, num_kv_heads=4,
        vocab_size=args.vocab, max_seq_len=args.ctx,
        future=args.future, exits=args.exits,
    )
elif args.M: # (M)edium @ 4090 => 666m, 64k
    model = WinGPT(
        ve=5, dim=1664, n_layers=30, # dim=1024 1280 1536 1792 2048
        num_heads=8, num_kv_heads=4, 
        vocab_size=args.vocab, max_seq_len=args.ctx,
        future=args.future, exits=args.exits,
    )
else:        # (S)mall @ 4090 => 333m, 36k
    # if args.bs == DEFAULT_BS: args.bs = 12
    if args.ctx == DEFAULT_CTX: args.ctx = 1024*3
    model = WinGPT(
        ve=3, dim=1280, n_layers=22,
        num_heads=8, num_kv_heads=4,
        vocab_size=args.vocab, max_seq_len=args.ctx,
        future=args.future, exits=args.exits,
    )
    
TOKS_PER_STEP = args.bs*args.ctx
model = model.cuda()
# print0(f"1st_LAYER {model.blocks[0]}")
# print0(f'ATTN_IMPL sdpa')

if INT8:
    count = convert_int8_mixed_precision(model)
    print0(f"INT8 Mixed Precision: {count} Linear converted.")

#################
## Data loader ##
#################
# https://github.com/karpathy/nanoGPT/blob/master/train.py#L115
data = np.memmap(f'data{args.vocab}.bin', dtype=np.uint16, mode='r')
ll = (args.ctx+2)
def get_batch():
    ix = torch.randint(len(data) - ll, (args.bs,))
    batch = torch.stack([torch.from_numpy((data[i:i+ll]).astype(np.int64)) for i in ix])
    # pin arrays which allows us to move them to GPU asynchronously (non_blocking=True)
    return batch.pin_memory().to("cuda", non_blocking=True)
batch = get_batch()

#############################
## Init Optimizer(s)
#############################

m = model
adam_n_params = {n: p for n, p in m.named_parameters() if "fc" not in n and "proj" not in n}
muon_n_params = {n: p for n, p in m.named_parameters() if "fc" in n or "proj" in n}

# Kiểm tra tính đúng đắn của việc phân loại tham số
a = set(adam_n_params.keys())
b = set(muon_n_params.keys())
assert len(a & b) == 0, f"trùng nhau {a & b}"

# Đảm bảo không bỏ sót tham số của model
all_names = set(name for name, _ in m.named_parameters())
classified_names = a | b
assert all_names == classified_names, f"""Parameter classification mismatch:
missing {all_names - classified_names},
extra {classified_names - all_names}"""

adam_params = list(adam_n_params.values())
muon_params = list(muon_n_params.values())
for x in muon_params: assert x.ndim >= 2


adam_params_count = sum(p.numel() for p in adam_params)
muon_params_count = sum(p.numel() for p in muon_params)
total_params = sum(p.numel() for p in model.parameters())

adam_ratio = adam_params_count / total_params
muon_ratio = muon_params_count / total_params

print0(f"""\nPHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: {adam_ratio*100:.1f}% {adam_params_count:,}
* Muon: {muon_ratio*100:.1f}% {muon_params_count:,}
 TOTAL: {           100:.1f}% {total_params:,}""")

import re
def find_key(s):
    m = re.search(r'(blocks\.\d+\.)?(.*)', s)
    # print(s, m.group(2))
    return m.group(2)

adam_keys = sorted(set(find_key(x) for x in adam_n_params.keys()))
muon_keys = sorted(set(find_key(x) for x in muon_n_params.keys()))

print0(f"Adam: {[x.replace('.weight','') for x in adam_keys]}")
print0(f"Muon: {[x.replace('_proj','').replace('.weight','') for x in muon_keys]}")

# Dùng torch.optim.AdamW cho chuẩn xác + fused to save vram
adam_optim = torch.optim.AdamW(adam_params, lr=args.adamlr, weight_decay=args.wd, fused=True)
muon_optim = Muon(muon_params, lr=args.muonlr, weight_decay=args.wd, rank=rank, world_size=world_size)

muon_lr_schedule = LRSchedule(args.muonlr, args.steps, **args.schedule)
adam_lr_schedule = LRSchedule(args.adamlr, args.steps, **args.schedule)

#############################
## LOSS FUNCTION & PREPARE ##
#############################
if   args.funloss ==  "simple": from wingpt import  simple_loss_fn as loss_fn
elif args.funloss ==   "fused": from wingpt import   fused_loss_fn as loss_fn
else: assert False, f"Not support {args.funloss}"

print0(f"""CHUẨN BỊ HUẤN LUYỆN
* is_dist {is_dist}, world_size {world_size}
* int8? {INT8},  loss_fn {args.funloss}, future? {not not model.future}
* device_bs {args.bs}, seq_len {args.ctx}, {(TOKS_PER_STEP)//1024}k tokens/step
* layers {len(model.blocks)}, exits {model.exit_ids}, scales {model.exit_scales}
""")
model.train()
step = 0
log_interval = 10
lossf = 9999 # cần cho args.minloss

if args.test:log_interval = 2
else: logger = wandb.init(dir="/tmp", config=args,)

#############################
## Training loop
#############################
while step < args.steps and lossf > args.minloss:
    # https://github.com/karpathy/nanoGPT/blob/master/train.py#L292C9-L292C20
    tokens, targets, future = batch[:, :-2], batch[:, 1:-1], batch[:, 2:]
    loss  = loss_fn(model, tokens, targets, future)
    batch = get_batch()  # async prefetch next batch
    loss.backward()

    adam_lr_schedule.set_lr(step, adam_optim)
    muon_lr_schedule.set_lr(step, muon_optim)

    if (step - 1) % log_interval == 0 or step == args.steps - 1:
        grad_norm = get_grad_norm(model)

        lossf = loss.item()
        adam_lr = adam_optim.param_groups[0]["lr"]
        muon_lr = muon_optim.param_groups[0]["lr"]
        log_dict = dict(loss=lossf, grad_norm=grad_norm, 
            muon_lr=muon_lr, adam_lr=adam_lr)

        if not args.test: logger.log(log_dict, step=step)
        pbar.set_postfix(loss=lossf, adam=adam_lr, muon=muon_lr)

    muon_optim.step(); muon_optim.zero_grad()
    adam_optim.step(); adam_optim.zero_grad()
 
    if step == 0:            # sau khi compile và chạy model forward & backward 1 lần ... 
        time0 = time.time()  # ... thì mới record time0 và khởi tạo pbar 
        pbar = tqdm(total=args.steps, dynamic_ncols=True, disable=not is_master)
    elif step == 1:
        time0 -= time.time() - time0 # điều chỉnh lại time0
    pbar.update()

    step += 1
    if step % log_interval == 0 and not args.test:
        tokens_per_batch = args.bs * args.ctx
        log_dict = dict(
            max_memory_allocated=torch.cuda.max_memory_allocated(),
            num_tokens_seen_millions=tokens_per_batch * step / 1e6,
            tokens_per_second=tokens_per_batch * log_interval / (time.time() - time0),
        )
        time0 = time.time()
        logger.log(log_dict, step=step)
    # END of Training Loop
if not args.test: logger.finish()
