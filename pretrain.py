#!/usr/bin/env python3
import os, sys, types, re
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
from optimus import Muon1GPU as Muon, convert_int8_mixed_precision

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=64) # 64k tokens/step works best in most cases
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--vocab", type=int, default=32000)
parser.add_argument("--ohmai", type=int, default=None)
parser.add_argument("--minloss", type=float, default=0)
parser.add_argument("--int8ig", type=str, default="head")   # int8 ignore params (`proj|head` => all Linear) 
parser.add_argument("--schedule", type=json.loads, default={"warmup": 0.05, "decay": 0.15})
parser.add_argument("--future", type=int, default=0, choices=range(50))  # % in final loss
parser.add_argument("--muonlr", type=float, default=0.030)  # default 0.02, modded gpt 0.025
parser.add_argument("--adamlr", type=float, default=0.003)  # 3e-4
parser.add_argument("--wd", type=float, default=0.01)       # std=0.01 (1e-2)
parser.add_argument("--ve", type=int, default=None)         # số value embeds được bổ xung 
parser.add_argument("--te", type=int, default=1)            # số token embeds 
for x in "T C XS S L M".split():
    parser.add_argument(f"--{x}", action="store_true")
args = parser.parse_args()

if args.T:
    args.steps = 100 # thử nhỏ thôi
    if args.bs == 64: args.bs = 1
rank, world_size, is_master = 0, 1, True # 1 GPU
def print0(msg): is_master and print(msg)
torch.manual_seed(1981 + rank)

#############################
## Init model for pretraining
#############################
from wingpt import WinGPT, get_cu_max_seqlens_from
tokens_per_batch = args.bs*1024

if  args.L: # (L)arge ~ 999m
    model = WinGPT(
        future_percent=args.future,
        ve=args.ve, dim=2048, n_layers=27,
        te=args.te, num_heads=8, num_kv_heads=4,
        vocab_size=args.vocab, max_seq_len=tokens_per_batch,
        active_vocab=args.ohmai,
    )
elif args.M: # (M)edium ~ 666m
    model = WinGPT(
        future_percent=args.future,
        ve=args.ve, dim=1664, n_layers=26,
        te=args.te, num_heads=8, num_kv_heads=4,
        vocab_size=args.vocab, max_seq_len=tokens_per_batch,
        active_vocab=args.ohmai,
    )
elif args.S: # (S)mall ~ 333m
    model = WinGPT(
        future_percent=args.future,
        ve=args.ve, dim=1280, n_layers=22,
        te=args.te, num_heads=8, num_kv_heads=4,
        vocab_size=args.vocab, max_seq_len=tokens_per_batch,
        active_vocab=args.ohmai,
    )
else:        # (XS)mall ~ 100m
    model = WinGPT(
        future_percent=args.future,
        ve=args.ve, dim=768, n_layers=16,
        te=args.te, num_heads=8, num_kv_heads=4,
        vocab_size=args.vocab, max_seq_len=tokens_per_batch,
        active_vocab=args.ohmai,
    )
model = model.cuda()
names, params = convert_int8_mixed_precision(model, ignore=args.int8ig)

def find_key(s):
    m = re.search(r'(blocks\.\d+\.)?(.*)', s)
    return "*" + m.group(2) if m.group(1) else m.group(2)

count = len(names)
total_params = sum(p.numel() for p in model.parameters())
total_names = sum(1 for p in model.parameters())
names = sorted(set(find_key(x) for x in names))
percent = (params/total_params)*100

print0(f"""\nPHÂN CHIA PARAMS VÀO DTYPES:
* {count} INT8 Mixed Weights {percent:.1f}% {params:,}
* {total_names - count} BF16/ FP32 Weights {100-percent:.1f}% {total_params - params:,}
INT8: {names}""")

#################
## Data loader ##
#################
data = np.memmap(f"data{args.vocab}.bin", dtype=np.uint16, mode="r")
CTX  = tokens_per_batch + 2
N    = len(data) - CTX
WIN  = torch.arange(CTX)

def get_batch():
    idx = torch.randint(0, N, (1,)) + WIN    # shape = (CTX)
    x = torch.from_numpy(data[idx.numpy()])  # Tensor → pin_memory → GPU.
    return x.pin_memory().to("cuda", dtype=torch.long, non_blocking=True)
batch = get_batch()


class LRSchedule:
    def __init__(self, lr, n_steps, decay_type="linear",
        warmup: float = 0.05, # 05% warmup đi từ 0 -> init_lr
        decay:  float = 0.15, # 80% stable @ init_lr, 15% decay to 0
    ):
        self.lr = lr
        self.t1 = int(n_steps * warmup)
        self.t2 = int(n_steps * (1 - decay))
        self.t3 = n_steps
        self.decay_type = decay_type
        assert self.t1 <= self.t2
        assert decay_type in ("linear", "cosine")

    def get_lr(self, step: int) -> float:
        if step < 0 or step > self.t3: return 0.0
        if step < self.t1: return self.lr * step / self.t1
        if step < self.t2: return self.lr

        progress = (step - self.t2) / (self.t3 - self.t2)
        if self.decay_type == "linear": return self.lr * (1 - progress)
        return 0.5 * self.lr * (1 + math.cos(progress * math.pi)) # cosine

    def set_lr(self, step: int, optim: torch.optim.Optimizer):
        for param_group in optim.param_groups:
            if isinstance(param_group["lr"], Tensor): param_group["lr"].fill_(self.get_lr(step))
            else: param_group["lr"] = self.get_lr(step)

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

adam_params_count = sum(p.numel() for p in adam_params)
muon_params_count = sum(p.numel() for p in muon_params)
total_params = sum(p.numel() for p in model.parameters())

adam_ratio = adam_params_count / total_params
muon_ratio = muon_params_count / total_params

print0(f"""\nPHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: {adam_ratio*100:.1f}% {adam_params_count:,}
* Muon: {muon_ratio*100:.1f}% {muon_params_count:,}
 TOTAL: {           100:.1f}% {total_params:,}""")

adam_keys = sorted(set(find_key(x) for x in adam_n_params.keys()))
muon_keys = sorted(set(find_key(x) for x in muon_n_params.keys()))

print0(f"Adam: {sorted(set(x.replace('.weight','') for x in adam_keys))}")
print0(f"Muon: {sorted(set(x.replace('.weight','').replace('future.block.','') for x in muon_keys))}")
for x in muon_params: assert x.ndim >= 2

# Dùng torch.optim.AdamW cho chuẩn xác
adam_optim = torch.optim.AdamW(adam_params, lr=args.adamlr, weight_decay=args.wd, fused=True)
muon_optim = Muon(muon_params, lr=args.muonlr, weight_decay=args.wd, rank=rank, world_size=world_size)

muon_lr_schedule = LRSchedule(args.muonlr, args.steps, **args.schedule)
adam_lr_schedule = LRSchedule(args.adamlr, args.steps, **args.schedule)

#############################
## LOSS FUNCTION & PREPARE ##
#############################
from wingpt import simple_loss_fn as lossf
if args.C: lossf = torch.compile(lossf); print(">>> torch.compile(lossf) <<<")

print0(f"""\nCHUẨN BỊ HUẤN LUYỆN:
* GPU(s) {world_size}
* compile? {args.C}
* future? {model.future_ratio}
* {lossf.__name__}
* {tokens_per_batch//1024}k seq/step
""")
model.train()
step = 0
log_interval = 5
lossv = 9999 # cần cho args.minloss

if args.T: log_interval = 2
else: logger = wandb.init(dir="/tmp", config=args,)

#############################
## Training loop
#############################
started_at = time.time()
while step < args.steps and lossv > args.minloss:

    tokens, targets, future = batch[:-2], batch[1:-1], batch[2:]
    c, m = get_cu_max_seqlens_from(tokens, eot=args.vocab-1)

    loss = lossf(model, tokens, targets, future, c, m)
    batch = get_batch() # async prefetch next batch
    loss.backward()

    adam_lr_schedule.set_lr(step, adam_optim)
    muon_lr_schedule.set_lr(step, muon_optim)

    if (step - 1) % log_interval == 0 or step == args.steps - 1:
        lossv = loss.item()
        adam_lr = adam_optim.param_groups[0]["lr"]
        muon_lr = muon_optim.param_groups[0]["lr"]
        log_dict = dict(loss=lossv, muon_lr=muon_lr, adam_lr=adam_lr)

        if not args.T: logger.log(log_dict, step=step)
        pbar.set_postfix(loss=lossv, lr=muon_lr) # tối thiểu chiều rộng

    muon_optim.step()
    adam_optim.step()
    muon_optim.zero_grad()
    adam_optim.zero_grad()
 
    if step == 0:            # sau khi compile và chạy model forward & backward 1 lần ...
        time0 = time.time()  # ... thì mới record time0 và khởi tạo pbar 
        pbar = tqdm(total=args.steps, dynamic_ncols=True, disable=not is_master)
    elif step == 1:
        print0(f">>> First Step Took {int(time.time() - started_at)} Seconds <<<")
        time0 -= time.time() - time0 # điều chỉnh lại time0
    pbar.update()

    step += 1
    if step % log_interval == 0 and not args.T:
        log_dict = dict(
            max_memory_allocated=torch.cuda.max_memory_allocated(),
            num_tokens_seen_millions=tokens_per_batch * step / 1e6,
            tokens_per_second=tokens_per_batch * log_interval / (time.time() - time0),
        )
        time0 = time.time()
        logger.log(log_dict, step=step)
    # END of Training Loop
model.update_embeddings()
if not args.T: logger.finish()
