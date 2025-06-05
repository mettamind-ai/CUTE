#!/usr/bin/env python3
import re, os, sys, types, argparse, json, time, torch, wandb, numpy as np
import torch.distributed as dist, torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch import Tensor, nn
from optimus import Muon1GPU as Muon, convert_int8_mixed_precision

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=64) # 64k tokens/step works best in most cases
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--vocab", type=int, default=32000)
parser.add_argument("--ohmai", type=int, default=None)
parser.add_argument("--int8ig", type=str, default="head")   # int8 ignore params (`proj|head` => all Linear) 
parser.add_argument("--schedule", type=json.loads, default={"warmup": 0.05, "decay": 0.15})
parser.add_argument("--muonlr", type=float, default=0.030)  # default 0.02, modded gpt 0.025
parser.add_argument("--adamlr", type=float, default=0.003)  # 3e-4
parser.add_argument("--wd", type=float, default=0.01)       # std=0.01 (1e-2)
for x in "C S L M".split(): parser.add_argument(f"--{x}", action="store_true")
args = parser.parse_args()

rank, world_size, is_master = 0, 1, True # 1 GPU
torch.manual_seed(1981 + rank)

def print0(msg): is_master and print(msg)
tokens_per_batch = args.bs*1024

data = np.memmap(f"data{args.vocab}.bin", dtype=np.uint16, mode="r")
CTX  = tokens_per_batch + 1
N    = len(data) - CTX
WIN  = torch.arange(CTX)
def get_batch():
    idx = torch.randint(0, N, (1,)) + WIN    # shape = (CTX)
    x = torch.from_numpy(data[idx.numpy()])  # Tensor → pin_memory → GPU.
    return x.pin_memory().to("cuda", dtype=torch.long, non_blocking=True)
batch = get_batch()

#############################
## Init model for pretraining
#############################
from wingpt import WinGPT, get_cu_max_seqlens_from

if  args.L: # (L)arge ~ 999m
    model = WinGPT(dim=2048, n_layers=27, num_heads=16, num_kv_heads=4, head_dim=64,
        vocab_size=args.vocab, max_seq_len=tokens_per_batch, active_vocab=args.ohmai,)
elif args.M:# (M)edium ~ 666m
    model = WinGPT(dim=1664, n_layers=26, num_heads=16, num_kv_heads=4, head_dim=64,
        vocab_size=args.vocab, max_seq_len=tokens_per_batch, active_vocab=args.ohmai,)
else:       # (S)mall ~ 333m
    model = WinGPT(dim=1280, n_layers=22, num_heads=16, num_kv_heads=4, head_dim=64,
        vocab_size=args.vocab, max_seq_len=tokens_per_batch, active_vocab=args.ohmai,)

names, params = convert_int8_mixed_precision(model, ignore=args.int8ig)
def find_key(s):
    m = re.search(r'(blocks\.\d+\.)?(.*)', s)
    return "*" + m.group(2) if m.group(1) else m.group(2)
total_params = sum(p.numel() for p in model.parameters())
short_names = sorted(set(find_key(x) for x in names))
percent = (params/total_params)*100
print0(f"""\nPHÂN CHIA PARAMS VÀO DTYPES:
* {len(names)} INT8 Mixed Weights {percent:.1f}% {params:,}
* {len(list(model.parameters())) - len(names)} BF16/ FP32 Weights {100-percent:.1f}% {total_params - params:,}
INT8: {short_names}""")

#########################
##  Init Optimizer(s)  ##
#########################
class LRSchedule:
    def __init__(self, lr, n_steps, decay_type="linear", warmup: float = 0.05, decay:  float = 0.15,):
        self.lr = lr
        self.t1 = int(n_steps * warmup)
        self.t2 = int(n_steps * (1 - decay))
        self.t3 = n_steps
        self.decay_type = decay_type
        assert self.t1 <= self.t2

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

adam_params = [p for n, p in model.named_parameters() if "proj" not in n]
muon_params = [p for n, p in model.named_parameters() if "proj" in n]

adam_optim = torch.optim.AdamW(adam_params, lr=args.adamlr, weight_decay=args.wd, fused=True)
muon_optim = Muon(muon_params, lr=args.muonlr, weight_decay=args.wd, rank=rank, world_size=world_size)

muon_lr_schedule = LRSchedule(args.muonlr, args.steps, **args.schedule)
adam_lr_schedule = LRSchedule(args.adamlr, args.steps, **args.schedule)


##############
## TRANING  ##
##############
from wingpt import simple_loss_fn as lossf
if args.C:
    lossf = torch.compile(lossf)
    for x in model.blocks: x.compile()
model = model.cuda()
model.train()

print0(f"""\nCHUẨN BỊ HUẤN LUYỆN:
* GPU(s) {world_size}
* compile? {args.C}
* {lossf.__name__}
* {tokens_per_batch//1024}k seq/step
""")
step = 0
log_interval = 5 
logger = wandb.init(dir="/tmp", config=args,)

started_at = time.time()
while step < args.steps:  # training loop

    tokens, targets = batch[:-1], batch[1:]
    c, m = get_cu_max_seqlens_from(tokens, eot=args.vocab-1)

    loss = lossf(model, tokens, targets, c, m)
    batch = get_batch() # async prefetch next batch
    loss.backward()

    adam_lr_schedule.set_lr(step, adam_optim)
    muon_lr_schedule.set_lr(step, muon_optim)

    if (step - 1) % log_interval == 0 or step == args.steps - 1:
        lossv = loss.item()
        adam_lr = adam_optim.param_groups[0]["lr"]
        muon_lr = muon_optim.param_groups[0]["lr"]
        log_dict = dict(loss=lossv, muon_lr=muon_lr, adam_lr=adam_lr)

        logger.log(log_dict, step=step)
        pbar.set_postfix(loss=lossv, lr=muon_lr) # tối thiểu chiều rộng

    muon_optim.step(); muon_optim.zero_grad()
    adam_optim.step(); adam_optim.zero_grad()
 
    if step == 0:            # sau khi compile và chạy model forward & backward 1 lần ...
        time0 = time.time()  # ... thì mới record time0 và khởi tạo pbar 
        pbar = tqdm(total=args.steps, dynamic_ncols=True, disable=not is_master)
    elif step == 1:
        print0(f">>> First Step Took {int(time.time() - started_at)} Seconds <<<")
        time0 -= time.time() - time0 # điều chỉnh lại time0

    pbar.update()
    step += 1

    if step % log_interval == 0:
        logger.log(dict(max_memory_allocated=torch.cuda.max_memory_allocated(), num_tokens_seen_millions=tokens_per_batch*step,
                        tokens_per_second=tokens_per_batch*log_interval / (time.time() - time0),), step=step)
        time0 = time.time()
model.update_async_weight()
logger.finish()
