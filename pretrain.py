#!/usr/bin/env python3
from wingpt import WinGPT, get_cu_max_seqlens_from, fused_loss_fn as lossf
from optimus import Muon1GPU as Muon, convert_int8_mixed_precision

import re, os, sys, types, argparse, json, time, torch, wandb, numpy as np
import torch.distributed as dist, torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch import Tensor, nn

parser = argparse.ArgumentParser()
parser.add_argument("--bs",     type=int, default=128)
parser.add_argument("--steps",  type=int, default=1000)
parser.add_argument("--vocab",  type=int, default=8192)
args = parser.parse_args()

torch.manual_seed(1981)
tokens_per_batch = args.bs*1024

model = WinGPT( dim=1024, expansion=4, n_layers=25, num_heads=16, num_kv_heads=8, head_dim=64,
                vocab_size=args.vocab, max_seq_len=tokens_per_batch) # 360m; config ~= qwen3 0.6b

## Load data, sooner better
data = np.memmap(f"data/{args.vocab}.bin", dtype=np.uint16, mode="r")
CTX  = tokens_per_batch + 1
N    = len(data) - CTX
WIN  = torch.arange(CTX)

def get_batch():
    idx = torch.randint(0, N, (1,)) + WIN    # shape = (CTX)
    x = torch.from_numpy(data[idx.numpy()])  # Tensor → pin_memory → GPU.
    return x.pin_memory().to("cuda", dtype=torch.long, non_blocking=True)
batch = get_batch()

## INT8 hoá
names, params = convert_int8_mixed_precision(model)
def find_key(s):
    m = re.search(r'(.*block.*\.\d+\.)*(.*)', s)
    return "*" + m.group(2) if m.group(1) else m.group(2)
total_params = sum(p.numel() for p in model.parameters())
short_names = sorted(set(find_key(x) for x in names))
percent = (params/total_params)*100
print(f"""\nPHÂN CHIA PARAMS VÀO DTYPES:
* {len(names)} INT8 Mixeds {percent:.1f}% {params:,}
* {len(list(model.parameters())) - len(names)} BF16 Weights {100-percent:.1f}% {total_params - params:,}
INT8: {short_names}""")

#########################
##  Init Optimizer(s)  ##
#########################
class LRSchedule:
    def __init__(self, n_steps, decay_type="linear", warmup: float = 0.05, decay:  float = 0.15,):
        self.t1 = int(n_steps * warmup)
        self.t2 = int(n_steps * (1 - decay))
        self.t3 = n_steps
        self.decay_type = decay_type
        assert self.t1 <= self.t2

    def get_lr(self, init_lr: float, step: int) -> float:
        if step < 0 or step > self.t3: return 0.0
        if step < self.t1: return init_lr * step / self.t1
        if step < self.t2: return init_lr
        progress = (step - self.t2) / (self.t3 - self.t2)
        if self.decay_type == "linear": return init_lr * (1 - progress)
        return 0.5 * init_lr * (1 + math.cos(progress * math.pi)) # cosine

lr_schedule   = LRSchedule(args.steps, warmup=0.05, decay=0.15)
muon_params   = [p for n, p in model.named_parameters() if "proj" in n]

adam_params   = [
    dict(params=[*model.embeds.parameters(), *model.v_embs.parameters() ], lr=0.1   ), 
    dict(params=[ model.scalars                                         ], lr=0.015 ),
    dict(params=[*model.unembeds.parameters()                           ], lr=1/300 ),
]
adam_optim  = torch.optim.AdamW(adam_params, weight_decay=0.0, fused=True)  # eps=1e-10,
muon_optim  = Muon(muon_params, lr=0.025, momentum=0.95, weight_decay=0.01)

for opt in [muon_optim, adam_optim]:
    for group in opt.param_groups:
        group["init_lr"] = group["lr"]

##############
## TRANING  ##
##############
lossf = torch.compile(lossf)
# for x in model.blocks: x.compile()

model = model.cuda()
model.train()

print(f"\nCHUẨN BỊ HUẤN LUYỆN:\n* {tokens_per_batch//1024}k_tok_seq / step\n\n")
log_interval = 5 
logger = wandb.init(dir="/tmp", config=args,)

## end-of-text token là 6399 cho 6k, 8k vocab, và 31999 cho 32k vocab
eot = 6399 if args.vocab < 32000 else 31999

started_at = time.time()
for step in range(args.steps):  # training loop

    tokens, targets = batch[:-1], batch[1:]
    c, m = get_cu_max_seqlens_from(tokens, eot=eot)

    loss = lossf(model, tokens, targets, c, m)
    batch = get_batch() # async prefetch next batch
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(muon_params, max_norm=1.0) # ko grad norm head và embeddings
    # grad_norm = sum(p.grad.square().sum() for p in muon_params if p.grad is not None).item() ** 0.5

    if (step - 1) % log_interval == 0 or step == args.steps - 1:
        lossv = loss.item()
        muon_lr = muon_optim.param_groups[0]["lr"]
        log_dict = dict(loss=lossv, grad_norm=grad_norm, lr=muon_lr)

        logger.log(log_dict, step=step)
        pbar.set_postfix(loss=lossv, lr=muon_lr) # tối thiểu chiều rộng

    # set optimization hyperparameters
    for opt in [muon_optim, adam_optim]:
        for group in opt.param_groups:
            group["lr"] = lr_schedule.get_lr(group["init_lr"], step)
            if opt == muon_optim:
                frac = min(step / 50, 1) # momentum warmup for muon
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    muon_optim.step()
    adam_optim.step()
    model.zero_grad(set_to_none=True)
    
    if   step == 0:
        time0 = time.time() # cần time0 asap để tính tokens_per_second
        pbar = tqdm(total=args.steps, dynamic_ncols=True, disable=False)
    elif step == 1:
        print(f">>> First Step Took {int(time.time() - started_at)} Seconds <<<")
        time1 = time.time()
    elif step == 2:
        step_time = time.time() - time1
        time0 = time1 - step_time # tính đúng time0 theo step timing chuẩn
    pbar.update()

    if step % log_interval == 0:
        logger.log(dict(
            max_memory_allocated     = torch.cuda.max_memory_allocated(), 
            num_tokens_seen_millions = tokens_per_batch*step,
            tokens_per_second        = tokens_per_batch*step / (time.time() - time0),
        ), step=step)
        if step % (5 * log_interval) == 0:
            print(f"""         ATTN___ MLP___  ATTN___ MLP___\n{model.scalars.view(-1, 2)}""")
logger.finish()
