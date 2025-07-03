#!/usr/bin/env python3
from wingpt import WinGPT, get_cu_max_seqlens_from, fused_loss_fn as lossf
from optimus import Muon1GPU as Muon, convert_int8_mixed_precision

import re, os, sys, types, argparse, json, time, math, torch, wandb, itertools, glob, numpy as np
import torch.distributed as dist, torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch import Tensor, nn
from bitsandbytes.optim import AdamW8bit

parser = argparse.ArgumentParser()
parser.add_argument("--bs",     type=int, default=None)
parser.add_argument("--steps",  type=int, default=20000)
parser.add_argument("--vocab",  type=int, default=1024*64)

args = parser.parse_args()
torch.manual_seed(1981)

## Config
if args.bs is None: args.bs = 64
tokens_per_batch =  args.bs*1024
cu_steps =  512 // args.bs # grad accum để đạt 1 triệu toks / step
model = WinGPT(dim=1024, n_layers=28, head_dim=64, vocab_size=args.vocab, ctxlen=tokens_per_batch)

## Load data, sooner better
def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def data_generator(filename_pattern: str, batch_size: int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]; print(files)
    file_iter = itertools.cycle(files) # iter(files); use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens): tokens, pos = _load_data_shard(next(file_iter)), 0
        buf     = tokens[pos + batch_size:][:batch_size + 1]
        inputs  = buf[  :-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1 :  ].to(device="cuda", dtype=torch.int32, non_blocking=True) # H2D in another stream isn't helpful.
        pos     = pos + batch_size
        yield inputs, targets

# end-of-text token là 6399 cho 6k, 8k vocab, và 31999 cho 32k vocab
eot = 6399 if args.vocab < 32000 else 31999 if args.vocab == 32000 else 50256; print(f"end-of-text: {eot}")
train_loader = data_generator("data/fineweb10B/fineweb_train_*.bin", tokens_per_batch)
tokens, targets = next(train_loader)

## INT8 hoá
names, params = convert_int8_mixed_precision(model)
def find_key(s):
    m = re.search(r'(.*block.*\.\d+\.)*(.*)', s)
    return "*" + m.group(2) if m.group(1) else m.group(2)
total_params = sum(p.numel() for p in model.parameters())
short_names = sorted(set(find_key(x) for x in names))
percent = (params/total_params)*100
print(f"""\nPHÂN CHIA PARAMS VÀO DTYPES:
* {len(names)} Linear {percent:.1f}% {params:,}
* {len(list(model.parameters())) - len(names)} Embeds {100-percent:.1f}% {total_params - params:,}
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

lr_schedule = LRSchedule(args.steps, warmup=0.05, decay=0.15)
muon_params = [p for n, p in model.named_parameters() if "proj" in n]
adam_params = [
    dict(params=model.embeds.parameters(),   lr=0.006 ), 
    dict(params=model.unembeds.parameters(), lr=0.003 ),
]
adam_optim  = torch.optim.AdamW(adam_params, fused=True)
muon_optim  = Muon(muon_params, lr=0.03, momentum=0.95, weight_decay=0.01)

for opt in [muon_optim, adam_optim]:
    for group in opt.param_groups:
        group["init_lr"] = group["lr"]

###############
##  TRANING  ##
###############
lossf = torch.compile(lossf)
model = model.cuda()
model.train()

print(f"\nCHUẨN BỊ HUẤN LUYỆN:\n* {tokens_per_batch//1024}k_tok_seq / step\n\n")
logger = wandb.init(dir="/tmp", config=args,)

total_docs = maxlen = tokens_seen = muon_lr = lossv = 0
for step in range(args.steps):  # training loop

    started_at = time.time()

    n_samples = lossv = 0
    for _ in range(cu_steps):
        cu_seqlens, max_seqlen = get_cu_max_seqlens_from(tokens, eot=eot)
        loss = lossf(model, tokens, targets, cu_seqlens, max_seqlen, cu_steps=cu_steps)
        tokens, targets = next(train_loader)
        loss.backward()
        lossv += loss.item()
        n_samples += len(cu_seqlens)
        if max_seqlen > maxlen: maxlen = max_seqlen

    total_docs += n_samples
    grad_norm = torch.nn.utils.clip_grad_norm_(muon_params, max_norm=1.0) # ko grad norm head và embeddings

    # set optimization hyperparameters
    for opt in [muon_optim, adam_optim]:
        for group in opt.param_groups:
            group["lr"] = lr_schedule.get_lr(group["init_lr"], step)
            if opt == muon_optim:
                frac = min(lr_schedule.t1,  1) # momentum warmup for muon
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

    muon_lr = muon_optim.param_groups[0]["lr"]
    tokens_seen = tokens_per_batch * step * cu_steps
    tokens_per_second_K = int(tokens_per_batch * cu_steps / (time.time() - started_at))/1000
    logger.log(dict(
        loss                 = lossv, 
        lr                   = muon_lr, 
        grad_norm            = grad_norm,
        max_memory_allocated = torch.cuda.max_memory_allocated(), 
        tokens_seen_M        = tokens_seen / 1e6,
        tokens_per_second_K  = tokens_per_second_K,
        n_samples            = n_samples,
        kmax                 = max_seqlen//1000,
    ), step=step)
    pbar.set_postfix(loss=lossv, kmax=max_seqlen//1000, kts=tokens_per_second_K)
    pbar.update()

logger.finish()
