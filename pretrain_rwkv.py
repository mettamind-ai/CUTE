#!/usr/bin/env python3
from winrwkv_varlen import WinRWKVVarlen, fused_loss_fn_varlen as lossf
from optimus import convert_int8_ffn_only

import re, os, sys, types, argparse, json, time, math, torch, wandb, itertools, glob, numpy as np
import torch.distributed as dist, torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--steps", type=int, default=80_000)
parser.add_argument("--vocab", type=int, default=1024 * 50)
parser.add_argument("--dim", type=int, default=1024)
parser.add_argument("--layers", type=int, default=24)
parser.add_argument("--no_int8", action="store_true")

args = parser.parse_args()

tokens_per_step = args.bs * 1024
ctxlen = tokens_per_step
model = WinRWKVVarlen(vocab_size=args.vocab, n_layers=args.layers, dim=args.dim, ctxlen=ctxlen).cuda()

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

torch.manual_seed(1981)

# Load data

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def data_generator(filename_pattern: str, batch_size: int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    print(files)
    file_iter = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + batch_size:][: batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int32, non_blocking=True)
        pos = pos + batch_size
        yield inputs, targets


def get_cu_seqlens_from(input_seq, eot):
    mask = (input_seq == eot)
    mask[-1] = True
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=input_seq.device),
            torch.where(mask)[0].to(torch.int32) + 1,
        ]
    )
    return cu_seqlens


# EOT tokens
# 6k/8k vocab use 6399, 32k uses 31999, else 50256
if args.vocab < 32000:
    eot = 6399
elif args.vocab == 32000:
    eot = 31999
else:
    eot = 50256
print(f"end-of-text: {eot}")

train_loader = data_generator("data/fineweb-tokmon-10B/english-50256-balanced-v2/*train*.bin", tokens_per_step)
tokens, targets = next(train_loader)


if not args.no_int8:
    def find_key(s):
        m = re.search(r"(.*block.*\.\d+\.)*(.*)", s)
        return "*" + m.group(2) if m.group(1) else m.group(2)

    linear_names, linear_params, sparsable_names, sparsable_params = convert_int8_ffn_only(model)
    total_params = sum(p.numel() for p in model.parameters())

    linear_params_ = sum(x.weight.numel() for x in linear_params)
    linear_short_names = sorted(set(find_key(x) for x in linear_names))
    linear_percent = (linear_params_ / total_params) * 100

    sparsable_params_ = sum(x.weight.numel() for x in sparsable_params)
    sparsable_short_names = sorted(set(find_key(x) for x in sparsable_names))
    sparsable_percent = (sparsable_params_ / total_params) * 100

    print(
        f"""\nPARAM SPLIT BY DTYPES:
* {len(linear_names)} INT8Linear {linear_percent:.1f}% {linear_params_:,}
* {len(sparsable_names)} Sparsable {sparsable_percent:.1f}% {sparsable_params_:,}
* {len(list(model.parameters())) - len(linear_names) - len(sparsable_names)} Embedding {100 - linear_percent - sparsable_percent:.1f}% {total_params - linear_params_ - sparsable_params_:,}
INT8: {linear_short_names}
SPARSABLE: {sparsable_short_names}"""
    )


#########################
##  Init Optimizer(s)  ##
#########################
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002, fused=True)
for group in optimizer.param_groups:
    group["init_lr"] = group["lr"]


class LRSchedule:
    def __init__(self, n_steps, decay_type="cosine", warmup: float = 0.05, decay: float = 0.15):
        self.t1 = int(n_steps * warmup)
        if self.t1 < 200:
            self.t1 = 200
        if self.t1 > 1000:
            self.t1 = 1000
        self.t2 = int(n_steps * (1 - decay))
        self.t3 = n_steps
        self.decay_type = decay_type
        assert self.t1 <= self.t2

    def get_lr(self, init_lr: float, step: int) -> float:
        if step < 0 or step > self.t3:
            return 0.0
        if step < self.t1:
            return init_lr * step / self.t1
        if step < self.t2:
            return init_lr
        progress = (step - self.t2) / (self.t3 - self.t2)
        if self.decay_type == "linear":
            return init_lr * (1 - progress)
        return 0.5 * init_lr * (1 + math.cos(progress * math.pi))


lr_schedule = LRSchedule(args.steps, decay=0.15)

################
##  TRAINING  ##
################

torch._dynamo.config.patch(error_on_recompile=True)
lossf = torch.compile(lossf)
model.train()

save_dir = Path("runs/") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
save_dir.mkdir(parents=True, exist_ok=True)

print(f"\nPRETRAIN READY:\n* {tokens_per_step//1024}k_tok_seq / step\n")
logger = wandb.init(dir="/tmp", config=args)

for step in range(args.steps):
    started_at = time.time()
    model.zero_grad(set_to_none=True)

    cu_seqlens = get_cu_seqlens_from(tokens, eot=eot)
    loss = lossf(model, tokens, targets, cu_seqlens)
    tokens, targets = next(train_loader)
    loss.backward()

    for group in optimizer.param_groups:
        group["lr"] = lr_schedule.get_lr(group["init_lr"], step)

    optimizer.step()

    if step == 0:
        time0 = time.time()
        pbar = tqdm(total=args.steps, dynamic_ncols=True, disable=False)

    elif step == 1:
        print(f">>> First Step Took {int(time.time() - started_at)} Seconds <<<")
        time1 = time.time()

    elif step == 2:
        step_time = time.time() - time1
        time0 = time1 - step_time

    if step % 2 == 0:
        grad_norm = sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None).item() ** 0.5
        lr = optimizer.param_groups[0]["lr"]
        tokens_seen = tokens_per_step * step
        tokens_per_second_k = int(tokens_per_step / (time.time() - started_at)) / 1000
        lossv = loss.item()
        logger.log(
            dict(
                loss=lossv,
                lr=lr,
                grad_norm=grad_norm,
                tokens_per_second_k=tokens_per_second_k,
                step=step,
                tokens_seen=tokens_seen,
            )
        )
        pbar.set_description(
            f"loss {lossv:.4f} | lr {lr:.4g} | gnorm {grad_norm:.2f} | {tokens_per_second_k:.1f}k tok/s"
        )
        pbar.update(2)

    if step % 200 == 0 and step > 0:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        torch.save(ckpt, save_dir / f"ckpt_step_{step}.pt")
