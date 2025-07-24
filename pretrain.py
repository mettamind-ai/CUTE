#!/usr/bin/env python3
from wingpt import WinGPT, get_cu_max_seqlens_from, fused_loss_fn as lossf
from optimus import Muon1GPU as Muon, convert_int8_mixed_precision

import re, os, sys, types, argparse, json, time, math, torch, wandb, itertools, glob, numpy as np
import torch.distributed as dist, torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--bs",    type=int, default=64)
parser.add_argument("--steps", type=int, default=80_000)
parser.add_argument("--vocab", type=int, default=1024*50)
parser.add_argument("--sparse", type=bool, default=True)

args = parser.parse_args()
tokens_per_batch =  args.bs*1024
model = WinGPT(dim=1024, n_layers=24, vocab_size=args.vocab, ctxlen=tokens_per_batch).cuda()
# model = WinGPT(dim=1536, n_layers=15, vocab_size=args.vocab, ctxlen=tokens_per_batch).cuda()

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
torch.manual_seed(1981)

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
    file_iter = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens): tokens, pos = _load_data_shard(next(file_iter)), 0
        buf     = tokens[pos + batch_size:][:batch_size + 1]
        inputs  = buf[  :-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1 :  ].to(device="cuda", dtype=torch.int32, non_blocking=True)
        pos     = pos + batch_size
        yield inputs, targets

# end-of-text token là 6399 cho 6k, 8k vocab, và 31999 cho 32k vocab
eot = 6399 if args.vocab < 32000 else 31999 if args.vocab == 32000 else 50256; print(f"end-of-text: {eot}")
train_loader = data_generator("data/fineweb10B/fineweb_train_*.bin", tokens_per_batch)
tokens, targets = next(train_loader)

## INT8 hoá
def find_key(s):
    m = re.search(r'(.*block.*\.\d+\.)*(.*)', s)
    return "*" + m.group(2) if m.group(1) else m.group(2)

linear_names, linear_params, sparsable_names, sparsable_params = convert_int8_mixed_precision(model)
total_params = sum(p.numel() for p in model.parameters())

linear_params_ = sum(x.weight.numel() for x in linear_params)
linear_short_names = sorted(set(find_key(x) for x in linear_names))
linear_percent = (linear_params_/total_params)*100

sparsable_params_ = sum(x.weight.numel() for x in sparsable_params)
sparsable_short_names = sorted(set(find_key(x) for x in sparsable_names))
sparsable_percent = (sparsable_params_/total_params)*100

print(f"""\nPHÂN CHIA PARAMS VÀO DTYPES:
* {len(linear_names)} Linear {linear_percent:.1f}% {linear_params_:,}
* {len(sparsable_names)} Sparse {sparsable_percent:.1f}% {sparsable_params_:,}
* {len(list(model.parameters())) - len(linear_names) - len(sparsable_names)} Embeds {100 - linear_percent - sparsable_percent:.1f}% {total_params - linear_params_ - sparsable_params_:,}
INT8: {linear_short_names}
SPARSE: {sparsable_short_names}""")

#########################
##  Init Optimizer(s)  ##
#########################
muon_params = [p for n, p in model.named_parameters() if "proj" in n]
embedding_params = [p for n, p in model.named_parameters() if "proj" not in n]
adam_params = [
    dict(params=model.embeds.parameters(),   lr=0.003),
    dict(params=model.unembeds.parameters(), lr=0.002),
]
adam_optim = torch.optim.AdamW(adam_params, betas=(0.8, 0.95), weight_decay=0, fused=True)
muon_optim = Muon(muon_params, lr=0.01, momentum=0.95, weight_decay=0.008)

for opt in [muon_optim, adam_optim]:
    for group in opt.param_groups: group["init_lr"] = group["lr"]

class LRSchedule:
    def __init__(self, n_steps, decay_type="cosine", warmup: float = 0.05, decay:  float = 0.15,):
        self.t1 = int(n_steps * warmup)
        if self.t1 <  200: self.t1 =  200  # min warmup steps
        if self.t1 > 1000: self.t1 = 1000  # max warmup steps
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

lr_schedule = LRSchedule(args.steps, decay=0.15)

################
##  TRAINING  ##
################
torch._dynamo.config.patch(error_on_recompile=True)
lossf = torch.compile(lossf)#, fullgraph=True)
model.train()

save_dir = Path("runs/") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
save_dir.mkdir(parents=True, exist_ok=True)

print(f"\nCHUẨN BỊ HUẤN LUYỆN:\n* {tokens_per_batch//1024}k_tok_seq / step\n\n")
logger = wandb.init(dir="/tmp", config=args,)

for step in range(args.steps):  # training loop
    started_at = time.time()
    model.zero_grad(set_to_none=True)

    cu_seqlens, max_seqlen = get_cu_max_seqlens_from(tokens, eot=eot)
    loss = lossf(model, tokens, targets, cu_seqlens, max_seqlen, cu_steps=1)
    tokens, targets = next(train_loader)
    loss.backward()

    # set optimization hyperparameters
    frac = min(step / lr_schedule.t1, 1)
    for opt in [muon_optim, adam_optim]:
        for group in opt.param_groups:
            group["lr"] = lr_schedule.get_lr(group["init_lr"], step)
            if opt == muon_optim:  # muon momentum warmup
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    # grad_norm = torch.nn.utils.clip_grad_norm_(muon_params, max_norm=1.0)
    muon_optim.step()
    adam_optim.step()
    
    if   step == 0:
        time0 = time.time() # cần time0 asap để tính tokens_per_second
        pbar = tqdm(total=args.steps, dynamic_ncols=True, disable=False)

    elif step == 1:
        print(f">>> First Step Took {int(time.time() - started_at)} Seconds <<<")
        time1 = time.time()

    elif step == 2:
        step_time = time.time() - time1
        time0 = time1 - step_time # tính đúng time0 theo step timing chuẩn

    if args.sparse and step == 2 * lr_schedule.t1:
        # Applies int8 dnynamic symmetric per-token activation and int8 per-channel weigh quantization + 2:4 sparsity
        from torchao.quantization.quant_api import quantize_, Int8DynamicActivationInt8WeightConfig
        from torchao.dtypes import SemiSparseLayout
        for m in sparsable_params: quantize_(m, Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()))
        # muon_optim.reset_momentum(shape=sparsable_params[0].weight.shape)

    if step % 2 == 0:
        linear_grad_norm = sum(p.weight.grad.square().sum() for p in linear_params).item() ** 0.5
        sparsable_grad_norm = sum(p.weight.grad.square().sum() for p in sparsable_params).item() ** 0.5
        embedding_grad_norm = sum(p.weight.grad.square().sum() for p in embedding_params).item() ** 0.5

        muon_lr = muon_optim.param_groups[0]["lr"]
        tokens_seen = tokens_per_batch * step
        tokens_per_second_K = int(tokens_per_batch / (time.time() - started_at))/1000
        lossv = loss.item()
        logger.log(dict(
            loss                 = lossv, 
            lr                   = muon_lr, 
            linear_grad_norm     = linear_grad_norm,
            sparsable_grad_norm  = sparsable_grad_norm,
            embedding_grad_norm  = embedding_grad_norm,
            max_memory_allocated = torch.cuda.max_memory_allocated(), 
            tokens_seen_M        = tokens_seen / 1e6,
            tokens_per_second_K  = tokens_per_second_K,
            n_samples            = len(cu_seqlens),
            kmax                 = max_seqlen//1000,
        ), step=step)
        pbar.set_postfix(loss=lossv, kmax=max_seqlen//1000, kts=tokens_per_second_K)
    pbar.update()

    '''
    if (step + 1) % 300 == 0 or step == args.steps - 1:
        args.current_step = step
        torch.save(args, save_dir / "hyperparams.pth")
        torch.save(model     .state_dict(), save_dir / "model.pth")
        torch.save(muon_optim.state_dict(), save_dir / "muon_optim.pth")
        torch.save(adam_optim.state_dict(), save_dir / "adam_optim.pth")
    # '''
logger.finish()
model.unembeds.update_async_weight()
