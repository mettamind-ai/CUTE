#!/usr/bin/env python3
import torch, math
import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset
from pathlib import Path

# datasets produced by tokenize_data.py
class TokenDataset(IterableDataset):
    def __init__(self, dataset_dir: str, seq_len: int, seed: int = 1981) -> None:
        super().__init__()
        # mỗi shard là 1 file .bin trong dataset_dir
        self.shards = sorted(Path(dataset_dir).glob("*.bin"))
        self.shards = [str(s) for s in self.shards]

        self.seq_len = seq_len
        self.seqlen_plus1 = seq_len + 1

        print(f"Found {len(self.shards)} shards of data")
        self._generator = torch.Generator().manual_seed(seed)

    # TODO: load and save state_dict
    def state_dict(self):
        # Lưu trạng thái cần thiết cho DataLoader.
        return dict(shards=self.shards, seq_len=self.seq_len, seed=self.seed)

    def _iter_shard(self, shard: Tensor):
        n_slices = math.floor(shard.shape[0] / self.seqlen_plus1)
        slice_indices = torch.randperm(n_slices, generator=self._generator)

        for slice_idx in reversed(slice_indices):
            batch = shard[
                 slice_idx      * self.seqlen_plus1 : 
                (slice_idx + 1) * self.seqlen_plus1
            ].long()
            yield batch[:-1], batch[1:]

    def __iter__(self):
        while True:
            # NOTE: we don't split data across workers. just depend on workers having different
            # random seeds to select different slice of data.
            shard_indices = torch.randperm(len(self.shards), generator=self._generator)

            for shard_idx in shard_indices:
                # divide a shard into n slices of (seq_len + 1)     # uint16: 0 -> 65k
                shard_np = np.memmap(self.shards[shard_idx], dtype=np.uint16, mode="r")
                shard = torch.from_numpy(shard_np.copy())

                # return sliced data one by one
                for data in self._iter_shard(shard): yield data

def get_dataset(type: str = "token", **kwargs):
    assert type == "token", "Support only token dataset"
    return TokenDataset(**kwargs)

#####################

import os, sys, types
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Bỏ compile sẽ khiến int8 chậm đi 1 nửa
COMPILE = True # compile hàm loss cuối là toàn bộ (trừ optim) được compiled

import argparse, json, time
from datetime import datetime
from pathlib import Path

import torch, wandb
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

torch.set_float32_matmul_precision('high') # enable fast bf16 mixed
torch.backends.cuda.matmul.allow_tf32  = True  

from tqdm import tqdm
from torch import Tensor, nn
from torchdata.stateful_dataloader import StatefulDataLoader

from train_utils import LRSchedule, print_model_stats, get_grad_norm
from optimus import Muon, convert_int8_mixed_precision

parser = argparse.ArgumentParser()
# Max throughput 12*6k=72k_toks/4090_llama1.2b
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=1024*4)
parser.add_argument("--activation_checkpoint", default=True)

parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--stop_loss", type=float, default=0)

parser.add_argument("--train_ds", type=json.loads, default='{"type":"token","dataset_dir":"."}')
parser.add_argument("--model_id", default=".save")
parser.add_argument("--int8", action="store_true")

parser.add_argument("--muon_lr", type=float, default=0.022) # default 0.02, modded gpt 0.025
parser.add_argument("--adam_lr", type=float, default=0.003) # 3e-4
parser.add_argument("--lr_schedule_kwargs", type=json.loads)

parser.add_argument("--seed", type=int, default=1981)
args = parser.parse_args()

is_dist = "RANK" in os.environ
rank = int(os.environ.get("RANK", 0))

is_master = (rank == 0)
def print0(msg): is_master and print(msg)

if not is_dist: world_size = 1
else:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    print0(f"Using distributed training with {world_size=} GPUs")

assert args.batch_size % world_size == 0
if args.seed is not None: torch.manual_seed(args.seed + rank)
args.torch_version = torch.__version__


#############################
## Init model for pretraining
#############################
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# Apply patch to enhance LLaMA model
if False and "llama" in args.model_id:
    print0("Applying WinGPT optimizations to LLaMA model...")
    from models.wingpt import init_linear, norm
    import transformers.models.llama.modeling_llama as modeling_llama
    from models.patch_modeling_llama import LlamaAttention, LlamaMLP
    # Replace with our optimized implementations
    modeling_llama.LlamaAttention = LlamaAttention
    modeling_llama.LlamaMLP = LlamaMLP
    print0("WinGPT optimizations applied: Q,K,V normalization, ReLU^2 MLP, and improved weight initialization")

cfg = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=args.model_id,
    max_position_embeddings=args.seq_len,
    use_cache=False,
)
model = AutoModelForCausalLM.from_config(cfg)

## Kiểm tra lần cuối
if is_master: print_model_stats(model)
for m in model.modules():
    if isinstance(m, (nn.Linear, nn.Embedding, LlamaRMSNorm)):
        m.bfloat16() # embeddings, linears & RMSNorm should be bf16

# keep RoPE cache in fp32
if "llama" in args.model_id:
    assert model.model.rotary_emb.inv_freq.dtype is torch.float32
    model.gradient_checkpointing_enable() # luôn bật cho llama
    print0(f"1st_LAYER {model.model.layers[0]}")
    print0(f'ATTN_IMPL eager')

if args.activation_checkpoint: 
    model.gradient_checkpointing_enable()

model.cuda() # to cuda, and read to use int8 mixed if needed
if args.int8:
    convert_int8_mixed_precision(model.model)

# Apply additional WinGPT optimizations after model is loaded
if "llama" in args.model_id:
    print0("Applying additional WinGPT optimizations...")
    
    # WinGPT optimizations have been applied at the class level

if is_dist:
    # use DDP => https://github.com/pytorch/pytorch/issues/104674
    # gradients all-reduce won't overlap with backward but speedup
    # thanks to full compiled graph outweighs comm overlap.
    if args.activation_checkpoint: torch._dynamo.config.optimize_ddp = False
    model = torch.nn.parallel.DistributedDataParallel(model)

#############################
## Init Optimizer(s)
#############################
weight_decay = 0.001 # std=0.01 (1e-2)

m = model.model
adam_n_params = {n: p for n, p in m.named_parameters() if p.ndim  < 2  or "emb"     in n}
muon_n_params = {n: p for n, p in m.named_parameters() if p.ndim >= 2 and "emb" not in n}

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
adam_params.append(model.lm_head.weight)

muon_params = list(muon_n_params.values())
for x in muon_params: assert x.ndim >= 2

adam_params_count = sum(p.numel() for p in adam_params)
muon_params_count = sum(p.numel() for p in muon_params)
total_params = sum(p.numel() for p in model.parameters())
muon_ratio = muon_params_count / total_params if total_params > 0 else 0
print0(f"""\nPHÂN CHIA PARAMS VÀO OPTIMIZERS: 
* Adam: {(1-muon_ratio)*100:.1f}% {adam_params_count:,}
* Muon: {   muon_ratio *100:.1f}% {muon_params_count:,}
 TOTAL: {               100:.1f}% {total_params:,}""")

# Dùng torch.optim.AdamW cho chuẩn xác + fused to save vram
adam_optim = torch.optim.AdamW(adam_params, lr=args.adam_lr, weight_decay=weight_decay, fused=True)
muon_optim = Muon(muon_params, lr=args.muon_lr, weight_decay=weight_decay, rank=rank, world_size=world_size)

_lr_schedule = args.lr_schedule_kwargs
if _lr_schedule is None: _lr_schedule = {}

muon_lr_schedule = LRSchedule(args.muon_lr, args.steps, **_lr_schedule)
adam_lr_schedule = LRSchedule(args.adam_lr, args.steps, **_lr_schedule)

#############################
## Data loader & checkpoint
#############################
bsize = args.batch_size // world_size
ds = get_dataset(seq_len=args.seq_len, seed=args.seed, **args.train_ds,)
dloader = StatefulDataLoader(
    ds, batch_size=bsize, num_workers=1, pin_memory=True, 
    # snapshot_every_steps=args.ckpt_interval,
)
dloader_iter = iter(dloader)

print0(f"""CHUẨN BỊ HUẤN LUYỆN
* is_dist {is_dist}, world_size {world_size}
* device_bs {bsize}, seq_len {args.seq_len}
* int8? {args.int8}, muon? {True}
* activation_checkpoint? {args.activation_checkpoint}
""")
model.train()
step = 0
log_interval = 10
lossf = 9999 # cần cho args.stop_loss

#############################
## Loss func and logger
#############################
if is_master:
    logger = wandb.init(
        dir="/tmp", config=args,
        # project=args.project, name=args.run_name,
    )

def loss_fn(model, tokens: Tensor, labels: Tensor):
    
    # Regular forward pass and loss calculation
    logits = model(tokens).logits.float()
    
    # Apply activation similar to WinGPT (line 220 in wingpt.py)
    scale_factor = 15.0
    logits = scale_factor * logits * torch.rsqrt(logits.square() + scale_factor * scale_factor)
    
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]), 
        labels.long().view(-1)
    )
if COMPILE: loss_fn = torch.compile(loss_fn)

#############################
## Training loop
#############################
while step < args.steps:
    tokens, labels = next(dloader_iter)
    loss = loss_fn(model, tokens.cuda(), labels.cuda())
    loss.backward()

    adam_lr_schedule.set_lr(step, adam_optim)
    muon_lr_schedule.set_lr(step, muon_optim)

    if (step - 1) % log_interval == 0 or step == args.steps - 1:
        if is_dist: dist.all_reduce(loss, dist.ReduceOp.AVG)
        grad_norm = get_grad_norm(model)

        lossf = loss.item()
        try: adam_lr = adam_optim.param_groups[0]["lr"].item()
        except: adam_lr = adam_optim.param_groups[0]["lr"]
        muon_lr = muon_optim.param_groups[0]["lr"]

        log_dict = dict(loss=lossf, grad_norm=grad_norm, muon_lr=muon_lr, adam_lr=adam_lr,)

        if is_master:
            logger.log(log_dict, step=step)
            pbar.set_postfix(loss=lossf, adam=adam_lr, muon=muon_lr)

    muon_optim.step(); muon_optim.zero_grad()
    adam_optim.step(); adam_optim.zero_grad()
 
    if step == 0: # sau khi compile và chạy model 1 lần 
        time0 = time.time() # record time0 và khởi tạo pbar để tính time cho chuẩn
        pbar = tqdm(total=args.steps, dynamic_ncols=True, disable=not is_master)

    step += 1
    pbar.update()

    if lossf <= args.stop_loss: break # stop training

    if step % log_interval == 0 and is_master:
        tokens_per_batch = args.batch_size * args.seq_len
        time1 = time.time()
        log_dict = dict(
            max_memory_allocated=torch.cuda.max_memory_allocated(),
            num_tokens_seen_millions=tokens_per_batch * step / 1e6,
            tokens_per_second=tokens_per_batch * log_interval / (time1 - time0),
        )
        time0 = time1
        logger.log(log_dict, step=step)
        if is_dist: dist.barrier()
        model.train()
# END training loop

if is_master: logger.finish()
if is_dist: dist.destroy_process_group()
