#!/usr/bin/env python3
"""
Minimal end-to-end training loop for HNetForCausalLM.
- Loads config JSON (same format as hnet/2stage_S.json)
- Builds HNetForCausalLM (byte-level vocab 256)
- Trains in packed mode with fixed-length sequences per batch
- Loss = CrossEntropy (next token prediction) + router load-balancing loss
- Parameter groups: zero weight_decay for bias/norm via group_params

Example:
```sh
./hnet/pretrain.py --config hnet/2stage_S.json --steps 100 --batch_size 4 --seq_len 256
```
Note: This is a minimal trainer intended to run on a single GPU.
"""
import argparse, os, sys, json, math, time, random
from pathlib import Path
from typing import Iterator, List

import torch
import torch.nn.functional as F
from torch import nn

# Ensure repo root (parent of this folder) is on sys.path so we can import sibling package `flash`
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from hnet import HNetForCausalLM
from utils import HNetConfig, AttnConfig, SSMConfig
from train import load_balancing_loss, group_params


def read_all_text_files(path: str | Path) -> bytes:
    p = Path(path)
    if p.is_file():
        return p.read_bytes()
    # If directory, read all .txt files recursively
    chunks: List[bytes] = []
    for file in sorted(p.rglob("*.txt")):
        try: chunks.append(file.read_bytes())
        except: pass
    if not chunks: raise FileNotFoundError(f"No text files found at {p}")
    return b"\n".join(chunks)


def bytes_batch_stream(
    data_bytes: bytes,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    infinite: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Yields batches of shape (B, L) in [0,255] for byte-level LM.
    Targets are next-token shifted; the last position is ignored.
    """
    data = torch.tensor(list(data_bytes), dtype=torch.int64)
    if len(data) < (seq_len + 1) * batch_size:
        # Repeat data if too small
        reps = math.ceil(((seq_len + 1) * batch_size) / max(1, len(data)))
        data = data.repeat(reps)

    total_len = len(data)
    pos = 0
    while True:
        if pos + (seq_len + 1) * batch_size >= total_len:
            # Shuffle a random starting point to add stochasticity
            pos = random.randint(0, max(0, total_len - (seq_len + 1) * batch_size))
        x_list = []
        y_list = []
        for _ in range(batch_size):
            buf = data[pos : pos + seq_len + 1]
            pos += seq_len + 1
            x = buf[:-1]
            y = buf[1:]
            x_list.append(x)
            y_list.append(y)
        inputs = torch.stack(x_list).to(device=device, dtype=torch.long)
        targets = torch.stack(y_list).to(device=device, dtype=torch.long)
        # Make targets have same shape as inputs and ignore last column by padding -100 at the end
        # Here we already aligned by shifting; we can keep as is and just ignore nothing.
        # But to be consistent with common practice we can append an ignore token at last step;
        # However we already dropped last label by constructing y from x[1:], so it's fine.
        yield inputs, targets
        if not infinite:
            break


def build_model_from_config(config_path: str, device: torch.device, dtype: torch.dtype) -> HNetForCausalLM:
    with open(config_path, "r") as f: cfg_dict = json.load(f)
    attn_cfg = AttnConfig(**cfg_dict.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**cfg_dict.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**cfg_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to HNet JSON config", default=f"{current_dir}/2stage_S.json")
    parser.add_argument("--data_txt", type=str, default=None, help="Path to a .txt file or directory of .txt files. If omitted, synthetic data is used.")
    parser.add_argument("--outdir", type=str, default="runs/hnet_train")

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--router_N", type=float, default=3.0, help="Downsampling factor N used in load balancing loss")
    parser.add_argument("--router_loss_weight", type=float, default=0.01)

    parser.add_argument("--lr_multiplier", type=str, default=None, help="Comma-separated per-stage LR multipliers, e.g. '3.0,1.7,0.9'")
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument("--save_every", type=int, default=500)

    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data to host, construct stream (or synthesize if not provided)
    if args.data_txt is None:
        print("No --data_txt provided. Generating synthetic byte data ...")
        # Ensure enough tokens for training without excessive CPU<->GPU overhead
        num_tokens = (args.seq_len + 1) * args.batch_size * max(100, args.steps)
        raw_bytes = torch.randint(0, 256, (num_tokens,), dtype=torch.uint8).cpu().numpy().tobytes()
        print(f"Synthesized {len(raw_bytes):,} bytes.")
    else:
        print(f"Loading data from {args.data_txt} ...")
        raw_bytes = read_all_text_files(args.data_txt)
        print(f"Loaded {len(raw_bytes):,} bytes.")

    # Build model (always bfloat16)
    torch.set_default_dtype(torch.bfloat16)
    model = build_model_from_config(args.config, device=device, dtype=torch.bfloat16)

    # If desired, initialize weights specifically (optional)
    if hasattr(model, "backbone") and hasattr(model.backbone, "_init_weights"):
        print("Initializing model weights with custom initializer ...")
        model.backbone._init_weights(initializer_range=args.init_std)

    # Apply LR multipliers so group_params can attach _optim to all params
    if args.lr_multiplier is not None:
        lr_mults = [float(x) for x in args.lr_multiplier.split(",")]
    else:
        # Default: 1.0 for each stage
        n_stages = len(model.config.d_model)
        lr_mults = [1.0 for _ in range(n_stages)]
    if hasattr(model, "backbone") and hasattr(model.backbone, "_apply_lr_multiplier"):
        print(f"Applying per-stage LR multipliers: {lr_mults}")
        model.backbone._apply_lr_multiplier(lr_mults)

    # Build optimizer with parameter groups
    param_groups = group_params(model)
    # Apply actual per-group lr using lr_multiplier if present; and default weight_decay for groups without explicit override
    for g in param_groups:
        base_mult = g.get("lr_multiplier", 1.0)
        g["lr"] = args.lr * float(base_mult)
        if "weight_decay" not in g: g["weight_decay"] = args.weight_decay
    optim = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, fused=True if device.type == "cuda" else False)

    # Training loop
    model.train()
    use_amp = (device.type == "cuda")  # always bfloat16 under autocast on CUDA

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}")

    stream = bytes_batch_stream(raw_bytes, seq_len=args.seq_len, batch_size=args.batch_size, device=device)

    global_step = 0
    avg_loss = 0.0
    t0 = time.time()

    while global_step < args.steps:
        inputs, targets = next(stream)  # (B, L)
        # Packed mode: mask=None so model internally flattens and constructs cu_seqlens/max_seqlen
        with (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.enable_grad()):
            logits, bpred_outputs, _ = model(inputs, mask=None)
            # logits: (B, L, V)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            # Router loss across all levels
            if isinstance(bpred_outputs, list) and len(bpred_outputs) > 0:
                router_loss = 0.0
                for ro in bpred_outputs:
                    router_loss = router_loss + load_balancing_loss(ro, N=args.router_N)
            else:
                router_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            loss = ce + args.router_loss_weight * router_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        global_step += 1
        avg_loss = 0.98 * avg_loss + 0.02 * loss.item() if global_step > 1 else loss.item()

        if global_step % 10 == 0:
            dt = time.time() - t0
            tok_per_step = args.batch_size * args.seq_len
            kts = int(tok_per_step / max(1e-6, dt)) / 1000
            print(f"step {global_step:6d} | loss {loss.item():.4f} (avg {avg_loss:.4f}) | kTok/s {kts:.1f}")
            t0 = time.time()

        if args.save_every > 0 and global_step % args.save_every == 0:
            ckpt_path = Path(args.outdir) / f"model_step_{global_step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_ckpt = Path(args.outdir) / "model_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete. Saved final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
