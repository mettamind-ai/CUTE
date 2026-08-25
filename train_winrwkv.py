#!/usr/bin/env python3
'''Training script for WinRWKV'''
import os, sys, argparse, time, math, torch, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import glob, itertools

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.bfloat16)

from winrwkv import WinRWKV, fused_loss_fn
from optimus import Muon1GPU as Muon, convert_int8_mixed_precision

# Model size presets matching benchmark.py
MODEL_CONFIGS = {
    "S":   {"dim": 128, "layers": 6,   "ctxlen": 4096, "vocab": 4096},
    "M":   {"dim": 256, "layers": 6,   "ctxlen": 4096, "vocab": 4096},
    "L":   {"dim": 384, "layers": 12,  "ctxlen": 4096, "vocab": 4096},
    "XL":  {"dim": 512, "layers": 12,  "ctxlen": 8192, "vocab": 4096},
    "XXL": {"dim": 640, "layers": 12,  "ctxlen": 8192, "vocab": 4096},
}

parser = argparse.ArgumentParser(description='Train WinRWKV model')
parser.add_argument("--model_size", type=str, default=None, choices=list(MODEL_CONFIGS.keys()) + [None],
                    help="Model size preset (S, M, L, XL, XXL). Overrides dim/layers/ctxlen/vocab if set.")
parser.add_argument("--bs", type=int, default=1, help="Batch size (sequences per batch)")
parser.add_argument("--ctxlen", type=int, default=4096, help="Context length (sequence length)")
parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
parser.add_argument("--vocab", type=int, default=4096, help="Vocabulary size")
parser.add_argument("--dim", type=int, default=256, help="Model dimension")
parser.add_argument("--layers", type=int, default=6, help="Number of layers")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--data_pattern", type=str, default=None, help="Data file pattern (e.g., 'data/fineweb-tokmon-10B/english-50256-balanced-v2/*train*.bin')")
parser.add_argument("--use_muon", action="store_true", help="Use Muon optimizer for projection layers")
parser.add_argument("--use_int8", action="store_true", help="Use INT8 mixed precision")
parser.add_argument("--save_dir", type=str, default="runs/winrwkv", help="Directory to save checkpoints")
parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

args = parser.parse_args()

# Apply model size preset if specified
if args.model_size:
    config = MODEL_CONFIGS[args.model_size]
    args.dim = config["dim"]
    args.layers = config["layers"]
    args.ctxlen = config["ctxlen"]
    args.vocab = config["vocab"]
    args.save_dir = f"runs/winrwkv/{args.model_size}"
    print(f"Using model size preset: {args.model_size}")

# Validate constraints
assert args.ctxlen % 16 == 0, f"ctxlen ({args.ctxlen}) must be divisible by 16 (CHUNK_LEN)"
assert args.dim % 64 == 0, f"dim ({args.dim}) must be divisible by 64 (HEAD_SIZE)"
assert args.layers >= 2, "n_layers must be >= 2"

print(f"""
{'='*70}
WinRWKV Training Configuration
{'='*70}
Batch size:      {args.bs}
Context length:  {args.ctxlen}
Vocabulary:      {args.vocab}
Model dim:       {args.dim}
Layers:          {args.layers}
Learning rate:   {args.lr}
Steps:           {args.steps}
Use Muon:        {args.use_muon}
Use INT8:        {args.use_int8}
{'='*70}
""")

# Initialize model
torch.manual_seed(1981)
model = WinRWKV(args.vocab, args.layers, args.dim, args.ctxlen).cuda()
print(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# Resume from checkpoint if specified
start_step = 0
checkpoint = None
if args.resume:
    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_step = checkpoint.get('step', 0)
    print(f"Resumed from step {start_step}")

# Apply INT8 if requested
if args.use_int8:
    print("Applying INT8 mixed precision...")
    convert_int8_mixed_precision(model)

# Setup optimizers
if args.use_muon:
    muon_params = [p for n, p in model.named_parameters() if "proj" in n]
    adam_params = [p for n, p in model.named_parameters() if "proj" not in n]
    muon_optim = Muon(muon_params, lr=args.lr * 5, momentum=0.95, weight_decay=0.008)
    adam_optim = torch.optim.AdamW(adam_params, lr=args.lr, weight_decay=0.002, fused=True)
    optimizers = [muon_optim, adam_optim]
    print(f"Using Muon optimizer: {len(muon_params)} param groups, {len(adam_params)} Adam groups")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    optimizers = [optimizer]
    print(f"Using AdamW optimizer")

# Load optimizer state if resuming
if args.resume and 'optimizer_state_dict' in checkpoint:
    if len(optimizers) == 1:
        optimizers[0].load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        if isinstance(checkpoint['optimizer_state_dict'], dict):
            if 'muon' in checkpoint['optimizer_state_dict']:
                optimizers[0].load_state_dict(checkpoint['optimizer_state_dict']['muon'])
            if 'adam' in checkpoint['optimizer_state_dict']:
                optimizers[1].load_state_dict(checkpoint['optimizer_state_dict']['adam'])

# Data loading
def load_data_shard(file: Path):
    """Load a data shard file (same format as pretrain.py)"""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "token count mismatch"
    return tokens

def data_generator(filename_pattern: str, batch_size: int, ctxlen: int):
    """Generate batches from data files"""
    if filename_pattern is None:
        # Use random data if no pattern provided
        print("No data pattern provided, using random data for testing")
        while True:
            input_seq = torch.randint(5, args.vocab // 4, (batch_size, ctxlen), dtype=torch.long).cuda()
            target = F.pad(input_seq[:, 1:], (0, 1), mode='constant', value=-100)
            yield input_seq, target

    files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
    if not files:
        print(f"Warning: No files found matching pattern: {filename_pattern}")
        print("Falling back to random data")
        while True:
            input_seq = torch.randint(5, args.vocab // 4, (batch_size, ctxlen), dtype=torch.long).cuda()
            target = F.pad(input_seq[:, 1:], (0, 1), mode='constant', value=-100)
            yield input_seq, target

    print(f"Found {len(files)} data files")
    file_iter = itertools.cycle(files)
    tokens, pos = load_data_shard(next(file_iter)), 0

    while True:
        tokens_needed = batch_size * ctxlen + 1
        if pos + tokens_needed >= len(tokens):
            tokens, pos = load_data_shard(next(file_iter)), 0

        buf = tokens[pos:pos + tokens_needed]
        input_seq = buf[:-1].view(batch_size, ctxlen).to(device="cuda", dtype=torch.long, non_blocking=True)
        target_seq = buf[1:].view(batch_size, ctxlen).to(device="cuda", dtype=torch.long, non_blocking=True)

        # Set first token of each sequence to -100 (ignore in loss)
        target_seq[:, 0] = -100

        pos += batch_size * ctxlen
        yield input_seq, target_seq

# Initialize data loader
train_loader = data_generator(args.data_pattern, args.bs, args.ctxlen)
input_seq, target = next(train_loader)

# Create save directory
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# Training loop
model.train()
print(f"\nStarting training...\n")

pbar = tqdm(total=args.steps, initial=start_step, dynamic_ncols=True)
start_time = time.time()

for step in range(start_step, args.steps):
    step_start = time.time()

    # Get next batch
    if step > 0:
        input_seq, target = next(train_loader)

    # Zero gradients
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    # Forward pass
    loss = fused_loss_fn(model, input_seq, target)

    # Backward pass
    loss.backward()

    # Optional: gradient clipping
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    for opt in optimizers:
        opt.step()

    # Logging
    if step % 10 == 0 or step == args.steps - 1:
        elapsed = time.time() - step_start
        tokens_per_sec = (args.bs * args.ctxlen) / elapsed if elapsed > 0 else 0
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'tok/s': f'{int(tokens_per_sec)}',
            'mem': f'{memory_mb:.0f}MB'
        })

    pbar.update()

    # Save checkpoint
    if (step + 1) % args.save_every == 0 or step == args.steps - 1:
        checkpoint_path = save_dir / f"checkpoint_step_{step+1}.pth"
        checkpoint_data = {
            'step': step + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizers[0].state_dict() if len(optimizers) == 1 else {
                'muon': optimizers[0].state_dict(),
                'adam': optimizers[1].state_dict()
            },
            'loss': loss.item(),
            'model_config': {
                'vocab_size': args.vocab,
                'n_layers': args.layers,
                'dim': args.dim,
                'ctxlen': args.ctxlen,
            },
            'args': vars(args),  # Save as dict for easier loading
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")

pbar.close()

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"Training completed!")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Average speed: {args.steps * args.bs * args.ctxlen / total_time:.0f} tokens/second")
print(f"Final loss: {loss.item():.4f}")
print(f"{'='*70}")
