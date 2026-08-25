#!/usr/bin/env python3
'''Inference script for WinRWKV - forward pass and autoregressive generation'''
import os, argparse, torch, torch.nn.functional as F
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.bfloat16)

from winrwkv import WinRWKV

def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif 'args' in checkpoint:
        args = checkpoint['args']
        if isinstance(args, dict):
            config = {
                'vocab_size': args.get('vocab', 4096),
                'n_layers': args.get('layers', 6),
                'dim': args.get('dim', 256),
                'ctxlen': args.get('ctxlen', 4096),
            }
        else:
            config = {
                'vocab_size': args.vocab,
                'n_layers': args.layers,
                'dim': args.dim,
                'ctxlen': args.ctxlen,
            }
    else:
        raise ValueError("Checkpoint missing model config")

    # Create model
    model = WinRWKV(
        vocab_size=config['vocab_size'],
        n_layers=config['n_layers'],
        dim=config['dim'],
        ctxlen=config['ctxlen']
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: {config['dim']} dim, {config['n_layers']} layers, vocab={config['vocab_size']}")
    return model, config

def forward_inference(model, input_ids, device='cuda', num_runs=10):
    """Forward pass inference - process full sequences"""
    model.eval()
    input_ids = input_ids.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids, return_logits=True)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            logits = model(input_ids, return_logits=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = input_ids.numel() * num_runs
    tokens_per_sec = total_tokens / elapsed

    return tokens_per_sec, elapsed, logits

def sample_logits(logits, temperature=1.0, top_k=50, top_p=0.9):
    """Sample from logits with temperature, top-k, and top-p"""
    logits = logits / temperature

    if top_k > 0:
        # Top-k filtering
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p < 1.0:
        # Top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def autoregressive_inference(model, prompt, max_tokens=100, temperature=1.0, top_k=50, top_p=0.9, device='cuda'):
    """Autoregressive generation - token by token"""
    model.eval()
    prompt = prompt.to(device) if isinstance(prompt, torch.Tensor) else torch.tensor(prompt, device=device)

    generated = prompt.clone()

    # Warmup
    with torch.no_grad():
        for _ in range(min(3, max_tokens)):
            if generated.shape[-1] % 16 != 0:
                # Pad to multiple of 16
                pad_len = 16 - (generated.shape[-1] % 16)
                padded = F.pad(generated, (0, pad_len), value=0)
            else:
                padded = generated
            _ = model(padded.unsqueeze(0), return_logits=True)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Generate
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_tokens):
            # Ensure sequence length is multiple of 16
            if generated.shape[-1] % 16 != 0:
                pad_len = 16 - (generated.shape[-1] % 16)
                padded = F.pad(generated, (0, pad_len), value=0)
            else:
                padded = generated

            # Forward pass
            logits = model(padded.unsqueeze(0), return_logits=True)

            # Get next token (last position)
            next_token_logits = logits[0, -1, :]
            next_token = sample_logits(next_token_logits.unsqueeze(0), temperature, top_k, top_p)

            # Ensure next_token is 1D tensor for concatenation
            # multinomial returns [batch, 1], we need [1] for concatenation
            next_token = next_token.view(-1)  # Flatten to 1D

            generated = torch.cat([generated, next_token], dim=-1)

    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_generated = max_tokens
    tokens_per_sec = tokens_generated / elapsed

    return generated, tokens_per_sec, elapsed

def main():
    parser = argparse.ArgumentParser(description='Inference with WinRWKV')
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--mode", type=str, default="both", choices=["forward", "autoregressive", "both"],
                       help="Inference mode")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (as comma-separated token IDs)")
    parser.add_argument("--prompt_len", type=int, default=100, help="Prompt length for forward pass")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens for autoregressive generation")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs for forward pass benchmark")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Load model
    model, config = load_checkpoint(args.checkpoint, device=args.device)

    print(f"\n{'='*70}")
    print(f"Inference Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Forward pass inference
    if args.mode in ["forward", "both"]:
        print("Forward Pass Inference:")
        print("-" * 70)

        # Create input
        if args.prompt:
            prompt_tokens = [int(x.strip()) for x in args.prompt.split(',')]
            input_ids = torch.tensor(prompt_tokens, dtype=torch.long)
            if len(input_ids) % 16 != 0:
                pad_len = 16 - (len(input_ids) % 16)
                input_ids = F.pad(input_ids, (0, pad_len), value=0)
        else:
            # Random input
            seq_len = args.prompt_len
            if seq_len % 16 != 0:
                seq_len = (seq_len // 16 + 1) * 16
            input_ids = torch.randint(5, config['vocab_size'] // 4, (seq_len,), dtype=torch.long)

        input_ids = input_ids.unsqueeze(0)  # Add batch dimension

        tokens_per_sec, elapsed, logits = forward_inference(model, input_ids, args.device, args.num_runs)

        print(f"  Sequence length: {input_ids.shape[1]}")
        print(f"  Runs: {args.num_runs}")
        print(f"  Total tokens: {input_ids.numel() * args.num_runs}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        print()

    # Autoregressive inference
    if args.mode in ["autoregressive", "both"]:
        print("Autoregressive Generation:")
        print("-" * 70)

        # Create prompt
        if args.prompt:
            prompt_tokens = [int(x.strip()) for x in args.prompt.split(',')]
            prompt = torch.tensor(prompt_tokens, dtype=torch.long)
        else:
            # Random prompt
            prompt_len = 16  # Start with small prompt
            prompt = torch.randint(5, config['vocab_size'] // 4, (prompt_len,), dtype=torch.long)

        generated, tokens_per_sec, elapsed = autoregressive_inference(
            model, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, args.device
        )

        print(f"  Prompt length: {len(prompt)}")
        print(f"  Generated tokens: {args.max_tokens}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Speed: {tokens_per_sec:.2f} tokens/sec")
        print(f"  Prompt tokens: {prompt.tolist()[:10]}..." if len(prompt) > 10 else f"  Prompt tokens: {prompt.tolist()}")
        print(f"  Generated tokens: {generated[len(prompt):len(prompt)+20].tolist()}..." if len(generated) > len(prompt) + 20 else f"  Generated tokens: {generated[len(prompt):].tolist()}")
        print()

if __name__ == "__main__":
    main()
