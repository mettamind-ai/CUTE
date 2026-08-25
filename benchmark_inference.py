#!/usr/bin/env python3
'''Benchmark inference performance for all WinRWKV model sizes on GPU and CPU'''
import os, argparse, torch, time
from pathlib import Path
import csv

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.bfloat16)

from infer_winrwkv import load_checkpoint, forward_inference, autoregressive_inference

# Model size configs
MODEL_CONFIGS = {
    "S":   {"dim": 128, "layers": 6,   "ctxlen": 4096, "vocab": 4096},
    "M":   {"dim": 256, "layers": 6,   "ctxlen": 4096, "vocab": 4096},
    "L":   {"dim": 384, "layers": 12,  "ctxlen": 4096, "vocab": 4096},
    "XL":  {"dim": 512, "layers": 12,  "ctxlen": 8192, "vocab": 4096},
    "XXL": {"dim": 640, "layers": 12,  "ctxlen": 8192, "vocab": 4096},
}

def find_latest_checkpoint(model_size_dir):
    """Find the latest checkpoint in a directory"""
    checkpoints = list(Path(model_size_dir).glob("checkpoint_step_*.pth"))
    if not checkpoints:
        return None

    # Sort by step number
    def get_step(path):
        try:
            return int(path.stem.split('_')[-1])
        except:
            return 0

    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]

def benchmark_model(model_size, checkpoint_dir, device='cuda', forward_runs=10, autoregressive_tokens=50):
    """Benchmark a single model"""
    print(f"\n{'='*70}")
    print(f"Benchmarking {model_size} on {device.upper()}")
    print(f"{'='*70}")

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        print(f"  No checkpoint found in {checkpoint_dir}")
        return None

    print(f"  Checkpoint: {checkpoint_path}")

    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("  CUDA not available, skipping")
        return None

    try:
        # Load model
        model, config = load_checkpoint(checkpoint_path, device=device)

        # Forward pass benchmark
        print(f"\n  Forward Pass Benchmark:")
        seq_len = min(config['ctxlen'], 1024)  # Use reasonable sequence length
        if seq_len % 16 != 0:
            seq_len = (seq_len // 16) * 16

        input_ids = torch.randint(5, config['vocab_size'] // 4, (1, seq_len), dtype=torch.long)

        forward_tps = None
        forward_time = None
        try:
            forward_tps, forward_time, _ = forward_inference(model, input_ids, device, forward_runs)
            print(f"    Sequence length: {seq_len}")
            print(f"    Runs: {forward_runs}")
            print(f"    Time: {forward_time:.3f}s")
            print(f"    Throughput: {forward_tps:.0f} tokens/sec")
        except NotImplementedError as e:
            if 'CPU' in str(e):
                print(f"    Error: CPU inference not supported (model requires CUDA)")
            else:
                raise

        # Autoregressive benchmark
        print(f"\n  Autoregressive Generation Benchmark:")
        prompt_len = 16
        prompt = torch.randint(5, config['vocab_size'] // 4, (prompt_len,), dtype=torch.long)

        autoregressive_tps = None
        autoregressive_time = None
        try:
            generated, autoregressive_tps, autoregressive_time = autoregressive_inference(
                model, prompt, autoregressive_tokens, device=device
            )
            print(f"    Generated tokens: {autoregressive_tokens}")
            print(f"    Time: {autoregressive_time:.3f}s")
            print(f"    Speed: {autoregressive_tps:.2f} tokens/sec")
        except NotImplementedError as e:
            if 'CPU' in str(e):
                print(f"    Error: CPU inference not supported (model requires CUDA)")
            else:
                raise

        # Cleanup
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

        return {
            'model_size': model_size,
            'device': device,
            'forward_tokens_per_sec': forward_tps,
            'forward_time': forward_time,
            'autoregressive_tokens_per_sec': autoregressive_tps,
            'autoregressive_time': autoregressive_time,
            'config': config,
        }

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Benchmark WinRWKV inference')
    parser.add_argument("--checkpoint_dir", type=str, default="runs/winrwkv",
                       help="Base directory containing model checkpoints")
    parser.add_argument("--model_sizes", type=str, nargs='+', default=list(MODEL_CONFIGS.keys()),
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model sizes to benchmark")
    parser.add_argument("--devices", type=str, nargs='+', default=["cuda", "cpu"],
                       choices=["cuda", "cpu"],
                       help="Devices to benchmark on")
    parser.add_argument("--forward_runs", type=int, default=10,
                       help="Number of runs for forward pass benchmark")
    parser.add_argument("--autoregressive_tokens", type=int, default=50,
                       help="Number of tokens to generate for autoregressive benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                       help="Output CSV file for results")

    args = parser.parse_args()

    # Filter devices based on availability
    available_devices = []
    for device in args.devices:
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, skipping CUDA benchmarks")
            continue
        available_devices.append(device)

    if not available_devices:
        print("Error: No available devices")
        return

    print(f"\n{'='*70}")
    print("WinRWKV Inference Benchmark")
    print(f"{'='*70}")
    print(f"Model sizes: {', '.join(args.model_sizes)}")
    print(f"Devices: {', '.join(available_devices)}")
    print(f"Forward runs: {args.forward_runs}")
    print(f"Autoregressive tokens: {args.autoregressive_tokens}")
    print(f"{'='*70}")

    results = []

    # Benchmark each model size on each device
    for model_size in args.model_sizes:
        checkpoint_dir = Path(args.checkpoint_dir) / model_size

        for device in available_devices:
            result = benchmark_model(
                model_size, checkpoint_dir, device,
                args.forward_runs, args.autoregressive_tokens
            )
            if result:
                results.append(result)

    # Print summary table
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    # Forward pass table
    print("Forward Pass Throughput (tokens/sec):")
    print(f"{'Model':<8} {'GPU':>15} {'CPU':>15}")
    print("-" * 40)
    for size in args.model_sizes:
        gpu_result = next((r for r in results if r['model_size'] == size and r['device'] == 'cuda'), None)
        cpu_result = next((r for r in results if r['model_size'] == size and r['device'] == 'cpu'), None)

        gpu_tps = f"{gpu_result['forward_tokens_per_sec']:.0f}" if (gpu_result and gpu_result['forward_tokens_per_sec'] is not None) else "N/A"
        cpu_tps = f"{cpu_result['forward_tokens_per_sec']:.0f}" if (cpu_result and cpu_result['forward_tokens_per_sec'] is not None) else "N/A"

        print(f"{size:<8} {gpu_tps:>15} {cpu_tps:>15}")

    # Autoregressive table
    print(f"\nAutoregressive Generation Speed (tokens/sec):")
    print(f"{'Model':<8} {'GPU':>15} {'CPU':>15}")
    print("-" * 40)
    for size in args.model_sizes:
        gpu_result = next((r for r in results if r['model_size'] == size and r['device'] == 'cuda'), None)
        cpu_result = next((r for r in results if r['model_size'] == size and r['device'] == 'cpu'), None)

        gpu_tps = f"{gpu_result['autoregressive_tokens_per_sec']:.2f}" if (gpu_result and gpu_result['autoregressive_tokens_per_sec'] is not None) else "N/A"
        cpu_tps = f"{cpu_result['autoregressive_tokens_per_sec']:.2f}" if (cpu_result and cpu_result['autoregressive_tokens_per_sec'] is not None) else "N/A"

        print(f"{size:<8} {gpu_tps:>15} {cpu_tps:>15}")

    # Save to CSV
    if results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'model_size', 'device', 'forward_tokens_per_sec', 'forward_time',
                'autoregressive_tokens_per_sec', 'autoregressive_time',
                'dim', 'layers', 'vocab_size', 'ctxlen'
            ])
            writer.writeheader()
            for r in results:
                row = {
                    'model_size': r['model_size'],
                    'device': r['device'],
                    'forward_tokens_per_sec': r['forward_tokens_per_sec'],
                    'forward_time': r['forward_time'],
                    'autoregressive_tokens_per_sec': r['autoregressive_tokens_per_sec'],
                    'autoregressive_time': r['autoregressive_time'],
                    'dim': r['config']['dim'],
                    'layers': r['config']['n_layers'],
                    'vocab_size': r['config']['vocab_size'],
                    'ctxlen': r['config']['ctxlen'],
                }
                writer.writerow(row)

        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
