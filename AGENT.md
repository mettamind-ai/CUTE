# AGENT.md - CUTE Project Guidelines

## Commands
- **Run training**: `./pretrain.py --bs 2 --steps 1000` (default small model)
- **Run wingpt**: `./wingpt.py` 
- **Install deps**: `./run.sh` (installs torch, flash-attn, dependencies)
- **No tests found** - codebase uses direct execution and manual verification

## Code Style & Conventions
- **Language**: Python 3 with PyTorch, heavy GPU optimization focus
- **Imports**: Group stdlib, then torch, then local modules. Use `from torch import Tensor, nn`
- **Shebang**: Use `#!/usr/bin/env python3` for executable scripts
- **Type hints**: Use `Tensor` from torch, basic type hints on function signatures
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Comments**: Vietnamese comments allowed, concise English docstrings preferred
- **Error handling**: Minimal - relies on PyTorch's built-in error handling

## Architecture Notes
- This is a GPT training framework optimized for gaming GPUs (3090, 4090, 5090)
- Key components: WinGPT model, Muon optimizer, int8 mixed precision, flash attention
- Uses custom CUDA kernels and Triton for performance optimization
- Memory-mapped data loading from .bin files
- No traditional test suite - verification through training runs and loss curves
