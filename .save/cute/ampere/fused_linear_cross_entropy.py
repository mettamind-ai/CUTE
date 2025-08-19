import argparse, torch, time
from typing import Type
from torch import Tensor, nn

import cuda.bindings.driver as cuda
import cutlass, cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

# playground/685a53dca0c04c9413622e8c
@cute.kernel
def per_label_cross_entropy_kernel(
    logits: cute.Tensor,      # (num_targets, vocab)
    target: cute.Tensor,      # (num_targets,)
    loss:   cute.Tensor,      # (num_targets,)
    reduction:    cutlass.Float32,
    vocab:        cutlass.Int32,
    ignore_index: cutlass.Int32,
):
    # Lấy pid từ block index (mỗi block xử lý 1 target)
    pid = cute.arch.block_idx()[0]
    
    # Load target value
    tgt = target[pid]
    
    # Skip nếu target là ignore_index
    if tgt == ignore_index:
        return

