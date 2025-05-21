import math
import torch
from torch import Tensor, nn

@torch.no_grad()
def get_grad_norm(model: nn.Module):
    grad_norm_sq = sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)
    if hasattr(grad_norm_sq, "full_tensor"):
        grad_norm_sq = grad_norm_sq.full_tensor()
    return grad_norm_sq.item() ** 0.5

def print_model_stats(model: nn.Module):
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")

class LRSchedule:
    def __init__(
        self,
        lr: float,
        n_steps: int,
        warmup: float = 0.05, # 05% warmup đi từ 0 -> init_lr
        decay:  float = 0.15, # 80% stable @ init_lr, 15% decay to 0
        decay_type: str = "linear",
    ) -> None:
        self.lr = lr
        self.t1 = int(n_steps * warmup)
        self.t2 = int(n_steps * (1 - decay))
        self.t3 = n_steps
        self.decay_type = decay_type
        assert self.t1 <= self.t2
        assert decay_type in ("linear", "cosine")

    def get_lr(self, step: int) -> float:
        if step < 0 or step > self.t3: return 0.0

        if step < self.t1: return self.lr * step / self.t1
        if step < self.t2: return self.lr

        progress = (step - self.t2) / (self.t3 - self.t2)
        if self.decay_type == "linear": return self.lr * (1 - progress)
        elif self.decay_type == "cosine":
            return 0.5 * self.lr * (1 + math.cos(progress * math.pi))

    def set_lr(self, step: int, optim: torch.optim.Optimizer):
        lr = self.get_lr(step)
        for param_group in optim.param_groups:
            if isinstance(param_group["lr"], Tensor): param_group["lr"].fill_(lr)
            else: param_group["lr"] = lr
