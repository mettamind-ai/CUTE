import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from rwkv7 import RWKV

''' L2Wrap được sử dụng để tính đạo hàm theo phương pháp L2 regularization.
Cụ thể, L2Wrap thêm một chi phí bổ sung vào hàm mất mát (loss) hiện tại.
Công thức chi phí bổ sung được tính bằng cách lấy giá trị lớn nhất trong ma trận đầu vào y,
nhân với một hệ số nhỏ và gán giá trị này trở lại cho ma trận gy (khởi tạo zeros) có cùng kích thước như y.
Hệ số factor được sử dụng để giảm thiểu giá trị của chi phí bổ sung trong quá trình huấn luyện.
'''
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y) # khởi tạo gy có kích thước giống y và giá trị là 0
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


''' Racoon bọc bên ngoài RWKV
sử dụng lighting để quản lý việc huấn luyện RWKV và điều chỉnh tốc độ học (learning rate)
và phân tải dùng DeepSpeedCPUAdam khi cần
'''
class Racoon(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV(args)

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self.rwkv(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if ("att.w0" in n):
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')

        param_dict = {n: p for n, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

        if args.weight_decay > 0:
            optim_groups.append({
                "params": [param_dict[n] for n in lr_decay],
                "weight_decay": args.weight_decay,
                "my_lr_scale": 1.0,
            })
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, 
                    lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, 
                    bias_correction=True, adamw_mode=True, amsgrad=False,
                )
            return FusedAdam(optim_groups, 
                    lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, 
                    bias_correction=True, adam_w_mode=True, amsgrad=False,
            )
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, 
                    lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, 
                    bias_correction=True, adamw_mode=False, amsgrad=False, weight_decay=0,
                )
            return FusedAdam(optim_groups, 
                    lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, 
                    bias_correction=True, adam_w_mode=False, amsgrad=False, weight_decay=0,
            )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
