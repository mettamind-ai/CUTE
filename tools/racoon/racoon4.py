import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from rwkv4 import RWKV

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

    # Training step đổi để chạy với packed_dataset
    def training_step(self, batch, batch_idx=None):
        # input_ids = [[1,2,3][a,b,c]], targets = [[2,3,4],[b,c,d]]
        input_ids = batch[:, 0 : self.args.ctx_len].contiguous()
        targets = batch[:, 1 : self.args.ctx_len + 1].contiguous()
        logits = self.rwkv(input_ids) # => (B, T, vocab_size), targets => (B, T)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    # Training step gốc
    def training_step_original(self, batch, batch_idx=None):
        idx, targets = batch # idx = [[1,2,3][a,b,c]], targets = [[2,3,4],[b,c,d]]
        logits = self.rwkv(idx) # => (B, T, vocab_size), targets => (B, T)
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
        lr_3x = set()

        for name, p in self.named_parameters():
            not_a_scalar = ( len(p.squeeze().shape) >= 2 )
            if     "time_mix" in name and (args.layerwise_lr > 0): lr_1x.add(name) # time_mix params vào lr_x1
            elif "time_decay" in name and (args.layerwise_lr > 0): lr_2x.add(name) # time_decay vào lr_x2
            elif "time_first" in name and (args.layerwise_lr > 0): lr_3x.add(name) # time_first vào lr_x3
            elif      not_a_scalar and (args.weight_decay > 0): lr_decay.add(name) # weight_decay 1 hệ riêng
            else: lr_1x.add(name) # trong mọi trường hợp lr_1x luôn có phần tử

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        params = {name: param for name, param in self.named_parameters()}
        optim_groups = [{"params": [params[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if len(lr_2x) > 0:
            optim_groups += [{"params": [params[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0}]

        if len(lr_3x) > 0:
            optim_groups += [{"params": [params[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0}]

        if len(lr_decay) > 0:
            optim_groups += [{"params": [params[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, 
                    bias_correction=True, adamw_mode=True, amsgrad=False)
            else:
                return FusedAdam(
                    optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, 
                    bias_correction=True, adam_w_mode=True, amsgrad=False)

        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, 
                    bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            else:
                return FusedAdam(
                    optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, 
                    bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        else:
            return False
