# Rút gọn từ https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/train.py và 
# https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/trainer.py

import math, time, datetime, os, warnings, glob
import numpy as np

import torch, deepspeed
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from typing import Tuple, Optional
from argparse import ArgumentParser


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            total_steps = args.epoch_count * args.epoch_steps
            progress = (real_step - w_step + 1) / (total_steps - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))

            if trainer.global_step < w_step:
                lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        for param_group in trainer.optimizers[0].param_groups:
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        if trainer.global_step == 0:
            if trainer.is_global_zero: # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    import wandb; print("Login to wandb...")
                    wandb.init(project=args.wandb, config=args, save_code=False,
                        name=args.run_name + " " + args.my_timestamp,)
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        if trainer.is_global_zero: # logging
            t_now = time.time_ns()
            token_per_step = args.ctx_len * args.real_bsz
            real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0: lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))


    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size


    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        if trainer.is_global_zero:  # logging & save state_dict
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or \
                    trainer.current_epoch == args.epoch_count - 1:
                to_save_dict = pl_module.rwkv.state_dict()
                try:
                    torch.save( to_save_dict, \
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth")
                except Exception as e:
                    print('Error\n\n', e, '\n\n')
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()
            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

@rank_zero_only
def generate_init_weight(model, init_weight_name):

    if args.use_retnet:
        mm = model.generate_init_weight()
    else:
        mm = model.init_weight()

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)


########################################################################################################
# Kịch bản huấn luyện

rank_zero_info("########## work in progress ##########")
parser = ArgumentParser()
parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb

parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
parser.add_argument("--proj_dir", default="out", type=str)
parser.add_argument("--random_seed", default="-1", type=int)

parser.add_argument("--data_file", default="", type=str)
parser.add_argument("--tokenizer", default="utf-8", type=str)
parser.add_argument("--vocab_file", default="utf-8", type=str)
parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto

parser.add_argument("--ctx_len", default=1024, type=int)
parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

parser.add_argument("--micro_bsz", default=16, type=int)  # micro batch size (batch size per GPU)
parser.add_argument("--n_layer", default=6, type=int)
parser.add_argument("--n_embd", default=512, type=int)
parser.add_argument("--dim_att", default=0, type=int)


parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
parser.add_argument("--lr_final", default=1e-5, type=float)
parser.add_argument("--warmup_steps", default=0, type=int)  # try 50 if you load a model
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
parser.add_argument("--adam_eps", default=1e-8, type=float)

parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpoints: saves VRAM, but slower
parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough

# activate retnet official model implementation 
parser.add_argument("--use_retnet", default=False, action='store_true')
parser.add_argument("--retnet_official_name", default='retnet_base', type=str)

# nhân ma trận cách bướm
parser.add_argument("--use_monarch_mlp", default=False, action='store_true')
parser.add_argument("--monarch_mlp_nblocks", default=4, type=int)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

if args.random_seed >= 0:
    print(f"########## WARNING: GLOBAL SEED {args.random_seed} WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
    seed_everything(args.random_seed)

np.set_printoptions(precision=4, suppress=True, linewidth=200)
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

args.precision = "bf16" # mặc định sử dụng bf16 !!!
args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
args.enable_checkpointing = False
args.replace_sampler_ddp = False
args.logger = False
args.gradient_clip_val = 1.0
args.num_sanity_val_steps = 0
args.check_val_every_n_epoch = int(1e20)
args.log_every_n_steps = int(1e20)
args.max_epochs = -1  # continue forever
args.betas = (args.beta1, args.beta2)
args.real_bsz = int(args.devices) * args.micro_bsz
os.environ["RWKV_T_MAX"] = str(args.ctx_len)

if args.use_retnet:
    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} arch-{args.retnet_official_name}"
else:
    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"

if not os.path.exists(args.proj_dir):
    os.makedirs(args.proj_dir)

samples_per_epoch = args.epoch_steps * args.real_bsz
tokens_per_epoch = samples_per_epoch * args.ctx_len
rank_zero_info(f"""
############################################################################
#
# RWKV-4 {args.precision} on {args.devices} {args.accelerator.upper()}, bsz {args.devices}x{args.micro_bsz}={args.real_bsz}, {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.tokenizer}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}
# Found deepspeed {deepspeed.__version__}
# Found pytorch_lightning {pl.__version__}
#
############################################################################""")
rank_zero_info(str(vars(args)) + "\n")

if args.lr_final == 0 or args.lr_init == 0:
    rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

os.environ["RWKV_JIT_ON"] = "1"

if "deepspeed_stage_3" in args.strategy:
    os.environ["RWKV_JIT_ON"] = "0"

if args.use_monarch_mlp:
    print('!!!!! Ma trận cánh bướm không tương thích JIT, disabling it')
    os.environ["RWKV_JIT_ON"] = "0"

if args.grad_cp == 1:
    print('!!!!! Warning: Gradient Checkpointing requires JIT off, disabling it')
    os.environ["RWKV_JIT_ON"] = "0"

if args.dim_att <= 0:
    args.dim_att = args.n_embd


########################################################################################################
## Load training data

import sys; sys.path.append("../common")
from packed_dataset_utils import create_dataloader

# test_dataloader = create_dataloader( ... ) 
train_dataloader, vocab_size = create_dataloader(
    vocab_file=args.tokenizer,
    jsonl_files=args.data_file,
    batch_size=args.micro_bsz,
    block_size=args.ctx_len,
    shuffle=True,
    seed=8888,
)
# Đừng quên update args.vocab_size để init embedding cho đúng
args.vocab_size = vocab_size

########################################################################################################
## Start the trainer

if args.use_retnet:
    from rocket import get_retnet_model
    print("USING RETNET WRAPPER")
    model = get_retnet_model(args)
else:
    from racoon import Racoon
    model = Racoon(args)

if len(args.load_model) == 0: # shall we build the initial weights?
    args.load_model = f"{args.proj_dir}/rwkv-init.pth"
    if args.use_retnet:
        generate_init_weight(model, args.load_model)  # save initial weights
    else:
        generate_init_weight(model.rwkv, args.load_model)  # save initial weights
else:
    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try: 
        load_dict = torch.load(args.load_model, map_location="cpu")
        model.rwkv.load_state_dict(load_dict)
    except: rank_zero_info(f"Bad checkpoint {args.load_model}")

trainer = Trainer.from_argparse_args(args, callbacks=[train_callback(args)])

if args.strategy is not None and "deepspeed" in args.strategy:
    trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb*1000*1000
    trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb*1000*1000

torch.set_float32_matmul_precision('high')
trainer.fit(model, train_dataloader)
