COMMON_ARGS="--wandb rocket-the-raccoon \
--data_file ../data/tinystories-vi_train_aa.jsonl.xz,../data/tinystories-vi_train_ab.jsonl.xz \
--tokenizer ../_tokenmonster/vnonly_16k_consistent.vocab --ctx_len 512 \
--lr_init 5e-5 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
--warmup_steps 0 --accelerator gpu --strategy deepspeed_stage_2 --grad_cp 1 \
--devices 1 --epoch_begin 0 --epoch_save 1"

## wandb: ⭐️ View project at https://wandb.ai/tiendung/rocket-the-raccoon

## 305m params
# Epoch 1: : 2659it [1:12:03, loss=2.040, lr=4.91e-5, Kt/s=161.0]
# Epoch 2: :   13it [  00:21, loss=1.920, lr=4.91e-5, Kt/s=162.0]
python3 train.py --load_model "" $COMMON_ARGS \
--n_embd 1024 --n_layer 20 --micro_bsz 48 \
--proj_dir ./out/rwkv4-tinystories-vi-16k-ctx512-d1024-l20



##############################################################################
# Mô hình đủ lớn
# Các tham số bên dưới được thử nghiệm trên DGX A100 (4 x A100, 40G vram each)
##############################################################################

## 2.5b vs16k bs32 ctx1535_l28_d2560 cp1 =>  31Kt/s
# python3 train.py --load_model "" --proj_dir "../models_2b5" \
# --data_file "../100vi/shortnews_000_079_symato_16k_text_document" --tokenizer "sentencepiece" \
# \
# --epoch_begin 0 --epoch_save 1 \
# --bigdata_stage 1 --bigdata_portion 0.223 --tokens_per_hour 80_000_000 --data_shift 0 \
# --lr_init 6e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# \
# --ctx_len 768 --micro_bsz 64 --n_layer 28 --n_embd 2560 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 1


## 1.2b vs16k bs16  ctx448_l20_d2048 cp0 => 77Kt/s
## Tốc độ huấn luyện 77kt/s, 1h save 1 lần => 277m tokens save 1 lần
# python3 train.py --load_model "../models_laws_symato_2944/rwkv-4.pth" --proj_dir "../models_news_symato_2944/" \
# --data_file "../60gb/_laws_symato_2944_text_document" --tokenizer "symato" \
# --epoch_begin 0 --epoch_save 1 --bigdata_stage 1 --tokens_per_hour 277_000_000 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 \
# --lr_init 5e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0
