## Còn thời gian thì tiết tục train laws_symato_16k_refined
python3 train.py --load_model "../models_laws_symato_16k_refined_train_more/rwkv-12.pth" \
--proj_dir "../models_laws_symato_16k_refined_train_more" --wandb "laws_symato_16k_refined_train_more" \
--data_file "../laws/laws_symato_16k_refined" --tokenizer "symato" \
--ctx_len 1024 --micro_bsz 100 --n_layer 24 --n_embd 2048 \
--epoch_begin 0 --epoch_save 1 --epoch_count 20 --epoch_steps 500 \
--lr_init 1.5e-5 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
--warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_1 --grad_cp 1


## - - - ##
# Stage 1 có vẻ chậm hơn stage 2 nhưng tiết kiệm vram hơn
## - - - ##


## Train 3 lần, data_shift 0, 170, 340
# python3 train.py --load_model "../models_laws_symato_16k_refined/rwkv-3.pth" \
# --proj_dir "../models_laws_symato_16k_refined" --wandb "laws_symato_16k_refined" \
# --data_file "../laws/laws_symato_16k_refined" --tokenizer "symato" \
# --bigdata_stage 1 --bigdata_portion 0.33 --tokens_per_hour 137_000_000 --data_shift 0 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 \
# --epoch_begin 4 --epoch_save 1 \
# --lr_init 4.18e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

# python3 train.py --load_model "models_laws_symato_16k_refined/rwkv-7.pth" \
# --proj_dir "../models_laws_symato_16k_refined" --wandb "laws_symato_16k_refined" \
# --data_file "../laws/laws_symato_16k_refined" --tokenizer "symato" \
# --bigdata_stage 1 --bigdata_portion 0.33 --tokens_per_hour 137_000_000 --data_shift 170 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 \
# --epoch_begin 0 --epoch_save 1 \
# --lr_init 1.326e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

# python3 train.py --load_model "models_laws_symato_16k_refined/rwkv-7.pth" \
# --proj_dir "../models_laws_symato_16k_refined" --wandb "laws_symato_16k_refined" \
# --data_file "../laws/laws_symato_16k_refined" --tokenizer "symato" \
# --bigdata_stage 1 --bigdata_portion 0.33 --tokens_per_hour 137_000_000 --data_shift 340 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 \
# --epoch_begin 0 --epoch_save 1 \
# --lr_init 6e-5 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

## Note: chỉnh tokens_per_hour xuống còn 1/2 để phần dữ liệu cuối (test) bé đi 1/2 epoch
# python3 train.py --load_model "../models_laws_sp_16k/rwkv-7.pth" \
# --proj_dir "../models_laws_sp_16k" --wandb "laws_sp_16k" \
# --data_file "../laws/laws_text_document" --tokenizer "sentencepiece" \
# --vocab_file "../laws/sp_txt_16384.model" \
# --bigdata_stage 1 --bigdata_portion 0.33 --tokens_per_hour 137_000_000 --data_shift 0 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 \
# --epoch_begin 0 --epoch_save 1 \
# --lr_init 8e-5 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

# python3 train.py --load_model "../models_laws_sp_16k/rwkv-3.pth" --proj_dir "../models_laws_sp_16k" --wandb "laws_sp_16k" \
# --data_file "../laws/laws_text_document" --tokenizer "sentencepiece" --vocab_file "../laws/sp_txt_16384.model" \
# --bigdata_stage 1 --bigdata_portion 0.33 --tokens_per_hour 137_000_000 --data_shift 340 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 \
# --epoch_begin 4 --epoch_save 1 \
# --lr_init 8e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

## 1b2 symato16k fim
# python3 train.py --load_model "../models_laws_symato_16k/rwkv-3.pth" --proj_dir "../models_laws-fim_symato_16k" \
# --data_file "../laws_symato_16k_text_document" --tokenizer "symato" \
# --epoch_begin 0 --epoch_save 1 --bigdata_stage 1 --bigdata_portion 0.33 --tokens_per_hour 137_000_000 --data_shift 340 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 --fill_in_the_middle true \
# --lr_init 11e-5 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

## 1b2 symato3k fim
# python3 train.py --load_model "../models_laws_symato_2944/rwkv-2.pth" --proj_dir "../models_laws-fim_symato_2944" \
# --data_file "../60gb/_laws_symato_2944_text_document" --tokenizer "symato" \
# --epoch_begin 1 --epoch_save 1 --bigdata_stage 1 --bigdata_portion 0.33 --tokens_per_hour 277_000_000 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 --fill_in_the_middle true \
# --lr_init 6e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0
