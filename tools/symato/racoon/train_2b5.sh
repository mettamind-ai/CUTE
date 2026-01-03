
## 2.5b vs16k bs32 ctx1535_l28_d2560 cp1 =>  31Kt/s

python3 train.py --load_model "" --proj_dir "../models_2b5" \
--data_file "../100vi/shortnews_000_079_symato_16k_text_document" --tokenizer "symato" \
\
--epoch_begin 0 --epoch_save 1 \
--bigdata_stage 1 --bigdata_portion 0.223 --tokens_per_hour 80_000_000 --data_shift 0 \
--lr_init 6e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
\
--ctx_len 768 --micro_bsz 64 --n_layer 28 --n_embd 2560 \
--warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 1


# python3 train.py --load_model "" --proj_dir "../models_2b5" \
# --data_file "../100vi/news_030_137_symato_16k_text_document" --tokenizer "symato" \
# \
# --epoch_begin 0 --epoch_save 1 \
# --bigdata_stage 1 --bigdata_portion 0.197 --tokens_per_hour 80_000_000 --data_shift 0 \
# --lr_init ??e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# \
# --ctx_len 1536 --micro_bsz 32 --n_layer 28 --n_embd 2560 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 1
