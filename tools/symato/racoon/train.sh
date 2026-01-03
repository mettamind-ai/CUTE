# pip3 install --pre --force-reinstall deepspeed==0.7.0 torch==2.0.0.dev20230128 --extra-index-url https://download.pytorch.org/whl/nightly/cu117
# pip3 install -r ../requirements.txt
# python3 -m venv ~/venv/racoon
# source ~/venv/racoon/bin/activate

## 1.2b vs16k bs16  ctx448_l20_d2048 cp0 => 77Kt/s
## Tốc độ huấn luyện 77kt/s, 1h save 1 lần => 277m tokens save 1 lần
# python3 train.py --load_model "../models_laws_symato_2944/rwkv-4.pth" --proj_dir "../models_news_symato_2944/" \
# --data_file "../60gb/_laws_symato_2944_text_document" --tokenizer "symato" \
# --epoch_begin 0 --epoch_save 1 --bigdata_stage 1 --tokens_per_hour 277_000_000 \
# --ctx_len 512 --micro_bsz 24 --n_layer 20 --n_embd 2048 \
# --lr_init 5e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

## 1.2b vs16k bs16  ctx768_l20_d2048 cp0 => 77Kt/s => 6G laws gpt + fim
# python3 train.py --load_model "../models_laws_symato_2944/rwkv-2.pth" --proj_dir "../models_laws_symato_2944" \
# --data_file "../60gb/_laws_symato_2944_text_document" --tokenizer "symato" \
# --ctx_len 512 --epoch_steps 4000 --epoch_begin 0 --epoch_save 1 \
# --micro_bsz 24 --n_layer 20 --n_embd 2048 --fill_in_the_middle true \
# --lr_init 4.8e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

# python3 train.py --load_model "../models/rwkv-4.pth" --proj_dir "../models" \
# --data_file "../60gb/_laws_symato_2944_text_document" --tokenizer "symato" \
# --ctx_len 768 --epoch_steps 5000 --epoch_begin 0 --epoch_save 1 \
# --micro_bsz 16 --n_layer 20 --n_embd 2048 --fill_in_the_middle true \
# --lr_init 5e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 0

# - - -

## 2.5b vs16k bs32 ctx1535_l28_d2560 cp1 =>  31Kt/s  => 18GB
# python3 train.py --load_model "../models/rwkv-init.pth" --proj_dir "../models" \
# --data_file "../60gb/_laws_symato_2944_text_document" --tokenizer "symato" \
# --ctx_len 1536 --epoch_steps 1000 --epoch_begin 0 --epoch_save 1 \
# --micro_bsz 32 --n_layer 28 --n_embd 2560 --fill_in_the_middle true \
# --lr_init 5e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_2 --grad_cp 1

## Laws fill-in-the-middle, 200m params, 4 x A100 40G
# python3 train.py --load_model "../models/rwkv-1.pth" --proj_dir "../models" \
# --data_file "../laws/laws_text_document" --tokenizer "sentencepiece" --vocab_file "../laws/sp_txt_16384.model" \
# --ctx_len 256 --epoch_steps 8000 --epoch_begin 0 --epoch_save 1 \
# --micro_bsz 160 --n_layer 12 --n_embd 1024 --fill_in_the_middle true \
# --lr_init 4.9e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 4 --strategy deepspeed_stage_1 --grad_cp 0

# - - - - - - -

## fb_comments_200 50m params
# python3 train.py --load_model "" --proj_dir "../models" \
# --data_file "../fb_comments/fb_comments_200_text_document" --tokenizer "sentencepiece" \
# --vocab_file "../fb_comments/fb_comments_16384.model" \
# --ctx_len 384 --epoch_steps 2500 --epoch_begin 0 --epoch_save 1 \
# --micro_bsz 152 --n_layer 6 --n_embd 640 \
# --lr_init 8e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 1 --strategy deepspeed --grad_cp 1

## Laws fill-in-the-middle
# python3 train.py --load_model "../models/rwkv-12.pth" --proj_dir "../models" \
# --data_file "../laws/laws_text_document" --tokenizer "sentencepiece" --vocab_file "../laws/sp_txt_16384.model" \
# --ctx_len 256 --epoch_steps 2000 --epoch_begin 0 --epoch_save 1 \
# --micro_bsz 192 --n_layer 8 --n_embd 768 --fill_in_the_middle true \
# --lr_init 8e-5 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 1 --strategy deepspeed_stage_2 --grad_cp 1

## Laws 6gb, 200m params
# python3 train.py --load_model "../models/rwkv-54.pth" --proj_dir "../models" \
# --data_file "../laws/laws_text_document" --tokenizer "sentencepiece" --vocab_file "../laws/sp_txt_16384.model" \
# --ctx_len 512 --epoch_steps 1200 --epoch_begin 0 --epoch_save 1 \
# --micro_bsz 64 --n_layer 12 --n_embd 1024 \
# --lr_init 2.66e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 1 --strategy deepspeed --grad_cp 1


## Truyen dataset
# cd ../data; wget https://huggingface.co/tiendung/symato/resolve/main/truyen.binidx.7z

# Tối ưu hòa tính toàn bằng cách: 
# 1/ ctx_len = 512 để nạp đc nhiều dữ liệu loss giảm nhanh tới bão hòa
# 2/ ctx_len = 1024 để loss tiếp tục giảm nữa! (một kiểu tinh chỉnh nhưng mà với ctx_len)

# 1/
# python3 train.py --load_model "../models/rwkv-10.pth" --proj_dir "../models" \
# --data_file "../data/truyen_text_document" --tokenizer "symato" \
# --ctx_len 512 --epoch_steps 1200 --epoch_begin 0 --epoch_save 1 \
# --micro_bsz 320 --n_layer 6 --n_embd 512 \
# --lr_init 4.5e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
# --warmup_steps 0 --accelerator gpu --devices 1 --strategy deepspeed --grad_cp 1

# 2/
## Tốc độ huấn luyện 20kt/s, 1h (3600 s) save 1 lần => 72m tokens save 1 lần
python3 train.py --load_model "" --proj_dir "../models" \
--data_file "../data/truyen_text_document" --tokenizer "symato" \
--epoch_begin 0 --epoch_save 1 --bigdata_stage 1 --tokens_per_hour 72_000_000 --data_shift 0 \
--ctx_len 100 --micro_bsz 8 --n_layer 3 --n_embd 128 \
--lr_init 5e-4 --lr_final 1e-5 --beta1 0.95 --beta2 0.98 --adam_eps 1e-8 \
--warmup_steps 0 --accelerator gpu --devices 1 --strategy deepspeed_stage_2 --grad_cp 1
