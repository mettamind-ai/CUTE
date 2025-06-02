- [x] 1 node multi GPUs training pipeline đơn giản, hiệu quả dựa trên pytorch
- [x] INT8 bench llama 1.2b trên 4090 => 1.5x so với bf16 + vram giảm 2G
- [x] loại bỏ grad_accum để tránh nhầm lẫn và đơn giản hoá pretrain options and code
- [x] loại bỏ token_old.py
- [x] flash-muon + adamw8bit
  - [x] Đã chạy đc với 1200m, 800m Llama models
  - [x] double check Llama params
- [x] samba
    - [x] Đọc paper và slides => [doc](https://docs.google.com/document/d/1ujo1P9DX6JV5PbD-WRZgfxYvwkmLr9btf5ywbH-OoNE/edit)
    - [x] Hiểu cách samba kết hợp với FSDP ([siêu đơn giản](/models/samba.md))
    - [x] Nâng cấp samba lên mamba2 => samba2
    - [x] Hiểu cách samba dùng flash-attn cho swa (sliding window attention), và liệu có thể đổi sang flex attention 
    - [x] Tìm hiểu `hymba`, `rwkvx` models xem có giúp cải tiến samba2? => `mambase`
- [x] ~~`Muon8bit` giảm từ 20.2 xuống 19.5G vram, không hội tụ tới expected loss~~
- [x] [OpOp - Optimizer Optimization](/OPTIM.md)
    - [x] Hiểu **(INT8) training**
    - [x] ~~activation = fuse optimizer step with backward~~
        - KHÔNG DÙNG vì bị conflict với torch.compile và ko support gradient accumulation & gradient clipping 
        - https://github.com/mettamind-ai/MAD/blob/main/OPTIM.md#optimize-giảm-vram-khi-training
  - [x] ~~Adam_mini, kém cả vram và hội tụ~~
- [x] [`LiWin` - `Li`near Attn + `Win`dow Attn](/LIWIN.md)
- [x] `Li`near dùng `rwkv7`
- [x] So sánh **flash-attn SWA** vs **flex attention SWA** => `flex attn Win`!
    - [x] `win/nsa_triton` sử dụng flash-attn SWA như thế nào?
    - [x] `win/nsa_pytorch` sử dụng flex attention như thế nào?
    - [x] `win/local_attention` sử dụng pytorch flex attention SWA như thế nào?
    - [x] code SWA = flash-attn (có vẻ dễ hơn?)
    - [x] code SWA = pytorch flex attention
        - https://pytorch.org/blog/flexattention/#sliding-window--causal
        ```python
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        from .sliding_window import generate_sliding_window
        sliding_window_mask_mod = generate_sliding_window(window_size=2048)
        block_mask = create_block_mask(sliding_window_mask_mod, 1, 1, S, S, device=device)
        out = flex_attention(query, key, value, block_mask=block_mask)
        ```
## ~~Best Linear (4k - 6k ctxln thua transformer độ)~~
- [x] rwkv7 nhiều vấn đề, dở nhất là tốn vram và chỉ ăn điện 50% ... ngoài ra int8 và muon cũng chưa xài đc
- [x] fla samba + mamba-ssm => hiệu suất thua 
  - cách flame train fla model, đặc biệt lấy loss
  - fla + optimus/muon.py", line 150 => "Input `d_in` must be a 2D tensor"

**LÝ DO THUẦN TRANSFORMER WIN**
- quá phổ biến nên độ hỗ trợ và ổn định rất cao
- quá flexible nên cách dùng, phối hợp, chế độ lại rất linh hoạt va dễ dàng
  - `packed dataset` để tránh attn chéo giữa các samples
  - `prefix`, `window`, `sparse` ...
- các `trainable sparse attention` (NSA) rất hiệu quả cho long context chả kém linear attn 

- [x] llama2 sử dụng flash-attn hoặc flex attention để so tốc độ với `int8`
  - [x] tìm xem HF's LlamaForCausalLM họ impl llama2 như thế nào?
  - => `int8` cho tốc độ gần x2 so với flash-attn???

## WinGPT
- [x] ~~flex attention: long short attn~~ (bỏ vì chậm hơn attn đơn thuần)
- [x] `modded 012...012` Value Embeddings & U-net like Connect
- [x] `modded` attn tricks
    - [x] 0-init projection layers
    - [x] QKNorm (RMSNorm cho Queries và Keys trước Attn)
    - [x] Increased Attention Scale
- [x] Prime & Modded's ReLU²
- [x] activation checkpointing
- [x] **Stable Embedding** <= Embedding state nên dùng float32
      https://huggingface.co/docs/bitsandbytes/v0.45.4/en/reference/nn/embeddings#bitsandbytes.nn.StableEmbedding
- [x] LigerEmbedding => ko works với bf16
- [x] ~~SwigLU. ko tốt hơn, params đội lên + chậm đi~~
- [x] 8k vocab
  - [x] chọn textbook data & train [BPEasy](https://github.com/gautierdag/bpeasy)
  - [x] Tạo numpy .bin datafile
- [x] **Revamp token dataset, đẩy data nhanh hết mức cho GPU**  
- Cần `toks/device/step` lớn! Bottle neck `loss_fn`. `cce` help a lot for bf16 but cannot run for int8
  - `8k vocab`, 0.6b model, int8 => 96kt/step; bf16+cce => 112kt/step
  - [x] int8 xài complied chunked loss
- [x] `bf16` sử dụng (fused) kernels tốt nhất cho 30xx GPUs
## nTP: n-future Token Prediction
- ~~DSv3 & MiMo: Fused heads + thêm block https://www.alphaxiv.org/abs/2505.07608. bỏ qua~~
- Paper https://www.alphaxiv.org/abs/2404.19737
- Notes https://github.com/mettamind-ai/MAD/blob/main/MULTI_TOKEN_PREDICTION.md
  - **Shared Transformer Trunk `fs`**: layers chung xử lý ngữ cảnh đầu vào `x₁:t` để tạo ra biểu diễn ẩn `z₁:t`
  - **n Independent Output Heads `fhᵢ`**: `n transformer layers` độc lập, mỗi layer dự đoán một token tương lai
  - **Shared Unembedding Matrix `fu`**: `lm_head` chung chuyển đổi từ vector biểu diễn sang logits
  - **CÔNG THỨC DỰ ĐOÁN**: `P₍(xₜ₊ᵢ|xₜ:₁) = softmax(fu(fhᵢ(fs(xₜ:₁))))` với `i = 1, 2, ..., n`

  - *Parallel Architecture*: Các output heads hoạt động độc lập, không chia sẻ thông tin với nhau
  - *Memory-Efficient*: Thực hiện forward/backward theo trình tự để giảm sử dụng bộ nhớ GPU từ O(nV + d) xuống O(V + d)

  - **Số lượng token dự đoán tối ưu**: `n=4` cho kết quả tốt nhất trên hầu hết các benchmark
  - **Độ dài ngữ cảnh**: 4096 tokens trong phần lớn các thí nghiệm
  - **Cấu trúc head**: Mỗi head là một transformer layer đầy đủ (không phải chỉ là linear layer)
  - **Loss function**: Cross-entropy loss tổng hợp trên tất cả n dự đoán
- Sample code https://github.com/facebookresearch/lingua/blob/main/apps/mtp/transformer.py#L83-L91
- Kinh nghiệm từ MiMo https://www.alphaxiv.org/abs/2505.07608
  - train thì dùng 1 MTP block
  - infer thì dup 1 thành 2, freeze main model & 1st MTP head, và finetune 2 heads vừa dup
  - MTP loss weight is set to 0.3 for the first 10.3T tokens, then reduced to 0.1 for the remainder of pre-training.

- [x] `Parallel Layers` [Primer](https://www.alphaxiv.org/abs/2109.08668) => chính là multi head attn?
- [x] `Conv Attn` [Baichuan M1 14b](https://www.alphaxiv.org/abs/2502.12671)
  ![](https://arxiv.org/html/2502.12671v2/extracted/6253923/images/kv_shift_attention.png) => giống DConv 3x1 trong primer
  ![](https://user-images.githubusercontent.com/544269/134764948-4aef8641-f9c5-43a5-9bfd-c2316df3a434.png)

- [x] fused params: val_embs, tok_embs và qk
  https://github.com/KellerJordan/modded-nanogpt/blob/master/records/120424_ValueEmbed/train_gpt2.py#L268
- [x] tinh gọn `optimus.py`
- [x] mix 2 local (rope) + 1 global (nope), 0,1,(2),3,4,(5) ...
- [x] bench các sdpa engines
- [x] áp dụng `flash_attn` cho SWA và `packed sequence` (mỗi sample chỉ attn chính nó)

- [x] seq packing without flash-attn <= nested tensor của pytorch chưa chín  (cần đợi thêm)
- [x] `Conv Attn` [Baichuan M1 14b](https://www.alphaxiv.org/abs/2502.12671) (chưa thấy hiệu quả)
- [x] thử nghiệm ý tưởng chỉ update gradients với tokens của active batches  (rất tốt khi dùng nhiều embeddings)
  - [x] thử nghiệm với `OhMaiEmbedding` hiệu quả
  - [x] ~~Tìm cách khuếch tán gradients ra các tokens không được load vào head~~ PHỨC TẠP => LÀM SAU!
    - Mỗi lần update gradients load ngẫu nhiên 1 số tokens vào head chẳng hạn ...
    - Đo lường sự giống và khác nhau giữa việc load full head và load part of head ...
    - => tìm ra quy luật + rút ra kinh nghiệm
- [x] thử lưu weight ở fp32 xem có giúp cải thiện loss? NO!, làm chậm đi
- [x] int8_mm trả về fp32 để tiện bf16 sr và lưu activations ở int8 + row_scale

- [x] ~~lưu `activations` (đầu ra của mỗi layer) ở INT8 + row_scale?~~ <= phức tạp, convert qua lại nhiều lần
  - không khả thi với row scale vì khi bwd input phải transpose
  - với tile scale thì kernel hiện tại đang chậm, ko hơn đc bf16 là mấy
- [x] ~~int4 mixed mm ko nhanh hơn mấy + vỡ loss~~
- Tích hợp qwen 3
  <img src="https://pbs.twimg.com/media/GsNBJ7VXEAAIErD?format=jpg" width="50%">

- [x] ~~smooth để giảm thiểu outliers => hadamard transform từ quest và qllmt~~ <= rất chậm
- [x] Fused Linear Chunked Cross Entropy Loss is the best! Tích hợp vào optimus.py
- [x] Mẹo tính mutiple exits loss từ torchtune
  - https://github.com/pytorch/torchtune/blob/main/torchtune/modules/early_exit_loss.py
  ```py
  # Stack tất cả hidden states: [e(xits), b(atch), s(seqlen), d(im)]
  hidden_states_stacked = torch.stack(hidden_states)
  # Tính logits một lần cho tất cả: [e, b, s, out_dim]
  logits_early = model.unembed(hidden_states_stacked)
  ```
- [x] OhMaiHead bản thử nghiẹm

