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
- [x] Hoàn thiện OhMaiHead (dừng ở bản minimal)


20250603
--------
## Others
- Canon https://github.com/fla-org/flash-linear-attention/pull/388
- Gluon https://www.alphaxiv.org/abs/2505.13416
- Scion https://github.com/LIONS-EPFL/scion
- GLM / UL2 learning objectives và multi purpose models
- Học cách thu nhỏ model và NAS
  - https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1
  - NAS https://arxiv.org/abs/2411.19146
  - Llama-Nemotron https://arxiv.org/abs/2505.00949
- Cross-Layer Attention (CLA) -  sharing key and value heads between adjacent layers - https://arxiv.org/abs/2405.12981
- LIMe https://www.alphaxiv.org/abs/2502.09245 | https://github.com/corl-team/lime
  - giải quyết vấn đề representation collapse trong Transformers
  - Thảm hoạ bộ nhớ vì phải cache lại kv_sate của mọi layers
  ![](https://pbs.twimg.com/media/Gtdn_yWbgAAZAHB?format=jpg)
- tìm hiểu cách torch.compile tối ưu và fuse các phép toán ...

--------

- [x] Tìm hiểu byte level LLM (BLT, EveByte)
- [x] Biên dịch nhanh flash-attn-2 on-the-fly
- [x] Thử block sparse attn (infllmv2 và NSA)

- [x] Revisit `Primer-EZ = Squared ReLu + Depthwise Conv 3x1`, và lược bớt attn
  - https://www.alphaxiv.org/overview/2109.08668
  ![](https://paper-assets.alphaxiv.org/figures/2109.08668/img-4.jpeg)
  ![](https://paper-assets.alphaxiv.org/figures/2109.08668/img-3.jpeg)
  - `depthwise convolution 3x1`: tích chập theo chiều sâu, xử lý từng kênh độc lập

- [x] Full / LoRA một hoặc vài `attn -> mlp ...` cuối cho MTP
- [x] ~~Adaptive Softmax~~ https://docs.pytorch.org/docs/stable/_modules/torch/nn/modules/adaptive.html (xấp xỉ, old tech)

20250616
--------

- SPARSE
  - NSA, block/sparse attn nói chung (moba, mosa ...)
  - https://github.com/microsoft/SeerAttention
    - SeerAttention-R không hiệu quả với chuỗi ngắn 1k tokens. Framework này được tối ưu cho các tác vụ suy luận dài như AIME (trung bình 11k-18k tokens) và chỉ phát huy tác dụng khi sequence length đủ lớn để overhead của sparse attention được bù đắp bởi lợi ích tính toán.
    - NSA vs Seer: Cả hai phương pháp đều hoạt động ở mức độ block-level để tận dụng hiệu quả tính toán trên GPU hiện đại. Như NSA đề cập: "Blockwise selection is crucial to achieve efficient computation on modern GPUs" và SeerAttention cũng nhấn mạnh "we focus on learning block sparsity, which can seamlessly integrate with the tiling computation scheme of FlashAttention."
    - Cả hai đều sử dụng cơ chế scoring để chọn blocks quan trọng. SeerAttention tạo gating scores thông qua AttnGate, trong khi NSA tính importance scores cho từng block: "We retain tokens within the top-n sparse blocks ranked by block importance scores."
    - Cả hai đều có thể sử dụng TopK selection để tạo binary mask cuối cùng. SeerAttention mô tả: "users can adjust the TopK ratio or threshold at test time to achieve various trade-offs," tương tự NSA với "Top-n Block Selection."
    - Phương pháp training khác biệt. SeerAttention sử dụng self-distillation với "2DMaxPooled attention map from full attention as ground truth," trong khi NSA được pretrain end-to-end như một architecture hoàn chỉnh: "We enable end-to-end training, reducing pretraining computation without sacrificing model performance."


20250623
--------
- [x] chậm, chưa gain? <= ~~Canon https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/convolution.py~~
- [x] ~~stablemax~~ hoặc fp32 unembeddings + fp64 softmax, ortho optim đã có muon

20250703
--------
- Loss spike handling
  - https://www.alphaxiv.org/abs/2502.17055 Stable-SPAM grad norm & clipping for 4-bit training, **can apply for bf16**
    - (1) adaptively updates the clipping threshold
    - (2) normalizes the gradient matrix (giống phép trực giao?)
    - (3) momentum reset
  - https://www.alphaxiv.org/abs/2312.16903 scaled embed, **small sub layers + large residuals**
  - https://www.alphaxiv.org/abs/2410.16682 ngoài pre LN, cho thêm qk norm và softcap=50
  ```js
  Để ổn định training cần:
    1/ khởi tạo weight hợp lý + scaled embed hoặc embed norm. (Thần chú: small sub layers + large residuals)
    2/ pre LN + QK norm + softcap (phần bất ổn chủ yếu ở attn outliers)
    3/ tăng dần seqlen  + suitable batch size + better warmup + careful learning rate schedule
    4/ spec norm + auxilary loss + grad norm and clipping
    5/ spec clipping cho cả weight https://leloykun.github.io/ponder/spectral-clipping
  ```
  - Small sub-layers nghĩa là khởi tạo các tham số trong Transformer với giá trị rất nhỏ. `std_base = sqrt(2 / (5 * dim))`

20250705
--------
Huấn luyện model lớn:
- https://www.youtube.com/watch?v=__eeLqSlZ0w
- https://wandb.ai/craiyon/report/reports/Recipe-Training-Large-Models--VmlldzozNjc4MzQz
- https://www.jeremyjordan.me/distributed-training
- muP ...
- Multi step learning rate https://x.com/spiraldalat/status/1941661273990758471

- ~~MoD: Mixture of Depth~~ chưa tổng quát hoá bằng MoE
  - https://github.com/sramshetty/mixture-of-depths
    ![](https://graphcore-research.github.io/assets/images/posts/2024-04/potm/mixture-of-depths/mixture-of-depths-schematic.png)
  - https://www.alphaxiv.org/abs/2412.04449 p-MoD chỉ áp dụng cho visual tokens
    ![](https://github.com/MCG-NJU/p-MoD/raw/main/img/p-mod.png)
  - https://www.alphaxiv.org/abs/2412.20875 a-MoD dùng attn score để routing, tập trung ViT, bi-directional


20250705
--------
- [ ] Tìm vocab cân bằng https://arxiv.org/abs/2402.18376
- [ ] fast inference + hiệu chỉnh logits + sửa chữa tích luỹ sai lệch + phát hiện token "bất thường"
  - [ ] Modern LLM sampling https://rentry.org/samplers
  - https://pytorch.org/blog/accelerating-generative-ai-2
  - hiệu chỉnh logits top-nơ https://www.alphaxiv.org/abs/2411.07641
  - sửa chữa tích luỹ sai lệch https://www.alphaxiv.org/abs/2410.14655
    - Batch-scheduled Sampling (BASH), ngẫu nhiên kết hợp token từ dữ liệu gốc với token do mô hình tự sinh, giúp mô hình làm quen với việc xử lý các token không hoàn hảo trong quá trình huấn luyện.
    - Reference-Answer-based Correction (RAC), tích hợp khả năng tự sửa lỗi vào mô hình bằng cách dạy nó cách điều chỉnh những token sai lệch dựa trên ngữ cảnh tham chiếu.
  - Tránh tokens "bất thường" trong prompt https://www.alphaxiv.org/abs/2504.01002
    ```
    Hãy tưởng tượng token embeddings như một bản đồ 3D, token là một điểm trên bản đồ này. 
    Trong một bản đồ "bình thường", địa hình sẽ tương đối mượt mà - không có vách đá dựng đứng hay hố sâu bất ngờ.
    Token "bất thường" là những điểm có địa hình kỳ lạ - đỉnh núi nhọn hoắt, hố sâu, hoặc vách đá dựng đứng.

    => Xác định tokens bất thường: Với mỗi token, ta vẽ những vòng tròn có bán kính tăng dần xung quanh nó,
       rồi đếm xem có bao nhiêu token khác nằm trong mỗi vòng tròn ...

    Ví dụ: Khi họ vẽ bản đồ 3D của không gian xung quanh token "ember" (than hồng), 
    họ phát hiện ra nó nằm ở một vị trí rất kỳ lạ - 
    giống như một "đỉnh núi nhọn" hay "mũi nhọn" nhô ra khỏi bề mặt bình thường.
    Điều này khiến model khó "di chuyển" một cách mượt mà từ "ember" sang các từ khác.

    Irregularities lan truyền vì:
    - Residual connections bảo tồn lỗi gốc
    - Attention mechanism khuếch đại sự bất ổn
    - Geometric properties được giữ nguyên qua các layers
    - Context không thể "chữa lành" được structural problems
    - Accumulation effect làm vấn đề nghiêm trọng hơn theo thời gian
    ```

- [ ] Tìm độ đo ImportantCE, để đo độ quan trọng của các token trong input seq. How?
  - [ ] Sử dụng điểm attention từ Long vs Short SWA? Có tương đồng LongCE?
  - [ ] **Dùng MTP loss để làm weight cho NTP**
  - https://www.alphaxiv.org/abs/2405.03869 đánh giá ảnh hưởng từng mẫu dữ liệu tới hiệu suất mô hình
  - https://www.alphaxiv.org/abs/2505.19653 TI-DPO gradient-based token-importance weights
  - https://www.alphaxiv.org/abs/2003.11963 Token Loss Dynamic Reweighting (TLDR)
  - https://www.alphaxiv.org/abs/2407.10114 TokenSHAP đánh giá tầm quan trọng của từng token hoặc chuỗi con trong đầu vào

- Linear Attn
  - https://sustcsonglin.github.io/blog/2024/deltanet-1
  - https://people.csail.mit.edu/yoonkim/data/efficient_architectures_talk.pdf
  - https://leloykun.github.io/ponder/test-time-regression
  - https://goombalab.github.io/blog/2025/tradeoffs

- [ ] Hybrid Attn với Mamba2 (tham khảo [LIWIN](/LIWIN.md))
  - [ ] Thay 8K long attn layers bằng Mamba2
  - Mamba2 comparable performance at `2K` sequence length and becomes up to 6 times faster at 16K sequence length
  - [ ] Hymba https://huggingface.co/nvidia/Hymba-1.5B-Instruct | https://youtu.be/a31C8ahIDhk
    - https://github.com/NVlabs/hymba/tree/main/barebones_hymba

- ~~`LVOT          1.5x` (LLM-based Vocab Optim for Tokenization: better & denser hidden representation)~~
- ~~`N-gram Embedding  ` Tăng perf, giảm sự bất thường trong không gian embeddings~~

20250720
--------
- [ ] grokking với spectral clipping https://leloykun.github.io/ponder/spectral-clipping
- [ ] enforce Lipschitz bounds in training https://www.alphaxiv.org/abs/2507.13338?conversation_id=687c63997c6168cf0c07c8f4
  Khi sử dụng với Muon https://github.com/Arongil/lipschitz-transformers/blob/main/nanogpt/train_spectral_cap.py
  Điểm hay của pp này (giống muonclip) là nó control weight thay vì activation => nhẹ hơn và Prevent explosion at source
  - Spectral Soft Cap hoặc Spectral Normalization
  - Bỏ ~~layernorm, qk_norm~~
  - Có thể thêm light weight decay

- https://github.com/Niccolo-Ajroldi/plainLM minimal LLM training (recent) code

- [x] Hyper param tuning & training stablization
  - [x] ~~Mutliple step learning rates giống DeepSeek và MiMo7B~~ (không hiệu quả)
  - [x] bỏ weight decay ở embedding và lm_head
  - [x] optim hyperparam tuning for small batch size https://arxiv.org/abs/2506.12543
  - [ ] Muon qk clipping (chờ PyTorch impl https://github.com/pytorch/pytorch/issues/148819#issuecomment-3070108227)
    - __NOTE__ Có thể chỉ cần áp dụng `qk_norm` là đủ nếu không dùng MLA
  - Reading:
    - https://x.com/giffmana/status/1943384733418950815
    - https://x.com/YouJiacheng/status/1944696254623264926
    - https://x.com/YouJiacheng/status/1943930850724524245
    - https://x.com/konstmish/status/1945113604534985012
    - https://x.com/konstmish/status/1945105731352469875
    - https://x.com/krizna_b/status/1944854671728005588
    - https://x.com/BetaTomorrow/status/1943614107258601829
    - https://x.com/krizna_b/status/1944854671728005588
    - https://x.com/egor_shulg/status/1946329743311442185

- BlockFFN https://huggingface.co/SparseLLM/BlockFFN-3B-SFT based on ReMoE https://arxiv.org/abs/2412.14711

- [ ] Sửa flash attn online softmax để trả về max logits sau đó áp dụng muon qk clipping
  - https://mp.weixin.qq.com/s?__biz=MjM5ODExNDA2MA==&mid=2449991079&idx=1&sn=b313f59a3da0fcf61138da723adb5da0
    Vấn đề chỉ xuất hiện ở model lớn, moonlight 16b vẫn tự hạ được

- Synergy có dùng concept với Hnet https://www.alphaxiv.org/abs/2507.12769

- Quy chiếu MoA / FFN / Attention về chung cơ chế Sparse (Sparse Matrix / Sparse Matmul / MegaBlocks)
  - https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training
  - Sparsing Law https://www.alphaxiv.org/abs/2411.02335
    => Càng nhiều dữ liệu huấn luyện thì activation ratio càng giảm (sparsity càng tăng).
    => Mô hình 2.4B với ReLU đạt sparsity ratio 93.52% và tăng tốc 4.1× so với phiên bản dense.
  - [x] 1.3x nếu độ thưa > 90% https://github.com/pytorch/ao/tree/main/torchao/sparsity#int8-dynamic-quant--24-sparasity
  - Spark: 1/2 Q@K + GeLU làm score rồi chọn stastical topk (sparse 92% mlp & 96% attn) https://www.alphaxiv.org/abs/2506.06644
  - Polynomial Composition Activations giúp tăng perf https://arxiv.org/abs/2411.03884v3
  - [ ] Dùng Spark tạo độ thưa > 90%, kết hợp Poly để tăng khả năng nhớ rồi dùng 2:4 Sparse cho Act giúp speedup


20250724
--------
```py pretrain.py https://arxiv.org/html/2503.16672v1#S2
## https://github.com/pytorch/ao/tree/main/torchao/prototype/sparsity
# Activation quant dạng int8 đối xứng động theo từng token và 
# định lượng trọng số (weight) int8 theo từng kênh (per-channel) cho các lớp tuyến tính (linear).
# Usage: `quantize_(module, Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()))`
# Note: chỉ apply khi đã pretrain được vài ngàn steps để sparse pattern của activation được ổn định

# 5% training progress thì 2:4 sparse hoá sparsable_params
# TODO: điều tra lỗi trong 2:4 sparse engine khiến loss đi lên !!!
    if args.sparse and step == 2 * lr_schedule.t1:
        # Applies int8 dnynamic symmetric per-token activation and int8 per-channel weigh quantization + 2:4 sparsity
        from torchao.quantization.quant_api import quantize_, Int8DynamicActivationInt8WeightConfig
        from torchao.dtypes import SemiSparseLayout
        for m in sparsable_params: quantize_(m, Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()))
        # muon_optim.reset_momentum(shape=sparsable_params[0].weight.shape)
```
- Quy chiếu MoA / FFN / Attn về chung cơ chế Sparse (Sparse Matrix / Sparse Matmul / MegaBlocks)
  - `2:4 sparse` 1.3x nếu độ thưa > 90%
    - tích hợp https://github.com/pytorch/ao/tree/main/torchao/sparsity vào int8 mixed
      - https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py#L1363
  - `Spark` 1/2 Q@K + GeLU làm score rồi chọn stastical topk (sparse 92% mlp & 96% attn) https://www.alphaxiv.org/abs/2506.06644
  - `Polynomial` Composition Activations giúp tăng perf https://arxiv.org/abs/2411.03884v3
  - `Selective Attn` sử dụng lại 1 attn head làm selective mask https://www.alphaxiv.org/abs/2410.02703
  - Tận dụng độ thưa từ `Spark` / `Selective Attn` và hỗ trợ `FlashMask`
  - DistrAttention xấp xỉ ở hidden dim, 1.35x speedup https://www.alphaxiv.org/abs/2507.17245
  - `Stack Transformer` giúp học ngôn ngữ (hình thức) tốt hơn https://www.alphaxiv.org/abs/2507.15343
  - `PaTH` nâng cấp RoPE giúp Attn từ TC0 lên NC1 (bị loss=NaN)
  - `Gated Attention` Đầu ra đã được điều chỉnh `attn = attn ⊙ sigmoid(W_g(x))` (ko thấy tốt lên)
