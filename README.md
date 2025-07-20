## 🌸`CUTE`🌸 Center for Upgrading Training Efficient
- `ONE_` gamming GPUs can train < 1b models
- `TWO_` gamming GPUs can train < 2b models
- `FOUR` gamming GPUs can train < 4b models
```
                             BF16        INT8
3090      350W  24G     71 TFLOPS    284 TOPS
4090      450W  24G    165 TFLOPS    660 TOPS
5090      575W  32G    210 TFLOPS    838 TOPS
```
| Cấu hình        | Giá     | TFLOPs | TFLOPs/$ | DLPerf | DLPerf/$ |
| --------------- | ------- | ------ | -------- |------- | ---------|
|`032G`_2x5070ti  |$0.32/hr | 088    |**275.00**| 132    |**412.50**|
|`064G`_2x5090    |$0.97/hr | 216    |  222.68  | 283    | *291.75* |
| --------------- | ------- | ------ | -------- |------- | -------- |
|`096G`_4x4090    |$1.28/hr | 324    | *253.12* | 310    |  242.18  |
|`128G`_4x5090    |$1.96/hr | 432    |  220.41  | 524    |  267.34  |

- [x] `Muon          1.5x` (Muon optimizer giúp giảm vram và tăng tốc độ hội tụ so với Adam)
- [x] `int8          1.5x` (Linear matmul sử dụng INT8 mixed precision giúp tăng tốc 1.5 lần)
- [x] `Dense Arch    1.5x` (giản lược k_proj, v_proj, và bỏ o_proj trong attention; tối giản MLP với Relu^2)
- [x] `OhMai         1.3x` (Giảm vram và tăng tốc LCE khi finetune huge vocab models)
- [ ] `MoA           1.5x` (Mixture of Anything (Depth/Expert), quy chiếu MoA về Sparse)
- [ ] `Flexible Attn 1.5x` (vọc flash-attn để hỗ trợ flexible mask và sparse attn)
- [ ] `HNet dyna chunking` (có thể không tăng tốc nhưng giúp cải thiện perf / loại bỏ tknz và sparse attn?)

🌸 !!! TARGET x10 SPEEPUP !!! 🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is good! vram = 1/4 + loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt
- `per-layer value embeddings` + `future prediction` are all good!

## DATA
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**
- [Dùng GPU xử lý data](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/csrc/data.cpp)
- [ ] llm-scored data select giống seed coder https://www.alphaxiv.org/abs/2506.03524

## Tiny Monster Models
- `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) để tiền xử lý
- `4k hoặc 8k BPE vocab` + Stochastok (random phân giã) + 2,3-gram embeddings (random tổng hợp)
- Bài toán bộ gõ thông minh:
  - `auto/smart-edit`
  - `auto/smart-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
- TTS cần 1 bộ tokenization khác thiên về phát âm
- VLM đọc screenshots

---
# TODO

- [ ] Hyper param tuning & training stablization
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
  

- [ ] dùng hnet + tknz nhẹ (giống pre-processing) để cải thiện token embedding / representation
  - [ ] 
  - [ ] hnet có giúp loại bỏ sparse attn?

- [ ] MoA: Mixture Of Anthing (Expert, Depth, Các cơ chế học khác nhau ...)
  - [ ] Mixture-of-Recursions https://arxiv.org/abs/2507.10524
  - [ ] Gated DeltaNet + SWA + Mamba2 https://www.alphaxiv.org/abs/2412.06464

- [ ] Quy chiếu MoA / FFN / Attention về chung cơ chế Sparse (Block Sparse Matrix & Matmul pattern)
  - Sparsing law https://www.alphaxiv.org/abs/2411.02335 => ReLU tuân theo quy luật logspace power-law giảm dần
    => càng nhiều dữ liệu huấn luyện thì activation ratio càng giảm (sparsity càng tăng).
    => Mô hình 2.4B với ReLU đạt sparsity ratio 93.52% và tăng tốc 4.1× so với phiên bản dense.
  - Spark Transformers (sparse both mlp & attn) https://www.alphaxiv.org/abs/2506.06644
  - [ ] MegaBlock hỗ trợ SSD, DSD, DDS matmul
  - [ ] BlockFFN  https://huggingface.co/SparseLLM/BlockFFN-3B-SFT based on ReMoE https://arxiv.org/abs/2412.14711
  - [ ] PolyReLU  https://arxiv.org/abs/2411.03884v3
  - [ ] 1.3x FFN  https://github.com/pytorch/ao/tree/main/torchao/sparsity#int8-dynamic-quant--24-sparasity
