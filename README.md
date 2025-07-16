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
- [x] `Dense Arch    1.5x` (lược bỏ k_proj, v_proj, o_proj trong attention; tối giản MLP với Relu^2; MTP)
- [x] `OhMai         1.2x` (Giảm vram và tăng tốc LCE khi finetune huge vocab models)
- [ ] `MoA           1.5x` (Mixture of Anything (Depth/Expert), quy chiếu MoA về Sparse)
- [ ] `Flexible Attn 1.5x` (vọc flash-attn để hỗ trợ flexible mask và sparse attn)
- [ ] `HNet dyna chunking` (có thể không tăng tốc nhưng giúp cải thiện perf / loại bỏ tknz và sparse attn?)
- ~~`LVOT          1.5x` (LLM-based Vocab Optim for Tokenization: better & denser hidden representation)~~
- ~~`N-gram Embedding  ` Tăng perf, giảm sự bất thường trong không gian embeddings~~

🌸__!!! TARGET x10 SPEEPUP !!!__🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is good! vram = 1/4 + loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt
- `value embeddings` + `future prediction` are all good!

## DATA
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**
- [Dùng GPU xử lý data](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/csrc/data.cpp)

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

- [ ] Hyper param tuning
  - [x] Batch size Warmup giống MiMo7B
  - [x] Mutliple step learning rates giống DeepSeek và MiMo7B
  - [x] bỏ weight decay ở embedding và lm_head
  - [ ] áp dụng Muon qk clipping (kimi K2)
  - [ ] optim hyperparam tuning for small batch size

- [ ] dùng hnet + tknz nhẹ (giống pre-processing) để cải thiện token embedding / representation
  - [ ] hnet có giúp loại bỏ sparse attn?

- [ ] grokking với spectral clipping https://leloykun.github.io/ponder/spectral-clipping
- [ ] llm-scored data select giống seed coder https://www.alphaxiv.org/abs/2506.03524

- [ ] MoA: Mixture Of Anthing (Expert, Depth ...)
  - [ ] Mixture-of-Recursions https://arxiv.org/abs/2507.10524
- [ ] Quy chiếu MoA / Sparse Attention về chung cơ chế Sparse (Block Sparse Matrix & Matmul pattern)
  - Attn https://github.com/mit-han-lab/Block-Sparse-Attention
  - Sparsing law https://www.alphaxiv.org/abs/2411.02335 => ReLU tuân theo quy luật logspace power-law giảm dần
    => càng nhiều dữ liệu huấn luyện thì activation ratio càng giảm (sparsity càng tăng).
    => Mô hình 2.4B với ReLU đạt sparsity ratio 93.52% và tăng tốc 4.1× so với phiên bản dense.
  - Spark Transformers (sparse both mlp & attn) https://www.alphaxiv.org/abs/2506.06644
  - [ ] MegaBlock hỗ trợ SSD, DSD, DDS matmul
  - [ ] BlockFFN  https://huggingface.co/SparseLLM/BlockFFN-3B-SFT based on ReMoE https://arxiv.org/abs/2412.14711
  - [ ] PolyReLU  https://arxiv.org/abs/2411.03884v3
  - [ ] 1.3x FFN  https://github.com/pytorch/ao/tree/main/torchao/sparsity#int8-dynamic-quant--24-sparasity
