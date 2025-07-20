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
- [x] `Dense Arch    1.5x` (giản lược kv_proj, bỏ o_proj; sparse Relu^2; SWA 2k-8k; 1 norm / layer)
- [x] `OhMaiHead     1.3x` (Giảm vram và tăng tốc LCE khi finetune huge vocab models)
- [x] `Small batch   1.2x` (chỉ activation checkpoint với light ops)
- [ ] `MoA           1.5x` (Mixture of Anything (Depth/Expert/Cơ chế); quy chiếu MoA về Sparse)
- [ ] `HNet dyna chunking` (có thể không tăng tốc nhưng giúp cải thiện perf / loại bỏ tknz và sparse attn?)

🌸 !!! TARGET x10 SPEEPUP WITHOUT PERF REDUCE !!! 🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is good! vram = 1/4 + loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt
- `per-layer value embeddings` + `future prediction` are all good!
- `small batch size` giúp kích hoạt `ohmaihead` giảm LCE computing đáng kể

## DATA
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**
- [Dùng GPU xử lý data](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/csrc/data.cpp)
- [ ] llm-scored data select giống seed coder https://www.alphaxiv.org/abs/2506.03524

## Tiny Monster Models
- `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) để tiền xử lý
- `4k BPE vocab` + `Stochastok` (random phân giã) + `HNet` (dồn LCE computing cho HNet)
- Bài toán bộ gõ thông minh:
  - `auto/smart-edit`
  - `auto/smart-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)

---
# TODO

- HNet để cải thiện token embedding / abstract representation
  - [ ] tknz nhẹ (giống pre-processing)
  - [ ] hnet có giúp loại bỏ sparse attn?

- MoA: Mixture Of Anthing (Expert, Depth, Các cơ chế học khác nhau ...)
  - [ ] Mixture-of-Recursions https://arxiv.org/abs/2507.10524
  - [ ] Gated DeltaNet + SWA + Mamba2 https://www.alphaxiv.org/abs/2412.06464
  - [ ] DeltaFormer, áp dụng delta rule vào value của softmax attn

- Quy chiếu MoA / FFN / Attention về chung cơ chế Sparse (Block Sparse Matrix / Sparse Matmul / MegaBlocks)
  - Sparsing law https://www.alphaxiv.org/abs/2411.02335
    => ReLU tuân theo quy luật logspace power-law giảm dần
    => Càng nhiều dữ liệu huấn luyện thì activation ratio càng giảm (sparsity càng tăng).
    => Mô hình 2.4B với ReLU đạt sparsity ratio 93.52% và tăng tốc 4.1× so với phiên bản dense.
  - [ ] Spark Transformers (sparse both mlp & attn) https://www.alphaxiv.org/abs/2506.06644
  - [ ] 1.3x ReLU https://github.com/pytorch/ao/tree/main/torchao/sparsity#int8-dynamic-quant--24-sparasity
  - [ ] PolyReLU https://arxiv.org/abs/2411.03884v3
