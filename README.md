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

- [x] `Muon         ~1.5x` (Muon optimizer giúp giảm vram và tăng tốc độ hội tụ so với Adam)
- [x] `int8         ~1.5x` (Linear matmul sử dụng INT8 mixed precision giúp tăng tốc 1.5 lần)
- [x] `Dense Arch   ~2.0x` (giản lược kv_proj; sparse Relu^2; SWA 1k-8k; 1 norm per layer)
- [x] `OhMaiHead    ~1.3x` (Giảm vram và tăng tốc LCE khi finetune huge vocab models)
- [x] `Small batch  ~1.2x` (chỉ activation checkpoint với lite ops)
- [ ] `MoA          ~1.3x` (Mixture of Anything (Depth/Expert/Cơ chế); quy chiếu MoA về Sparse)
- [ ] `Modded Attn  ~1.3x` (Giảm IO khi không dùng RoPE; Flash/Dyna-Mask; Sparse Attn; FoX/PaTH ...)
- [ ] `Better Tknz  ~1.3x`

🌸 !!! TARGET x10 SPEEPUP WITHOUT PERF REDUCTION !!! 🌸

## [Kết quả thử nghiệm](https://github.com/mettamind-ai/CUTE/blob/research/.save/EXPER.md)
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

## Tiny Monster LLMs
- `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) để tiền xử lý
- `16k tokenmonster vocab` + **selected meaningful tokens** + `Stochastok` (random phân giã)
- Bài toán bộ gõ thông minh:
  - `auto/smart-edit/smart-complete/sửa lỗi chính tả/convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
  - tiny agents làm những task đơn nhiệm trong 1 quy trình nhỏ, mỗi task là 1 adaptor trên 1 based LLM chung

---

# TODO

- [ ] Thử thay hoặc kết hợp MTP với TOP (Token Order Prediction)
  - https://github.com/zaydzuhri/token-order-prediction
  - Related https://www.alphaxiv.org/abs/2404.07965

- Better tokenization
  - [STOCHASTOK](https://arxiv.org/abs/2506.01687) `p = 0.1` phân rã ngẫu nhiêu tokens trong batch để tăng tính robustness
  - OT, VEGAD https://github.com/mettamind-ai/CUTE/blob/research/.save/TKNZ.md

- PLE (Per-Layer Embedding) linh hoạt
  - [x] concat với value giúp giảm 1/2 value (loss giữ nguyên)

- MoA: Mixture Of Anthing (Expert, Depth, Các cơ chế học khác nhau ...)
  - MPAS alike https://www.alphaxiv.org/abs/2506.22389
  - LoRA + Sparse https://www.alphaxiv.org/abs/2508.02668
