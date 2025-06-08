## 🌸`CUTE`🌸 Center for Upgrading Training Efficient
- `ONE_` gamming GPUs can train ~1b models
- `TWO_` gamming GPUs can train ~2b models
- `FOUR` gamming GPUs can train ~4b models
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

- [x] **Muon**          2.0x
- [x] **int8**          1.5x
- [x] **Dense Arch**    1.5x @ 6k ctxlen (chưa đo lường)
- [x] **OhMai**         1.3x
- [ ] **Flexible MoE**  1.5x
- [ ] **Super Token**   2.0x (better & denser representations in the hidden space)

🌸__!!! TARGET x10 SPEEPUP !!!__🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is super good! vram = 1/4 + loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt
- `value embeddings` + `multi exits` + `future prediction` should be good nhưng chưa thể hiện trên loss


## DATA
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**
- [Dùng GPU xử lý data](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/csrc/data.cpp)

## Tiny Monster Models
- `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne)
- `6k vocab` = `3k symato` (Vietnam) + `3k BPE` (English)
- Bài toán bộ gõ thông minh:
  - `auto-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
- TTS cần 1 bộ tokenization khác thiên về phát âm
- VLM đọc screenshots

## 🌸LINH HOẠT🌸 Dense + MoE + Reused Block + Precision + Size + Text Token/Super Token + Multi-Modals + Đa Mục Tiêu Học?
Một sự linh hoạt toàn diện trong cách xây dựng model, và tìm kiếm hiệu quả thực sự trong các cách kết hợp
linh hoạt đó? `Linh hoạt không khó, linh hoạt mang lại hiệu quả mới khó!`

- [ ] save/quant params + inference
  - https://github.com/pytorch-labs/gpt-fast
  - https://pytorch.org/blog/accelerating-generative-ai-2

- [ ] Tìm hiểu các cách Attn trong NSA
- [ ] CUTE flash attn
- [ ] LongCE và cách làm giảm hạn chế của Causual Attn
- [ ] Token được tự do lựa chọn:
  - cách nó attn
  - cách nó chọn số computing / hidden dim để biểu diễn chính nó

- [ ] Với 1 model mạnh nói chung nhưng yếu domain, có thể kết hợp logits distill + pre-train để:
  - giảm dataset phải chuẩn bị cho nó học?
  - học cách phân bổ dữ liệu nhanh hơn?
  