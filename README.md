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

- [x] **Muon** 2-3x
- [x] **int8** 1.5x
- [x] **Arch** 1.5x @ 6k ctxlen (chưa đo lường)
- **Token**    1.5x (và **representation** nói chung)

🌸__!!! TARGET 10x SPEEPUP !!!__🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is super good! vram = 1/4 + loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt
- `value embeddings` + `multi exits` + `future prediction` should be good nhưng chưa thể hiện trên loss

---

# 🌸 TODO 🌸
### DATA
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**
- Áp dụng được packed dataset để tránh cross attention giữa các samples thì tốt. Nếu không dùng mẹo sau:
- Phối data cùng 1 domain theo tỉ lệ **chính-phụ** `40:20:20:10:10` 
  - là cái gì core sẽ đưa vào group 40 đó
  - cái nào phụ sẽ giảm dần theo tỉ lệ phía sau
## PLANING
- **Schedule free optim**
- **Spiral** Tăng dần hidden dim 4 layer 1 lần, đến cuối lại thu nhỏ lại Đối xứng theo U shape
- Tìm hiểu `exits > 1` + `torch.compile` khiến vram bị đội lên

## [DONE](.save/DONE.md)
🌸__DOING__🌸
- [ ] save params +  HF's transformers wrapper cho wingpt để tiện inference
- [ ] Canon impl in triton https://github.com/fla-org/flash-linear-attention/pull/388
  - https://github.com/fla-org/flash-linear-attention/blob/canon/fla/modules/canon.py

## Build `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) Tiny Monster Models
- `6k vocab` = `3k symato` (Vietnam) + `3k BPE` (English)
- Bài toán bộ gõ thông minh:
  - `auto-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
- VLM đọc nội dung `screenshots (Anh + Việt)`
- TTS cần 1 bộ tokenization khác thiên về phát âm
