## 🌸`CUTE`🌸 Center for Upgrading Training Efficient
- `ONE_` gamming GPUs is optimal for training ~1b models
- `TWO_` gamming GPUs is optimal for training ~2b models
- `FOUR` gamming GPUs is optimal for training ~4b models
```
                        BF16          INT8
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
- [x] **Arch** ~~1.5x @ 6k ctxlen~~ (chưa đo lường)
- **Token**    1.5x (và **representation** nói chung)

🌸__!!! TARGET 10x SPEEPUP !!!__🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is super good! loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt

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

## [DONE](.save/DONE.md)
- [x] `Parallel Layers` [Primer](https://www.alphaxiv.org/abs/2109.08668) => chính là multi head attn?

🌸__DOING__🌸
- [ ] `Conv Attn` [Baichuan M1 14b](https://www.alphaxiv.org/abs/2502.12671)
  ![](https://arxiv.org/html/2502.12671v2/extracted/6253923/images/kv_shift_attention.png) => giống DConv 3x1 trong primer
  ![](https://user-images.githubusercontent.com/544269/134764948-4aef8641-f9c5-43a5-9bfd-c2316df3a434.png)
  `torch.nn.Conv2d(in_channels=head_size, out_channels=head_size, kernel_size=(3, 1), padding=0, groups=head_size)`
- [ ] Viết HuggingFace transformer wrapper cho wingpt để tiện inference

## SymMonsters: Build `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) + Tiny Monster Models
- `6k vocab` = `3k symato` (Vietnam) + `3k BPE` (English)
- Bài toán bộ gõ thông minh:
  - `auto-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
- VLM đọc nội dung `screenshots (Anh + Việt)`
- TTS cần 1 bộ tokenization khác thiên về phát âm
