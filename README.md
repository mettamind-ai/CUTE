# 🌸`CUTE`🌸 Center for Upgrading Training Efficient
- `ONE_` gamming GPUs is optimal for training ~1b models
- `TWO_` gamming GPUs is optimal for training ~2b models
- `FOUR` gamming GPUs is optimal for training ~4b models

| Cấu hình        | Giá     | TFLOPs | TFLOPs/$ | DLPerf | DLPerf/$ |
| --------------- | ------- | ------ | -------- |------- | ---------|
|`032G`_2x5070ti  |$0.32/hr | 088    |**275.00**| 132    |**412.50**|
|`064G`_2x5090    |$0.97/hr | 216    |  222.68  | 283    | *291.75* |
| --------------- | ------- | ------ | -------- |------- | -------- |
|`096G`_4x4090    |$1.28/hr | 324    | *253.12* | 310    |  242.18  |
|`128G`_4x5090    |$1.96/hr | 432    |  220.41  | 524    |  267.34  |

- [x] **Muon**2-3x
- [x]  int8   1.5x
- **Win**     1.5x @ 6k ctxlen
- **Token**   1.5x (và **representation** nói chung)

🌸__!!! TARGET 10x SPEEPUP !!!__🌸

## 1.5 GPUs là được gì?
```                    BF16          INT8
3090      350W  24G     71 TFLOPS    284 TOPS
4090      450W  24G    165 TFLOPS    660 TOPS
5090      575W  32G    210 TFLOPS    838 TOPS
5060-TI   180W  16G
```
- https://www.pugetsystems.com/labs/articles/nvidia-geforce-rtx-5090-amp-5080-ai-review/
- https://www.reddit.com/r/LocalLLaMA/comments/1hviw58/simple_table_to_compare_3090_4090_and_5090
- https://www.reddit.com/r/LocalLLaMA/comments/14tfr8h/doesnt_a_4090_massively_overpower_a_3090_for
- https://www.tomshardware.com/pc-components/gpus/nvidia-geforce-rtx-5060-ti-16gb-review/8

=> Đẩy được 1.3x perf với giá 15tr/50tr

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is super good! nó sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt

---

# 🌸 TODO 🌸
### DISTILL
- NAS + healing của Nvidia để thu 7b-8b về 4b-5b
- Sau đó chưng cất Attn về Linear https://huggingface.co/papers/2505.03005
- Nếu chưng cất thẳng 7b-8b về 4b-5b thì tốt!
### DATA
- vocab_size <= 65k để token_id lưu ở 16 bits (2-bytes)
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**
- Áp dụng được [packed dataset](https://github.com/mettamind-ai/MAD/blob/main/PACKED.md) 
  để tránh contamination giữa các samples thì tốt. Nếu không dùng mẹo sau:
- Phối data cùng 1 domain theo tỉ lệ **chính-phụ** `40:20:20:10:10` 
  - là cái gì core sẽ đưa vào group 40 đó
  - cái nào phụ sẽ giảm dần theo tỉ lệ lệ thuộc
## PLANING
- **Schedule free optim**
- **Spiral** Tăng dần hidden dim 4 layer 1 lần, đến cuối lại thu nhỏ lại Đối xứng theo U shape

## [DONE](.save/DONE.md)
🌸__DOING__🌸
- [ ] `Prime` Parallel Layers
- [ ] `Conv Attn` [Baichuan M1 14b](https://www.alphaxiv.org/abs/2502.12671)
- [ ] Đọc hiểu https://arxiv.org/abs/2410.17897

## SymMonsters: Build `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) + Tiny Monster Models
- `6k vocab` = `3k symato` + `3k BPE`
  **token embeds + head** chỉ còn `18m` so với `96m` (32k vocab, 1.5k hidden dim) 
- Bài toán bộ gõ thông minh:
  - `auto-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
- VLM đọc nội dung `screenshots (Anh + Việt)`
