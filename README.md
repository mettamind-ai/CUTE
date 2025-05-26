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
- [ ] **Super Token** 1.5x (better & denser representations in the hidden space)

🌸__!!! TARGET 10x SPEEPUP !!!__🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is super good! vram = 1/4 + loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt
- `value embeddings` + `multi exits` + `future prediction` should be good nhưng chưa thể hiện trên loss

---

## DATA
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**
- [Dùng GPU để xử lý nhanh](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/csrc/data.cpp)
## PLANING
- Canon https://github.com/fla-org/flash-linear-attention/pull/388
- Gluon https://www.alphaxiv.org/abs/2505.13416
- Scion https://github.com/LIONS-EPFL/scion
- GLM / UL2 learning objs
- Học cách thu nhỏ model và NAS
  - https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1
  - NAS https://arxiv.org/abs/2411.19146
  - Llama-Nemotron https://arxiv.org/abs/2505.00949
- Cross-Layer Attention (CLA) -  sharing key and value heads between adjacent layers - https://arxiv.org/abs/2405.12981
## Build `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) Tiny Monster Models
- `6k vocab` = `3k symato` (Vietnam) + `3k BPE` (English)
- Bài toán bộ gõ thông minh:
  - `auto-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
- TTS cần 1 bộ tokenization khác thiên về phát âm

## [DONE](.save/DONE.md)
- [x] seq packing without flash-attn <= nested tensor của pytorch chưa chín (đợi thêm)
- [x] `Conv Attn` [Baichuan M1 14b](https://www.alphaxiv.org/abs/2502.12671)

🌸__DOING__🌸
- [ ] save params + inference
  - https://github.com/pytorch-labs/gpt-fast
  - https://pytorch.org/blog/accelerating-generative-ai-2
- [ ] thử nghiệm ý tưởng chỉ update gradients với tokens của 10 batches.
  - [LigerKernel Embedding](/liger_kernel.py#L70) => cách họ tối ưu IO / embedding lookup ...
  - [ ] Tìm cách khuếch tán gradients ra các tokens không được load vào head
    - Mỗi lần update gradients load ngẫu nhiên 1 số tokens vào head chẳng hạn ...
    - Đo lường sự giống và khác nhau giữa việc load full head và load part of head ...
    - => tìm ra quy luật + rút ra kinh nghiệm