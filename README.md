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
- [ ] `MoA           1.5x` (Mixture of Anything (Depth/Expert))
- [ ] `Sparse Attn   1.5x` (vọc flash-attn để hỗ trợ flexible mask và sparse attn)
- [ ] `LVOT          1.5x` (LLM-based Vocab Optim for Tokenization: better & denser hidden representation)
- [ ] `N-gram Embedding  ` Tăng perf, giảm sự bất thường trong không gian embeddings

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

- [ ] grokking với spectral clipping https://leloykun.github.io/ponder/spectral-clipping
- [ ] llm-scored data select giống seed coder https://www.alphaxiv.org/abs/2506.03524
- [ ] Tìm vocab cân bằng https://arxiv.org/abs/2402.18376

- [ ] MoA: Mixture Of Anthing
  - https://www.alphaxiv.org/abs/2202.09368 để expert chọn top-k token với k cố định sẽ đơn giản hơn để token chọn expert
  - thêm điểm attn score từ layer trước để tránh bias ...

- [ ] Quy chiếu MoA, Sparse Attention về chung Block Sparse Matrix & Matmul pattern
  - Attn https://github.com/mit-han-lab/Block-Sparse-Attention
  - MLP  https://www.alphaxiv.org/abs/2506.06644
  - MegaBlock và https://www.deepspeed.ai/tutorials/sparse-attention hỗ trợ SSD, DSD, DDS matmul

- Linear Attn
  - https://sustcsonglin.github.io/blog/2024/deltanet-1
  - https://people.csail.mit.edu/yoonkim/data/efficient_architectures_talk.pdf
  - https://leloykun.github.io/ponder/test-time-regression
  - https://goombalab.github.io/blog/2025/tradeoffs

- [ ] Hyper param tuning
  - [x] Batch size Warmup giống MiMo7B
  - [x] Mutliple step learning rates giống DeepSeek và MiMo7B
  - [x] bỏ weight decay ở embedding và lm_head
