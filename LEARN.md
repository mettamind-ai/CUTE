# Learning Objectives

- Contrastrive
- GAN
- Mask
- Generative
- GLM
- T5

## GLM
- https://www.alphaxiv.org/abs/2103.10360
- https://github.com/THUDM/GLM/blob/main/model/modeling_glm.py#L40
![](/.save/learn-obj-00.png)
- `GLM410M` (1.25×BERTLarge) đạt hiệu suất tốt hơn các mô hình standalone.
- `Loại bỏ span shuffling` "leads to a severe performance drop on SuperGLUE"
- `GLM khắc phục điểm yếu của BERT`: "BERT fails to capture the interdependencies of masked tokens due to the independence assumption of MLM" bằng cách "randomly permute the order of the spans" để "fully capture the interdependencies between different spans."
- `Trade-off khi cùng số tham số`: "With the same amount of parameters, GLMDoc performs worse than GPTLarge. This is expected since GLMDoc also optimizes the blank infilling objective" - vì GLM phải tối ưu cho nhiều mục tiêu khác nhau thay vì chỉ tập trung vào language modeling như GPT.
- `Tăng tham số`: "Increasing the model's parameters to 410M (1.25× of GPTLarge) leads to a performance close to GPTLarge" và "GLM515M (1.5× of GPTLarge) can further outperform GPTLarge"

# Flash Attn in Triton
- [`./save/attn.py`](https://github.com/bryanzhang/triton_fusedattention/blob/main/fused-attention.py)
- https://www.youtube.com/watch?v=zEuwuCTEf_0
- https://www.youtube.com/watch?v=4jQTb6sRGLg
- https://www.youtube.com/watch?v=zy8ChVd_oTM

# mô phỏng long short layers
- Train 2k ctxlen trước, sau đó freeze 2/3 layers rồi train tiếp với 4k ctxlen

# INT8 SageBwd
https://www.alphaxiv.org/abs/2505.11594

Trong forward, SageBwd áp dụng `per-block quantization` cho Q, K, V và `per-token quantization` cho P. Đối với backward pass, phương pháp này quantize 4 trong 5 phép nhân ma trận xuống INT8, nhưng `giữ phép toán dOV^T ở FP16` để duy trì độ chính xác. Lý do là gradient của attention map rất nhạy cảm với quantization errors và có thể tích lũy lỗi qua các sequence dài. SageBwd nhanh hơn FlashAttention `1.67x`  RTX4090, với `end-to-end speedup khoảng 1.15` lần cho Llama models. Tuy nhiên, SageBwd có hạn chế trong pretraining tasks khi **tốc độ hội tụ chậm hơn so với BF16**. Các tác giả đề xuất tối ưu kernel implementation và `nghiên cứu thêm về ứng dụng low-bit attention trong pretraining`.

SageBwd có tốc độ hội tụ chậm hơn BF16 trong pretraining vì **quantization errors tích lũy** ảnh hưởng đến chất lượng gradient. Khi quantize các phép nhân ma trận xuống INT8, độ chính xác của gradient bị giảm, dẫn đến việc model học chậm hơn. Sự khác biệt giữa pretraining và fine-tuning là ở mức độ thay đổi cần thiết. Pretraining yêu cầu model học toàn bộ kiến thức từ đầu, cần gradient chính xác để cập nhật trọng số hiệu quả. Trong khi đó, `fine-tuning` chỉ cần điều chỉnh nhỏ từ model đã có kiến thức sẵn, nên `ít nhạy cảm hơn với noise trong gradient`.

---

- optim scheduler: scaling laws for wd & bs in llm training
  https://x.com/dmsobol/status/1925273068840390801
  https://x.com/dmsobol/status/1895179989664047442
  - `wd = 0.1` is suboptimal, should scales linearly with bs
  - `EMA` (Exponential Moving Average) 

- command+a https://alphaxiv.org/abs/2504.00698
  - n x { `3 swa` (RoPE) + `1 full attn` (NoPE) }
  - NoPE giúp tổng quát hoá tốt hơn
  - fp8 then bf16 to stable training

- gemma3 & https://ai.google.dev/gemma/docs/gemma-3n
  
- weighted loss https://x.com/kalomaze/status/1880923963880300941
