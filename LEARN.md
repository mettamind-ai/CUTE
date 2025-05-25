# Learning Objectives
Contrastrive / GAN / Mask / Generative
- GLM
- T5
- UL2

## GLM
- https://www.alphaxiv.org/abs/2103.10360
- https://github.com/THUDM/GLM/blob/main/model/modeling_glm.py#L40
![](/.save/learn-obj-00.png)
- `GLM410M` (1.25×BERTLarge) đạt hiệu suất tốt hơn các mô hình standalone.
- `Loại bỏ span shuffling` "leads to a severe performance drop on SuperGLUE"
- `GLM khắc phục điểm yếu của BERT`: "BERT fails to capture the interdependencies of masked tokens due to the independence assumption of MLM" bằng cách "randomly permute the order of the spans" để "fully capture the interdependencies between different spans."
- `Trade-off khi cùng số tham số`: "With the same amount of parameters, GLMDoc performs worse than GPTLarge. This is expected since GLMDoc also optimizes the blank infilling objective" - vì GLM phải tối ưu cho nhiều mục tiêu khác nhau thay vì chỉ tập trung vào language modeling như GPT.
- `Tăng tham số`: "Increasing the model's parameters to 410M (1.25× of GPTLarge) leads to a performance close to GPTLarge" và "GLM515M (1.5× of GPTLarge) can further outperform GPTLarge"

- The models are trained on 64 V100 GPUs for `200K steps` with `batch size of 1024` and `maximum sequence length of 512`.
- GLMRoBERTa chỉ cần "250,000 steps, which are half of RoBERTa and BART's training steps and close to T5 in the number of trained tokens" nhưng vẫn đạt hiệu suất tương đương hoặc tốt hơn.
- Trade-off batch size: "For trade-off of training speed and fair comparison with BERT (batch size 256 and 1,000,000 training steps), we use batch size of 1024 and 200,000 training steps for GLMLarge" - GLM tăng batch size để giảm steps, tối ưu tốc độ.

## UL2
- https://huggingface.co/google/flan-ul2
![](https://raw.githubusercontent.com/google-research/google-research/master/ul2/figs/ul2.png)
![](https://raw.githubusercontent.com/google-research/google-research/master/ul2/figs/mod.png)
![](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjoRWMTOf1JUl345eb5BqKEPTRRxPvzPdzvspKtqlwNHqo4BVq98MJYkvEVPZAPdYmLaFMLQKAolOdzKD3uzbYTdYM8S9Z-y5BXgy6kotdukG8w9VCkrZt3Vb0H-BEDp8XC5bGIsA_OEQPWWll1vNRZbSBwJWowTCTf9cnW-7fDOXT8MmyH5s8KzieCQg/s16000/image3.gif)

- **R-Denoiser**: The regular denoising is the standard span corruption introduced in T5 that uses a range of 2 to 5 tokens as the span length, which masks about 15% of input tokens. These spans are short and potentially useful to acquire knowledge instead of learning to generate fluent text.

- **S-Denoiser**: A specific case of denoising where we observe a strict sequential order when framing the inputs-to-targets task, i.e., prefix language modeling. To do so, we simply partition the input sequence into two sub-sequences of tokens as context and target such that the targets do not rely on future information. This is unlike standard span corruption where there could be a target token with earlier position than a context token. Note that similar to the Prefix-LM setup, the context (prefix) retains a bidirectional receptive field. We note that S-Denoising with very short memory or no memory is in similar spirit to standard causal language modeling.

- **X-Denoiser**: An extreme version of denoising where the model must recover a large part of the input, given a small to moderate part of it. This simulates a situation where a model needs to generate long target from a memory with relatively limited information. To do so, we opt to include examples with aggressive denoising where approximately 50% of the input sequence is masked. This is by increasing the span length and/or corruption rate. We consider a pre-training task to be extreme if it has a long span (e.g., ≥ 12 tokens) or have a large corruption rate (e.g., ≥ 30%). X-denoising is motivated by being an interpolation between regular span corruption and language model like objectives.


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
