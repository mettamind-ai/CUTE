# Learning Objectives
Contrastrive / GAN / Mask / Generative (GLM / T5 / UL2)

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

---

# Others

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

---

# Marin test various optimizers (muon seem the best)
- https://github.com/marin-community/marin/issues/1290
- https://github.com/marin-community/marin/blob/main/docs/reports/marin-8b-retro.md#training-phases
- WSD Cycle https://github.com/marin-community/marin/blob/main/docs/reports/marin-8b-retro.md#wsd-cycle-change
![](https://github.com/marin-community/marin/raw/main/docs/images/tootsie-8b-retro-wsd-interval.png)
- https://github.com/marin-community/marin/blob/main/docs/reports/index.md

---


MIXTURE OF EXPERTS
------------------

- DS MoE https://arxiv.org/html/2401.06066v1
(1) segmenting the experts into `mN` ones and activating `mK` from them; (2) isolating `K_s` experts as `shared ones`, aiming at **capturing common knowledge** and `mitigating redundancy in routed experts`. Starting from a modest scale with 2B parameters, we demonstrate that `DeepSeekMoE 2B achieves comparable performance with GShard 2.9B`, which has 1.5 times the expert parameters and computation. In addition, DeepSeekMoE 2B nearly **approaches the performance of its dense counterpart** with the same number of total parameters, which set the upper bound of MoE models. Subsequently, we `scale up DeepSeekMoE to 16B` parameters and show that it `achieves comparable performance with LLaMA2 7B`, with **only about 40% of computations**.
![](https://arxiv.org/html/2401.06066v1/x2.png)

![](https://arxiv.org/html/2412.19437v2/x2.png)
DeepSeek-V3 adheres to the settings of DeepSeek-V2

![](https://arxiv.org/html/2412.19437v2/x6.png)
DS-V3 FP8 Training: only the Linear operator is illustrated.

Fprop (forward pass), Dgrad (activation backward pass), and Wgrad (weight backward pass), are executed in FP8. FP8 Wgrad GEMM **allows activations to be stored in FP8** for use in the backward pass. This significantly reduces memory consumption.

Maintain the original precision (e.g., BF16 or FP32) for:
- the embedding module,
- the output head,
- MoE gating modules,
- normalization operators, and 
- **attention operators**

To further guarantee numerical stability, we store in higher precision:
- the master weights,
- weight gradients, and 
- optimizer states.


## OLMoE
- https://arxiv.org/html/2409.02060v2
- https://www.alphaxiv.org/abs/2409.02060

![](https://arxiv.org/html/2409.02060v2/x4.png)

![](https://arxiv.org/html/2409.02060v2/x6.png)
**Figure 4**: MoE vs. Dense. We train a 1.3B parameter dense model and a 1.3B active, 6.9B total parameter MoE model, each on 128 H100 GPUs. Apart from MoE-related changes, we train both with the same configuration for 130B tokens. The MoE contains 64 experts out of which 8 are activated with an FFN dimension of 1,024, while the dense model has an FFN dimension of 8,192. Thus both have the same number of active parameters. Top: The MoE reaches the final dense performance with ∼3× fewer tokens (or FLOPs, as both have the same active parameters ignoring the trivial router parameters). Bottom: Due to some memory overhead, this equates to ∼2× faster training. More results, logs, and configurations: https://wandb.ai/ai2-llm/olmoe/reports/Plot-MoE-vs-Dense--Vmlldzo4OTM0Mjkx

Survey https://icml.cc/media/icml-2024/Slides/35222_1r94S59.pdf

---

# LongCE
- https://www.youtube.com/watch?v=A36u6DB_TgU
- https://asap-seminar.github.io/assets/slides/asap-yifei-wang.pdf

|![](https://pbs.twimg.com/media/GsugIlKbMAAdLsq?format=jpg)|![](https://pbs.twimg.com/media/Gsugh-obIAAkSEo?format=jpg)|
|-|-|
|![](https://pbs.twimg.com/media/Gsugh-obIAAkSEo?format=jpg)|![](https://pbs.twimg.com/media/GsuhRFybAAAxMvP?format=jpg)|
|![](https://pbs.twimg.com/media/GsulWpTbsAAfZTI?format=jpg)|![](https://pbs.twimg.com/media/GsulvkObEAAAmn3?format=jpg)|

## 1. Vấn đề chính - SSL (Self-Supervised Learning):
- Làm sao xác định "key tokens" mà **không cần người gán nhãn**?
- Giải pháp: Tìm tokens phản ánh khả năng xử lý long context của model

## 2. Phương pháp - Causal Intervention:
**Ví dụ cụ thể:**
```
Long context: "Sarah has a dog named Buddy [...] Sarah feels happy to play with Buddy."
Short context: "Sarah feels happy to play with Buddy."
```
**Log Probability Gain (LPG):**
```
    r(x_i) =     P_θ(x_i|l_i) /     P_θ(x_i|s_i) =>
log r(x_i) = log P_θ(x_i|l_i) - log P_θ(x_i|s_i)
```
Trong ví dụ:
- Token `Buddy` (lần 2): LPG = 0.8/0.1 = 8 → **Key token!**
- Token `happy`: LPG = 0.6/0.6 = 1 → Không quan trọng

### 2.a) Chỉ dùng LPG → 85.6% accuracy:
- Phương pháp: Đặt threshold cho LPG (ví dụ: LPG > 2)
- **Vấn đề 14.4% còn lại:** Một số non-answer tokens cũng có LPG cao

### 2.b) Phân tích 14.4% sai:
Nhìn biểu đồ (a và b) trong hình trên:
- **Answer tokens** có LPG cao thường có LPV vừa phải ([-0.5, 0])
- **Non-answer tokens** (màu xanh) có LPG cao thường có **LPV rất thấp** (vùng khoanh tròn: (-∞, -2))
  Những token này thường có **Log Probability Value (LPV) thấp** => khó dự đoán

### 2.c) Giải pháp - Thêm tiêu chí LPV:
Combining LPV and LPG critiria, we can predict key tokens with 98.2% ACCURACY!

## 3. Công thức LongPPL
`LongPPL(x;θ,θ₀) = exp(∑ᵢ -Î(xᵢ;θ₀) log P_θ(xᵢ|x<ᵢ))`
Trong đó `Î(xᵢ;θ₀) = 1` khi:
- `LSD_θ₀(xᵢ) > α` (Log Score Difference cao)
- `LCL_θ₀(xᵢ) > β` (Log Context Length cao)

**Ưu điểm**:
- Focus vào tokens quan trọng cho long context
- Correlation cao với benchmarks (0.84-0.96)
- Hiệu quả: Chỉ cần model nhỏ (Llama-3.1-8B) để xác định key tokens


“Log Context Length” (LCL) là phép đo đánh giá **mức độ token xᵢ phụ thuộc vào context dài đến đâu**. Cách tính như sau:
- LCL(xᵢ) = log(k)  
- Với **k** là chiều dài context ngắn nhất (k < L) để model dự đoán xᵢ đạt xác suất (hoặc LPG) vượt ngưỡng nhất định.

**Cách tính phổ biến**
- Di chuyển “cửa sổ” context từ ngắn tăng dần đến dài:
  - Tính P(xᵢ | context_length = k) cho các giá trị k tăng dần.
  - **Khi P vượt qua threshold (hoặc LPG tăng mạnh),** đánh dấu `k_min`  
- Sau đó tính ```LCL(xᵢ) = log(k_min)```

**Ý nghĩa**
- Nếu xᵢ dễ đoán ngay từ context ngắn → log(k) thấp → không cần context dài.
- Nếu xᵢ chỉ đoán được khi context rất dài → log(k) cao → đặc trưng cho long context.

## Training and PE bias
|![](https://pbs.twimg.com/media/GsuoFoNbsAAYMGq?format=jpg)|![](https://pbs.twimg.com/media/GsuoWj5bgAAHGpR?format=jpg)|
|-|-|
|![](https://pbs.twimg.com/media/GsupKgnaoAAgWcb?format=jpg)|![](https://pbs.twimg.com/media/GsupfRuaoAA1VBh?format=jpg)|
|![](https://pbs.twimg.com/media/Gsup3sLaQAAwbtY?format=jpg)|![](https://pbs.twimg.com/media/GsuqYFra4AAhFNl?format=jpg)|

## Cần build more Context Model (explicit) thay vì thuần Token Model (implicit contextualize)  
![](https://pbs.twimg.com/media/Gsuq3QcaQAA_bca?format=jpg&name=4096x4096)

---

# LongCE
- https://github.com/PKU-ML/LongPPL
- https://alphaxiv.org/abs/2410.23771
![](https://github.com/PKU-ML/LongPPL/raw/main/longppl.png)

**Filtered key tokens** là những token được lọc ra dựa trên hai tiêu chí quan trọng mà tác giả đề xuất:

1. **Tiêu chí lọc**: Theo công thức trong bài báo, một token được coi là key token nếu:
- LSD (Long-Short Difference) > α: Token này được dự đoán tốt hơn đáng kể khi có ngữ cảnh dài so với ngữ cảnh ngắn
- LCL (Long-Context Likelihood) > β: Token này vẫn có thể dự đoán được với ngữ cảnh dài (loại bỏ những token quá khó)

2. **Quy trình filtering**: Như tác giả giải thích: "The first criterion ensures that the generation of the token is enhanced by the additional information in the long-context. The second criterion excludes the fundamentally hard (misclassified) tokens that long context information does not help."

[`LSD`: đo sự cải thiện dự đoán nhờ ngữ cảnh dài; `LCL`: xác suất dự đoán token dưới ngữ cảnh dài; `α, β`: ngưỡng tham số để lọc]

**LongCE (Long-context Cross-Entropy)** là hàm loss mới được đề xuất để fine-tuning các mô hình ngôn ngữ, **tập trung vào việc cải thiện khả năng xử lý ngữ cảnh dài.**

Nguyên lý hoạt động: Thay vì tính loss đều trên tất cả token như CE truyền thống, LongCE **gán trọng số cao hơn cho các key tokens**:
- `LongCE(x; θ) = -1/n ∑ Isoft(xi; θ) log Pθ(xi|x<i)`
- `Isoft(xi; θ) = min(exp(LSDθ(xi)), γ) = min(Pθ(xi|li)/Pθ(xi|si), γ)` -
  trọng số dựa trên tỉ lệ xác suất dự đoán giữa ngữ cảnh dài và ngữ cảnh ngắn.

Ưu điểm:
- Tự bootstrap: Mô hình tự đánh giá key tokens và tối ưu hóa chúng theo kiểu EM
- Plug-and-play: Có thể áp dụng trực tiếp vào quá trình fine-tuning hiện có
- Hiệu quả: Cải thiện lên đến 22% accuracy trên LongEval
- Overhead tính toán: Khoảng 80% thời gian training so với CE thông thường.

**Cách giảm overhead**: Như trong Appendix B.2: "by changing the hyperparameters of LongCE, i.e., the short context-length K and the sliding window length d, this overhead can be **`further reduced to 36%`, with almost no loss in model performance**"
=> có thể giảm xuống chỉ 36% mà vẫn giữ hiệu suất. Và **so với tổng chi phí training LLM**, overhead này không đáng kể.


Overhead cao do phải tính toán bổ sung nhiều forward passes:
Nguyên nhân chính: Để tính LongCE, cần tính LSD (Long-Short Difference) cho mỗi token, điều này yêu cầu:

- Forward pass cho long context: Pθ(xi|li) - như bình thường
- Forward pass cho short context: Pθ(xi|si) - THÊM cho mỗi token

**Giải pháp tối ưu**: Tác giả dùng `sliding window` technique: "resulting in a complexity of O((N − K)K²/d)" với **step size d=1024**, giảm overhead từ 80% xuống 36%.

Kỹ thuật sliding window được tác giả sử dụng với hai tham số kích thước chính. Đầu tiên là **context window size K được đặt ở mức `4096 tokens`**, đây chính là độ dài ngữ cảnh ngắn dùng **để tính toán short-context probability**. Thứ hai là **step size d với giá trị `1024 tokens`**.

Thay vì xử lý từng token riêng lẻ, phương pháp này **nhóm 1024 tokens lại để tính toán cùng lúc**, giúp cải thiện đáng kể hiệu suất. Khi tính short-context probabilities cho các token từ xi đến xi+d-1, hệ thống sẽ đặt token bắt đầu của ngữ cảnh một cách đồng nhất.

Tác giả cũng đã thử nghiệm nhiều kết hợp khác nhau trong Table 7. Khi sử dụng K=1k và d=1k thì overhead chỉ tăng 43%, còn với K=4k và d=4k thì overhead giảm xuống chỉ 36%. Tuy nhiên nếu giảm step size xuống d=512 thì overhead lại tăng vọt lên 150%. Nhờ thiết lập mặc định K=4096 và d=1024, độ phức tạp tính toán được giảm từ O((n-K)K²) xuống O((n-K)K²/d), mang lại hiệu quả đáng kể trong quá trình training.

hmm 4k window context mà đc cho là short? => Tác giả đang tính cho VERY LONG CONTEXT, có thể lên tới 32k?. Đúng rồi paper có mặc định:
- Long context: 32k tokens (full sequence)
- Short context: 4k tokens (truncated version)

- => long vs short là tương đối. Tác giả thử nghiệm K=1k cũng cho kết quả tốt.
- => chứng tỏ tác giả đang áp dụng vào giai đoạn long context finetune phía sau pretrain

---

# Áp dụng LongCE trong pretrain

Ưu điểm:

- Model học focus vào key tokens ngay từ đầu instead of learning bad habits
- Có thể tránh được "lost in the middle" problem từ pre-training stage
- Potential compound benefits qua millions of training steps

Rủi ro:

- Với random initialization, bootstrap có thể chậm hơn
- Cost 36-80% overhead trên scale pre-training là rất lớn
- Chưa có experimental evidence cho pre-training

=> Rất promising nhưng cần pilot study nhỏ để verify cost-benefit ratio trước khi scale up.

---

## FTP: Future Token Prediction
- https://www.alphaxiv.org/abs/2410.18160?conversation_id=68426a6181a77e60840110cc

## Grad Norm
- https://github.com/TianjinYellow/StableSPAM/blob/master/main_pretrain.py#L256
log và tính grad clipping cho hợp lý ...
```
gt = -∇loss/∇weights
gnorm = ||gt||2  # L2 norm của gradient
mnorm = γ1 * mnorm + (1-γ1) * gnorm     # first moment
vnorm = γ2 * vnorm + (1-γ2) * gnorm²    # second moment
adaptive_norm = mnorm / sqrt(vnorm + ε)
adaptive_norm = mnorm / sqrt(vnorm + ε)
gt = (gt / gnorm) * adaptive_norm
```
- `mnorm`: trung bình lịch sử của gradient norm
- `vnorm`: variance lịch sử của gradient norm
- `adaptive_norm`: tỷ lệ mean/std → giá trị "lý tưởng" cho gradient norm
- `gt`: gradient gốc từ backprop

