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

---

# LOGITS / DISTILL / CALIBRATE / SPARSE / ADAPTIVE TKNZ / N-GRAM VOCAB / 2-GRAM LOSS VIA MTP

- https://alphaxiv.org/abs/2410.23771v4?conversation_id=68424c92c01a2de64aa2bd8c LongCE
- https://alphaxiv.org/abs/2408.12168v1 FIRST
- https://alphaxiv.org/abs/2506.01084v1 zip2zip ??
- https://alphaxiv.org/abs/2501.16975v2?conversation_id=684a2e78d4c8b5ddb23672b5 Over Tokenized Transformer
- https://proceedings.neurips.cc/paper_files/paper/2024/file/cdf00c97c0cb2cc35179f03363da6c4f-Paper-Conference.pdf ADATOK
- https://alphaxiv.org/abs/2410.04335v1?conversation_id=684b9e2201b4f61b63a7ab65 EVOT
- https://www.alphaxiv.org/abs/2410.01188 dùng gradient trên toàn bộ dataset để chấm điểm thay vì logits

---

ABBA PEFT Finetune https://www.alphaxiv.org/abs/2505.14238
```
s_ABBA = α²_LoRA/√(r₁r₂), where α_LoRA is the standard LoRA scaling coefficient
```

Tên gọi "ABBA" được tác giả giải thích một cách khá đơn giản trong bài báo. Họ cho biết: **"The name ABBA reflects the four low-rank matrices that define the architecture."**

Cụ thể, trong kiến trúc ABBA, weight update được định nghĩa là: **∆W = s(B₁A₁) ⊙ (B₂A₂)**, bao gồm bốn ma trận low-rank: **A₁, B₁, B₂, A₂**. Khi sắp xếp các ma trận này theo thứ tự xuất hiện, ta có: **A-B-B-A**, chính là từ "ABBA".

Đây là cách đặt tên khá sáng tạo và dễ nhớ, phản ánh đúng cấu trúc toán học của phương pháp. Thay vì chọn một tên phức tạp hoặc viết tắt khó nhớ, tác giả đã tạo ra một từ đơn giản mà bất kỳ ai quen thuộc với nhóm nhạc ABBA nổi tiếng cũng có thể nhớ được.

Cách đặt tên này cũng thể hiện tính đối xứng trong kiến trúc: hai cặp adapter (B₁A₁) và (B₂A₂) hoạt động song song và được kết hợp thông qua Hadamard product, tạo nên một cấu trúc cân bằng giống như tên gọi ABBA.

[Low-rank matrices: ma trận low-rank - ma trận có rank thấp hơn nhiều so với kích thước; Weight update: cập nhật trọng số; Hadamard product: tích Hadamard - phép nhân từng phần tử tương ứng; Adapter: bộ điều hợp - module nhỏ có thể huấn luyện được]

## Paper6: # Dynamic Token Pooling
- https://www.alphaxiv.org/abs/2211.09761
tự động pool các tokens (trong paper là characters) gần nhau lại thành 1 vector (giảm seqlen), tăng chất lượng, speedup và giảm vram. Transformer với các layers hình đồng hồ cát (phình 2 đầu với số tokens = nhau) và giảm ở giữa (pooling). hdim không đổi ở mọi layers.


|![](https://pbs.twimg.com/media/GtxpWG_WoAA6VIy?format=jpg)|![](https://pbs.twimg.com/media/GtxqA1DbIAEE2ez?format=jpg)|
|-|-|

Dynamic pooling hoạt động như một cỗ máy thông minh có khả năng "đọc hiểu" văn bản và tự động chia nhỏ thành các đoạn có ý nghĩa, thay vì cắt đại theo kích thước cố định như các phương pháp trước đây.

**Bước 1: Đọc và hiểu văn bản ban đầu**

Model nhận vào một chuỗi ký tự dài, ví dụ "with one of his greatest performances in last tango". Khối Transformer đầu tiên sẽ xử lý từng ký tự và tạo ra các hidden representations - như việc model "suy nghĩ" về từng phần của câu.

**Bước 2: Dự đoán ranh giới có ý nghĩa**

Một mạng neural nhỏ gọi là boundary predictor sẽ xem xét từng vị trí trong câu và quyết định có nên "cắt" tại đó không. Giống như cách con người đọc và tự nhiên chia câu thành các từ hoặc cụm từ có nghĩa. Ví dụ boundary predictor có thể quyết định cắt sau "with", "one", "of his", "greatest", "performances" để tạo thành các segment tự nhiên.

**Bước 3: Gom nhóm thông minh**

Thay vì chia đều thành các nhóm 3-4 ký tự như fixed pooling, dynamic pooling sẽ gom các ký tự thuộc cùng một đơn vị ý nghĩa lại với nhau. Ví dụ tất cả ký tự trong từ "greatest" sẽ được gom thành một representation duy nhất bằng cách lấy trung bình. Điều này giúp model hiểu được ranh giới tự nhiên của ngôn ngữ.

**Bước 4: Xử lý hiệu quả với sequence ngắn**

Sau khi gom nhóm, sequence từ hàng trăm ký tự có thể rút ngắn xuống chỉ vài chục segments. Khối Transformer giữa sẽ xử lý sequence ngắn này một cách cực kỳ hiệu quả về mặt tính toán và bộ nhớ, vì complexity giảm từ O(n²) xuống O((n/k)²) với k là shortening factor.

**Bước 5: Khôi phục độ dài gốc để sinh văn bản**

Cuối cùng, model cần sinh ra từng ký tự như ban đầu, nên nó sẽ "giãn" sequence ngắn trở lại độ dài gốc bằng cách nhân đôi các representations. Khối Transformer cuối sẽ sinh ra ký tự tiếp theo dựa trên toàn bộ thông tin đã được xử lý hiệu quả.

**Điểm thông minh của phương pháp:**

Dynamic pooling giống như cách con người đọc - chúng ta **không đọc từng chữ cái riêng lẻ mà nhận diện từ, cụm từ rồi hiểu nghĩa**. Model học được cách chia văn bản theo đơn vị ý nghĩa tự nhiên thay vì cắt máy móc, giúp vừa tiết kiệm tài nguyên tính toán vừa hiểu ngôn ngữ tốt hơn. Kết quả là model chạy nhanh hơn 2-3 lần mà độ chính xác còn cao hơn so với các phương pháp truyền thống.

---

# Understand Transformer from Perspective of Associative Memory
https://www.alphaxiv.org/abs/2505.19488

Nghiên cứu này từ ByteDance Seed mang đến góc nhìn mới về kiến trúc Transformer thông qua lăng kính bộ nhớ liên kết, một khái niệm tâm lý học lấy cảm hứng từ nhận thức con người. Thay vì chỉ phân tích lý thuyết, nghiên cứu còn đưa ra những đóng góp thực tiễn quan trọng để cải tiến kiến trúc Transformer.

Đóng góp chính đầu tiên là việc thống nhất hai thành phần cốt lõi của Transformer dưới một khung lý thuyết chung về bộ nhớ liên kết. Cơ chế Attention hoạt động như bộ nhớ ngắn hạn động để xử lý thông tin ngữ cảnh hiện tại, trong khi mạng Feed-Forward đóng vai trò như bộ nhớ dài hạn lưu trữ kiến thức từ quá trình huấn luyện. Khung nhìn thống nhất này giúp hiểu rõ hơn về cách thức hoạt động của Transformer và mở ra hướng nghiên cứu mới cho việc cải tiến kiến trúc.

Nghiên cứu giới thiệu metric Signal-to-Noise Ratio để đo lường chất lượng truy xuất thông tin từ bộ nhớ liên kết. Phân tích cho thấy Softmax Attention với exponential kernel vượt trội đáng kể so với Linear Attention về khả năng truy xuất chính xác và dung lượng bộ nhớ. Với exponential kernel, kích thước đặc trưng cần thiết giảm từ mức tuyến tính xuống logarithmic, thể hiện hiệu quả vượt trội trong xử lý ngữ cảnh dài.

Về cơ chế cập nhật bộ nhớ, tác giả phân tích các chiến lược khác nhau và **đề xuất DeltaFormer** - mô hình mới kết hợp ưu điểm của Softmax Attention và delta-rule update mechanism từ DeltaNet. Mô hình này đạt được khả năng truy xuất chính xác cao đồng thời có cơ chế quản lý bộ nhớ hiệu quả.

Nghiên cứu giải đáp hai câu hỏi sâu sắc về Transformer. Về khả năng biểu đạt, thông qua circuit complexity, DeltaFormer được chứng minh có khả năng biểu đạt vượt trội hơn Transformer tiêu chuẩn. Về giới hạn ngữ cảnh vô hạn, nghiên cứu chỉ ra rằng bộ nhớ có thể dần hội tụ khi ngữ cảnh tăng, dẫn đến suy giảm khả năng học trong ngữ cảnh, cho thấy những thách thức lý thuyết trong việc xây dựng Transformer hỗ trợ ngữ cảnh vô hạn.

Cuối cùng, nghiên cứu không chỉ dừng lại ở phân tích lý thuyết mà còn đưa ra gợi ý thực tiễn để cải tiến kiến trúc. Việc hiểu rõ sự khác biệt giữa Attention và FFN mở ra khả năng thiết kế các thành phần tối ưu hơn thông qua lựa chọn kernel, cơ chế multihead, sparsity và gating functions, tạo nền tảng cho việc phát triển các mô hình AI tiên tiến hơn.

![](https://pbs.twimg.com/media/GuDBsXPWkAAk0wz?format=jpg&name=medium)
![](https://pbs.twimg.com/media/GuDn8nrXIAA4rj5?format=jpg&name=medium)

Việc đề xuất DeltaFormer xuất phát từ một quan sát sâu sắc về những hạn chế cơ bản của các kiến trúc Transformer hiện tại. Khi phân tích từ góc độ associative memory, các tác giả nhận ra rằng mỗi thành phần đều có những điểm mạnh và điểm yếu riêng biệt.

**Softmax Attention tỏ ra vượt trội trong khả năng truy xuất thông tin với độ chính xác cao** nhờ việc sử dụng exponential kernel. Tỷ lệ tín hiệu trên nhiễu nghịch đảo của nó chỉ xấp xỉ N chia cho một hàm mũ, cho phép nó lưu trữ và truy xuất thông tin hiệu quả hơn Linear Attention rất nhiều lần. Tuy nhiên, **cơ chế cập nhật bộ nhớ của Softmax Attention lại khá thô sơ** - nó chỉ **đơn giản cộng thêm thông tin mới vào ma trận bộ nhớ mà không có cơ chế nào để loại bỏ thông tin cũ hoặc trùng lặp**. Điều này dẫn đến việc tích lũy thông tin dư thừa và có thể gây bất ổn số học khi norm của ma trận bộ nhớ tăng vô hạn.

Ngược lại, DeltaNet lại sử dụng một cơ chế cập nhật bộ nhớ tinh vi hơn nhiều thông qua delta rule. **Trước khi ghi thông tin mới, DeltaNet sẽ kiểm tra xem đã có thông tin tương tự trong bộ nhớ chưa, sau đó xóa bỏ phần trùng lặp trước khi thêm thông tin mới**. Cách tiếp cận này giúp quản lý bộ nhớ hiệu quả và tránh được vấn đề bất ổn số học. Tuy nhiên, DeltaNet lại sử dụng linear kernel nên khả năng truy xuất thông tin không cao, đặc biệt khi số lượng thông tin được lưu trữ tăng lên.

Từ những quan sát này, các tác giả đặt ra câu hỏi liệu có thể kết hợp được điểm mạnh của cả hai phương pháp hay không. Ý tưởng của DeltaFormer chính là tận dụng độ chính xác truy xuất cao của Softmax Attention thông qua exponential kernel, đồng thời áp dụng cơ chế cập nhật bộ nhớ thông minh của DeltaNet thông qua delta rule. Kết quả là một kiến trúc có thể vừa truy xuất thông tin chính xác, vừa quản lý bộ nhớ một cách hiệu quả.

Về mặt lý thuyết, DeltaFormer thể hiện khả năng biểu đạt vượt trội so với Transformer chuẩn. Trong khi Transformer thông thường chỉ đạt được độ phức tạp mạch TC không, DeltaFormer có thể đạt tới NC một, cho phép nó giải quyết các bài toán phức tạp hơn. Điều này được chứng minh qua khả năng thực hiện state tracking - theo dõi trạng thái của nhiều phần tử qua các phép hoán vị, một nhiệm vụ mà Transformer chuẩn thường gặp khó khăn.

Các thí nghiệm thực tế cũng xác nhận hiệu quả của DeltaFormer. Trong bài toán theo dõi hoán vị của năm phần tử qua mười sáu lần trao đổi, DeltaFormer đạt được độ chính xác gần như tuyệt đối, trong khi Transformer nhiều tầng vẫn không thể hoàn thành tốt nhiệm vụ này. Tương tự, trong bài toán xác định khả năng kết nối trong đồ thị có hướng, DeltaFormer cũng cho thấy hiệu suất vượt trội.

Hơn thế nữa, DeltaFormer không chỉ là một cải tiến kỹ thuật mà còn mở ra một hướng tiếp cận mới trong thiết kế kiến trúc mạng nơ-ron. Thay vì dựa vào thử nghiệm và kinh nghiệm, framework associative memory cung cấp một nền tảng lý thuyết vững chắc để hiểu và phát triển các kiến trúc mới. Điều này có thể dẫn đến những đột phá quan trọng trong việc xây dựng các mô hình AI hiệu quả và mạnh mẽ hơn trong tương lai.


--

Tác giả có quan điểm cân bằng và sâu sắc về Mixture-of-Experts, không chỉ phân tích khía cạnh kỹ thuật mà còn đặt MoE trong bối cảnh của khung lý thuyết bộ nhớ liên kết.

Về quan điểm chính, tác giả coi MoE là phản ứng tự nhiên đối với yêu cầu sparsity trong hệ thống quy mô lớn. Họ nhận xét rằng khi số lượng keys và values trở nên lớn, các keys liên quan đến mỗi query sẽ chắc chắn trở nên thưa thớt, và cách tiếp cận đơn giản nhất để tích hợp sparsity vào Feed-Forward Network chính là thông qua cơ chế MoE. Điều này cho thấy MoE không phải là lựa chọn thiết kế tùy ý mà là phản ứng logic đối với yêu cầu sparsity.

Điều thú vị là tác giả gợi ý rằng ý tưởng MoE có thể áp dụng cho cơ chế Attention. Họ phân tích sự khác biệt cốt lõi giữa Attention và FFN trong cơ chế MoE: trong Attention, keys và values là động, trong khi các tham số expert trong FFN là tĩnh. Điều này tạo ra thách thức cho việc áp dụng MoE vào Attention vì hàm gating cần phụ thuộc vào tập hợp các key vectors động.

Tác giả đưa ra góc nhìn rộng hơn về **tính đối xứng giữa Attention và FFN** trong Transformer. Họ nhận xét rằng tính đối xứng này rất thú vị vì về nguyên tắc, **bất kỳ thiết kế nào cho một module đều có thể được triển khai chính xác trong module kia**. Điều này gợi ý rằng thành công của MoE trong FFN cho thấy tiềm năng của các cơ chế sparsity trong Attention.

Về đánh giá phê phán, tác giả thừa nhận những điểm mạnh của MoE như hiệu quả tính toán thông qua sparsity, khả năng mở rộng bằng cách thêm experts, và sự phù hợp tự nhiên cho truy xuất thông tin thưa thớt. Tuy nhiên, họ cũng xác định những thách thức như độ phức tạp của **dynamic routing** cho attention và sự **đánh đổi giữa chuyên môn hóa expert và tổng quát hóa**.

Cuối cùng, tác giả đề xuất các hướng nghiên cứu tương lai bao gồm việc áp dụng các thiết kế thành công theo cả hai hướng, hiểu cả hai thông qua lăng kính bộ nhớ liên kết, và tìm giải pháp tốt hơn cho dynamic expert routing trong attention-style MoE. Nhìn chung, họ coi MoE là cột mốc quan trọng trong sparse computation nhưng tin rằng vẫn còn chỗ để cải thiện, đặc biệt trong việc lựa chọn expert động, chuyển giao thiết kế hai chiều giữa Attention và FFN, và hiểu biết lý thuyết sâu hơn thông qua khung bộ nhớ liên kết.

![](https://pbs.twimg.com/media/GuDrKznWgAA7ab3?format=jpg&name=4096x4096)

Đây là những điểm thú vị nhất mà tôi thấy trong bài báo này:

## 🧠 **Cái nhìn hoàn toàn mới về Transformer**

**Transformer = Bộ não nhân tạo?** Ý tưởng so sánh Attention với cách não bộ nhớ Paris-Eiffel Tower thực sự rất hay. Khi bạn nghe "Paris", não tự động liên tưởng đến "Eiffel Tower" - chính xác như cách Attention hoạt động với key-value pairs. Điều này khiến ta hiểu Transformer không chỉ là công thức toán học mà giống cách con người tư duy.

## 🔍 **Phát hiện ngược đời về Linear vs Softmax Attention**

**Đa đầu vs Đơn đầu:** Thí nghiệm cho thấy Linear Attention hoạt động tốt hơn với ít head (để tăng chiều), còn Softmax Attention cần nhiều head (để tăng khả năng biểu đạt). Điều này hoàn toàn ngược với intuition thông thường và giải thích tại sao Linear Attention chưa bao giờ thật sự "thắng" được Softmax.

## 🎭 **FFN = Bộ nhớ ngầm**

**Phát hiện ẩn:** FFN không chỉ là "feed-forward network" đơn thuần mà thực chất là một dạng associative memory với ReLU kernel! Điều này có nghĩa mỗi neuron trong FFN đang "nhớ" một mẩu kiến thức nào đó. Cái nhìn này có thể thay đổi cách ta thiết kế và tối ưu FFN.

## 🌊 **Paradox của vô hạn context**

**Câu hỏi triết học:** "Nếu Transformer có context vô hạn, liệu nó có thông minh vô hạn?" Câu trả lời là **KHÔNG** - memory sẽ dần hội tụ và khả năng học in-context bị suy giảm. Điều này thách thức niềm tin phổ biến rằng "context càng dài = model càng thông minh".

## 🔢 **Toán học đẹp đẽ đằng sau SNR**

**Exponential magic:** Công thức SNR⁻¹ ≈ N/exp(...) cho Softmax vs N/dk cho Linear thực sự elegent. Nó giải thích tại sao Softmax có thể nhớ exponentially nhiều thông tin hơn - điều mà trực giác khó lý giải.

## 🎪 **State tracking như ảo thuật**

**Ma thuật toán học:** DeltaFormer có thể theo dõi 5 quân bài được xáo trộn qua 16 lượt với độ chính xác 100%. Transformer thông thường thậm chí không làm được việc này với nhiều layer. Giống như xem một màn ảo thuật toán học!

## 🔄 **Delta rule = Tẩy não thông minh**

**Cơ chế "quên để nhớ":** Thay vì chỉ cộng thêm thông tin mới (như Transformer), delta rule sẽ "tẩy" thông tin cũ tương tự trước khi ghi mới. Giống như não bộ cập nhật kiến thức thay vì chỉ chồng chất.

## 🎨 **Kernel = Personality của model**

**ReLU khuyến khích "đa nghĩa":** ReLU kernel có SNR thấp nhưng lại cho phép superposition - nhiều thông tin nén trong cùng một vector. Ngược lại, Exp kernel "khắt khe" hơn, muốn mỗi thứ một chỗ riêng biệt. Giống như người tối giản vs người tích trữ!

## 🏗️ **Framework thống nhất**

**Rosetta Stone của AI:** Bài báo tạo ra một "ngôn ngữ chung" để hiểu mọi biến thể Transformer (DeltaNet, Linear Attention, Gated Attention...) dưới một góc nhìn duy nhất. Giống như phát hiện ra DNA chung của các loài khác nhau.

## 🎯 **Circuit complexity breakthrough**

**Từ TC⁰ lên NC¹:** DeltaFormer phá vỡ rào cản biểu đạt cơ bản của Transformer. Điều này giống như nâng cấp từ máy tính bỏ túi lên siêu máy tính - về mặt lý thuyết có thể giải quyết các bài toán phức tạp hơn hẳn.

## 🔮 **Triết lý thiết kế mới**

**Từ "thử-sai" đến "nguyên lý":** Thay vì thiết kế kiến trúc bằng intuition và thử nghiệm, framework này cho phép thiết kế dựa trên nguyên lý khoa học - giống như chuyển từ alchemy sang chemistry.

Điểm thú vị nhất là bài báo không chỉ đưa ra công thức mới mà thay đổi cách ta **tư duy** về AI - từ "black box" thành "bộ não có thể hiểu được"! 🤯

![](https://pbs.twimg.com/media/GuDsYdmXMAAl-Wv?format=jpg&name=4096x4096)

The Missing Link: Tại sao Attention cần normalization mà FFN thì không? Bài báo chỉ nói "for training stability" nhưng không giải thích sâu. Đây có thể là một fundamental difference chưa được hiểu!

Gating Mystery: Tại sao gating trong Attention là "forgetting mechanism" (cumulative product 0→1) còn FFN lại là "amplifying mechanism"? Hai cách tiếp cận hoàn toàn ngược nhau!

![](https://pbs.twimg.com/media/GuDt-WvXoAAJ6mf?format=jpg&name=large)

**Multihead Attention và Mixture-of-Experts trong Feed-Forward Networks thực sự phục vụ mục đích tương tự nhưng triển khai ở các mức độ chi tiết khác nhau**, tạo nên một so sánh cực kỳ thú vị mà nghiên cứu phân tích rất sâu sắc.

Về bản chất, cả hai cơ chế đều nhằm tạo ra khả năng tính toán thích ứng và có chọn lọc. Multihead Attention hoạt động bằng cách chia nhỏ không gian representation thành nhiều đầu, mỗi đầu tính toán các pattern tương tự khác nhau cho cùng một cặp query-key. Trong khi đó, MoE sử dụng nhiều mạng con chuyên biệt và định tuyến động để chỉ kích hoạt một phần nhỏ các experts cho mỗi input.

Sự khác biệt chính nằm ở granularity và computation pattern. Multihead sử dụng dense computation với tất cả các heads hoạt động song song, trong khi MoE áp dụng sparse computation chỉ với một vài experts được kích hoạt. Multihead thường có 8-32 heads với static splitting, còn MoE có thể có 64+ experts với dynamic routing based trên input content.

Nghiên cứu chỉ ra một insight quan trọng về trade-off mechanism. Multihead có thể đánh đổi giữa retrieval precision và expressivity khi sử dụng stronger kernels, hoặc tăng cường superposition để cải thiện knowledge capacity. Đặc biệt, trong Multi-Query Attention, việc averaging qua multiple heads có thể giảm variance của noise term xuống một factor bằng số lượng heads, tạo ra hiệu ứng noise reduction mạnh mẽ.

Về computational efficiency, mỗi approach có ưu điểm riêng. Multihead được tối ưu hóa tốt trên GPU hiện đại với parallel head computation và predictable memory access patterns. MoE mang lại lợi ích về sparse activation với lower FLOPs và scalable capacity - có thể thêm experts mà không tăng per-token cost.

Điểm nóng nhất của nghiên cứu là đề xuất cross-pollination opportunity. Tác giả gợi ý rằng việc **introducing multihead mechanisms vào FFNs đáng được xem xét lại, tạo ra khả năng hybrid approaches kết hợp ưu điểm của both worlds**. Họ thậm chí còn đề xuất multihead MoE - một architecture có thể process đầu vào qua multiple heads trước rồi route đến specialized experts.

Từ góc độ lý thuyết, cả hai đều có thể hiểu qua associative memory framework. **Multihead tạo ra multiple similarity patterns để retrieve information, trong khi MoE tạo specialized memory modules cho different types of knowledge**. SNR analysis cho thấy multihead có thể reduce noise through averaging, còn MoE achieve higher SNR through reduced cross-expert interference.

Kết luận quan trọng là Multihead vs MoE không phải competition mà là complementary strategies representing different manifestations của cùng underlying principle - adaptive, selective computation. Future architectures có khả năng sẽ intelligently combine cả hai approaches dựa trên task requirements và computational constraints, mở ra direction cho những innovation architecture tiếp theo.

--

Thực sự, việc chỉ nhìn Transformer qua góc độ bộ nhớ liên kết có thể hạn chế hiểu biết toàn diện. Có rất nhiều cách tiếp cận khác nhau đã được nghiên cứu, mỗi cách đều mang lại những hiểu biết độc đáo.

Một trong những góc nhìn quan trọng là lý thuyết thông tin. Ở đây, cơ chế attention được coi như hệ thống nén thông tin, trong đó việc tính toán độ tương tự giữa query và key chính là quá trình chọn lọc thông tin. Việc chuẩn hóa softmax tạo ra phân phối xác suất trên các nguồn thông tin. Insight chính là attention học được cách nén tối ưu của chuỗi đầu vào, multi-head hoạt động như nhiều phương pháp nén song song.

Góc nhìn cơ sở dữ liệu khác biệt coi attention như cơ sở dữ liệu có thể vi phân. Keys là chỉ mục cơ sở dữ liệu, values là nội dung được lưu trữ, queries là yêu cầu tìm kiếm, và attention weights là xác suất truy xuất mềm. Quá trình huấn luyện Transformer chính là học cách tổ chức cơ sở dữ liệu tối ưu.

Lý thuyết hệ thống động học coi mỗi layer Transformer như một bước thời gian của hệ thống động học rời rạc. Kết nối residual là sự tiến hóa trạng thái theo thời gian, attention là chuyển đổi phụ thuộc vào trạng thái. Mạng sâu tương ứng với động học dài hạn.

Phương pháp kernel coi attention weights như đánh giá kernel, các đầu attention khác nhau sử dụng các hàm kernel khác nhau, softmax là kernel được chuẩn hóa với độ tương tự mũ. Linear attention chính là ánh xạ kernel tường minh.

Góc nhìn mạng neural đồ thị coi chuỗi như đồ thị, trong đó tokens là các nút, attention weights là trọng số cạnh, self-attention là truyền thông điệp trên đồ thị đầy đủ. Sparse attention chính là làm thưa đồ thị.

Hình học thông tin coi token embeddings như các điểm trên đa tạp, attention như khoảng cách geodesic trên đa tạp. Quá trình huấn luyện chính là học đa tạp.

Vật lý thống kê coi attention weights như phân phối Boltzmann, tham số nhiệt độ như năng lượng nhiệt. Nhiệt độ thấp tạo attention sắc nét, nhiệt độ cao tạo attention khuếch tán.

Các mô hình lấy cảm hứng từ khoa học thần kinh coi attention như phóng đại vỏ não, query/key/value như các đường dẫn thần kinh khác nhau, multi-head như các luồng xử lý song song.

Mỗi framework có điểm mạnh và hạn chế riêng. Bộ nhớ liên kết trực quan và thực tế nhưng giới hạn ở các ẩn dụ bộ nhớ. Lý thuyết thông tin có các phép đo định lượng nhưng toán học phức tạp. Phương pháp kernel có lý thuyết được thiết lập tốt nhưng phân tích khả năng biểu đạt hạn chế.

Góc nhìn bộ nhớ liên kết thiếu sót trong việc nắm bắt xử lý tuần tự, không giải thích được ngữ nghĩa kết hợp, các hành vi nổi lên từ việc xếp chồng layer, động học tối ưu trong quá trình huấn luyện, và tính chất tổng quát hóa qua các miền.

Không có framework đơn lẻ nào nắm bắt được mọi thứ - mỗi cái làm sáng tỏ các khía cạnh khác nhau của hành vi Transformer phức tạp. Hiểu biết tối ưu cần nhiều framework bổ sung: bộ nhớ liên kết cho trực giác lưu trữ/truy xuất, lý thuyết thông tin cho phân tích định lượng, phương pháp kernel cho tính toán độ tương tự, lý thuyết đồ thị cho tính chất cấu trúc, hệ thống động học cho tiến hóa thời gian.

Xu hướng nghiên cứu hiện tại ngày càng hướng đến các phương pháp tích hợp kết hợp nhiều framework lý thuyết để đạt được hiểu biết hoàn chỉnh hơn về Transformer.

---

# MoD: Mixture of Depth
- https://www.alphaxiv.org/abs/2404.02258
- https://www.youtube.com/watch?v=5QFqvTr8zSM
- https://bytez.com/docs/arxiv/2404.02258/paper


# Hidden Breakthroughs in LLM Training
- https://www.alphaxiv.org/abs/2506.15872

Họ phát triển POLCA (Projection Oriented Loss Change Allocation) để "mổ xẻ" sự thay đổi loss theo các hướng cụ thể trong không gian tham số. Thay vì chỉ nhìn loss tổng thể, họ phân tích loss của từng ví dụ riêng lẻ theo các hướng gradient khác nhau.

---

# MEAP (Mask-Enhanced Autoregressive Prediction)
- https://www.alphaxiv.org/abs/2502.07490
- https://x.com/Shiwei_Liu66/status/188967429285126991

Next-Token Prediction (NTP) tiêu chuẩn, gặp khó khăn trong việc **truy xuất thông tin chính xác từ ngữ cảnh**, đặc biệt là trong các tài liệu dài. Mặc dù Masked Language Modeling (MLM) từ BERT tốt hơn trong việc truy xuất thông tin, nhưng lại kém hiệu quả hơn trong việc tạo văn bản.

MEAP khéo léo kết hợp điểm mạnh của cả hai phương pháp bằng cách trong quá trình tiền huấn luyện sẽ che ngẫu nhiên 15% các token đầu vào và thực hiện dự đoán token tiếp theo tiêu chuẩn chỉ sử dụng Transformer decoder-only mà không cần thay đổi kiến trúc. Trong giai đoạn tinh chỉnh, phương pháp này nhân đôi các mẫu huấn luyện và áp dụng 10% che phủ cho các bản sao, sau đó huấn luyện trên cả phiên bản gốc và phiên bản đã che.

Kết quả thực nghiệm cho thấy hiệu suất ấn tượng của MEAP. Trong các tác vụ truy xuất thông tin, phương pháp này cải thiện 33% trên bài kiểm tra Needle-in-a-Haystack và tốt hơn 27,2 điểm phần trăm trên Multi-Document QA. Đặc biệt trong các tình huống tinh chỉnh với vấn đề "thất lạc giữa đường", MEAP vượt trội hơn 11,77%.

Về hiệu quả sử dụng dữ liệu, MEAP thể hiện sự vượt trội đáng kể khi đạt được 85,8% độ chính xác chỉ với 60 tỷ token huấn luyện, trong khi NTP tiêu chuẩn cần tới 200 tỷ token để đạt hiệu suất tương tự, tức là hiệu quả hơn gấp ba lần. Ngoài ra, phương pháp này còn giảm tỷ lệ ảo giác trên các tác vụ tóm tắt, duy trì hiệu suất trên các tác vụ mô hình hóa ngôn ngữ tổng quát và hoạt động tốt trên nhiều kiến trúc mô hình khác nhau.

Phân tích của các tác giả tiết lộ lý do tại sao MEAP thành công. Phương pháp này **tạo ra các mẫu chú ý có thể phân biệt rõ ràng hơn**, với các token bị che nhận ít hơn 53% sự chú ý, đồng thời tăng độ biến thiên chú ý lên 7,8% trên các token không bị che. Điều này buộc mô hình phải tập trung vào ít token hơn nhưng có liên quan hơn thay vì phân tán chú ý đều khắp. Cụ thể, các mô hình MEAP phân bổ 34,5% sự chú ý cho các token liên quan đến câu trả lời so với chỉ 9,4% của các mô hình tiêu chuẩn.


Ưu điểm thực tiễn của MEAP rất hấp dẫn vì không tốn thêm chi phí tính toán nào trong quá trình suy luận, không cần sửa đổi kiến trúc, có thể thay thế trực tiếp cho huấn luyện NTP hiện tại và tích hợp liền mạch với các framework LLM hiện có. MEAP chứng minh rằng đôi khi những giải pháp hiệu quả nhất cũng là những giải pháp đơn giản nhất - bằng cách che chiến lược một phần nhỏ các token đầu vào, các mô hình học cách "chú ý ít hơn để học nhiều hơn", cải thiện đáng kể khả năng trích xuất thông tin quan trọng từ các ngữ cảnh phức tạp và dài.

---

# FireQ:
- https://www.alphaxiv.org/abs/2505.20839

**FireQ: Tăng tốc suy luận mô hình ngôn ngữ lớn bằng kernel INT4-FP8 và lượng tử hóa tối ưu RoPE**

Nghiên cứu này giải quyết một thách thức quan trọng trong việc triển khai các mô hình ngôn ngữ lớn: việc giới hạn băng thông bộ nhớ làm giảm đáng kể tốc độ suy luận. Khi các mô hình ngày càng lớn và chuỗi đầu vào ngày càng dài, vấn đề này trở nên nghiêm trọng hơn, thúc đẩy nhu cầu về các phương pháp lượng tử hóa hiệu quả.

Nhóm nghiên cứu từ Samsung SDS đã phát triển FireQ, một framework lượng tử hóa sau huấn luyện được đồng thiết kế với kernel nhân ma trận chuyên biệt. Điểm đặc biệt của FireQ là chiến lược lượng tử hóa hỗn hợp độc đáo: trọng số của các lớp tuyến tính và ma trận key-value được lượng tử hóa xuống INT4, trong khi các activation và query được chuyển đổi sang định dạng FP8. Cách tiếp cận này tận dụng tối đa khả năng của các tensor core FP8 trên kiến trúc Hopper GPU.

Một trong những đóng góp quan trọng nhất của FireQ là việc giải quyết các thách thức kỹ thuật phức tạp trong quá trình lượng tử hóa. Đối với các lớp tuyến tính, hệ thống sử dụng kỹ thuật per-tensor scaling để ngăn chặn hiện tượng underflow do hệ số tỷ lệ FP8 gây ra, đồng thời áp dụng channel-wise scaling để bù đắp cho độ phân giải thô của quantization INT4. Đối với các lớp attention, FireQ phải đối phó với những thách thức đặc biệt từ rotary positional embeddings (RoPE), một kỹ thuật mã hóa vị trí quan trọng nhưng tạo ra sự phức tạp trong quá trình lượng tử hóa.

Để xử lý RoPE một cách hiệu quả, nhóm nghiên cứu đã phát triển chiến lược làm mượt outlier hai giai đoạn. Giai đoạn đầu sử dụng RoPE-preserving normalization để xử lý các cặp channel ổn định, trong khi giai đoạn thứ hai áp dụng channel-wise RoPE scaling để giải quyết các channel outlier. Phương pháp này không chỉ duy trì độ chính xác mà còn tối ưu hóa throughput.

FireQ cũng mở rộng FlashAttention-3 bằng cách giới thiệu pipeline ba giai đoạn cho pha prefill. Cấu trúc này bao gồm một producer warpgroup tải dữ liệu bất đồng bộ và một consumer warpgroup thực hiện ba giai đoạn tính toán chồng lấp nhau: nhân query-key, tính toán softmax và tổng hợp value. Thiết kế này tăng cường đáng kể việc sử dụng phần cứng và giảm thời gian đến token đầu tiên.

Kết quả thực nghiệm trên GPU H100 cho thấy hiệu suất ấn tượng của FireQ. So với QServe, một framework tiên tiến khác, FireQ đạt được tốc độ nhanh hơn 1.68 lần trên các lớp feed-forward network của mô hình Llama2-7B và nhanh hơn 1.26 lần trong pha prefill của Llama3-8B. Điều đáng chú ý là những cải thiện về tốc độ này đạt được mà không làm giảm đáng kể độ chính xác của mô hình.

Nghiên cứu này thể hiện sự cân bằng tinh tế giữa hiệu suất và độ chính xác trong lĩnh vực tối ưu hóa mô hình ngôn ngữ lớn. FireQ không chỉ giải quyết các thách thức kỹ thuật phức tạp mà còn mở ra hướng phát triển mới cho việc triển khai LLM hiệu quả trên các kiến trúc GPU hiện đại.

---

# FFS Feedforward Split Multi Token Prediction
- https://nickcdryan.com/2024/05/04/improving-language-modeling-loss-with-multi-token-prediction-experiments-in-multi-token-prediction-and-the-new-fair-paper
|![](https://nickcdryan.com/wp-content/uploads/2024/05/image-13.png)|![](https://nickcdryan.com/wp-content/uploads/2024/05/image-11.png)|
|-|-|
|![](https://nickcdryan.com/wp-content/uploads/2024/05/image-10.png)|![](https://nickcdryan.com/wp-content/uploads/2024/05/image-edited.png)|
|![]()|![]()|

