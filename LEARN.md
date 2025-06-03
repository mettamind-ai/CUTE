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


## Train Dense, Infer MoE
- https://arxiv.org/html/2404.05567v1

https://icml.cc/media/icml-2024/Slides/35222_1r94S59.pdf

![](https://publish-p57963-e462109.adobeaemcloud.com/adobe/dynamicmedia/deliver/dm-aid--a07ea5e6-1d7f-4766-ac41-d955b3911795/figure-2-standard-moe-architecture-vs-arctic-4.png)
