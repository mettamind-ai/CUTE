## BPEasy
- Treat text data at the **byte-level first** --- convert to bytes before training.
- Always use a **regex-based split pre-tokenizer**.
```python
# example regex from GPT-4
r=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```
## Byte Latent Transformer
![](https://arxiv.org/html/2412.09871v1/extracted/6066458/assets/patching_types.png)
![](https://arxiv.org/html/2412.09871v1/x4.png)
![](https://arxiv.org/html/2412.09871v1/x5.png)
![](https://arxiv.org/html/2412.09871v1/x3.png)
![](https://arxiv.org/html/2412.09871v1/extracted/6066458/assets/patching.png)

---

## vec2vec: translate text embeddings across different spaces without any paired data or encoders
- https://x.com/rishi_d_jha/status/1925212069168910340
- **https://x.com/jxmnop/status/1925224618060587523**

## mixture of tokenizers
- https://x.com/omouamoua/status/1922934072730403228
- https://github.com/snimu/blog/tree/main/contents/mixture-of-tokenizers
- https://github.com/snimu/blog/blob/main/contents/mixture-of-tokenizers-math/article.md
- https://github.com/snimu/blog/blob/main/contents/mot-scaling/article.md

## Selftok: Discrete Visual Tokens
https://selftok-team.github.io/report

## CLIP-like models and VLMs from Meta 
https://x.com/gabriberton/status/1922542722558067079

---

- Learning Deep Representations of Data Distributions
  https://x.com/YiMaTweets/status/1924068626694598683
- https://x.com/_reachsumit/status/1924348175651135875
- qwen pre tknz https://x.com/YouJiacheng/status/1923712356229710319
- Self-Interpretation of LLM Embeddings https://x.com/TheAhmadOsman/status/1923294932845912344
- Demystifying Embedding Spaces using Large Language Models (2023) - arXiv:2310.04475
- Siêu vị trí trong không gian biểu diễn https://x.com/YizhouLiu0/status/1923210466198773867
- 