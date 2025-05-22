## BPEasy
- Treat text data at the **byte-level first** --- convert to bytes before training rather than using characters (Huggingface).
- Always use a **regex-based split pre-tokenizer**. This is a customisable regex that is applied to the text before training.
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
https://x.com/rishi_d_jha/status/1925212069168910340
