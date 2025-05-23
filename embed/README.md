# Efficient Tokenizer
_modded-nanogpt + tokenmonster = 40% speedup_
- https://x.com/alexjc/status/1881410039639863622
- https://huggingface.co/datasets/alexjc/fineweb-tokmon-10B/tree/main/english-28416-balanced
Tôi phát hiện có thể đạt điểm Common Sense tương đương nhưng nhanh hơn 40% bằng cách chuyển sang bộ từ vựng TokenMonster tùy chỉnh có cùng kích thước với GPT-2. Tuy nhiên, do validation loss giữa các bộ từ vựng khác nhau không thể so sánh trực tiếp, bộ từ vựng tôi dùng lúc đó không vượt trội hơn GPT-2 trên thang đo riêng của nó. Sau nhiều thử nghiệm để tận dụng lợi thế 40% này, tôi quyết định giảm kích thước từ vựng xuống hơn 40%. Vì **các mô hình đạt kỷ lục sử dụng embedding ở nhiều vị trí**, việc giảm số lượng token tạo ra sự khác biệt đáng kể về hiệu suất (giảm hơn 10% mỗi bước). Bộ từ vựng gồm 28_416 token được tạo ra bằng cách lọc các mục từ bộ từ vựng TokenMonster english-100256-balanced mặc định.
1. `lọc thủ công` các token dựa trên các quy tắc cứng đơn giản
2. `loại bỏ các token ít được sử dụng nhất`.
Các token còn lại có xu hướng nguyên tử hơn, và ít token tổng hợp kết hợp nhiều thành phần: ví dụ như từ và dấu câu. **Tôi tin rằng sự đơn giản tương đối của các token là yếu tố cho phép tăng tốc độ học lên 8%.**

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

---

## ModernBERT
https://arxiv.org/html/2412.13663v2

