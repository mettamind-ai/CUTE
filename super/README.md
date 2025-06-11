# Super Token
1 visual token có rất nhiều thông tin (diễn đạt bằng nhiều text token)
Liệu có thể làm tương tự như visual encoder nhưng mà cho text? => **SUPER TEXT TOKEN**
1 cụm n text tokens giờ đc biểu diễn = 1 embedding vector thay vì n như trước =>
đó là biên giới hạn của ngôn ngữ con người.

Máy có thể khai thác cái này để thông minh hơn ví dụ: Con lai của hổ và sư tử miêu tả là 
`"1 con gì đó giống con hổ và con sư tử"` => từ mới `hổ sư` thay vì biểu diễn bằng 1 câu 
sẽ có 1 miền trong token embeddings biểu diễn cái này
miền đó nằm giữa trung điểm của vector hổ và vector sư tử

biển diễn cho input và ouput thì vẫn là tokens trong vocab (con chữ) 
để đảm bảo tính cross entropy loss như bình thường, 
nhưng khi vào model mình có thể **map con chữ thành concept vector** ...

---


## vec2vec: translate text embeddings across different spaces without any paired data or encoders
- https://x.com/rishi_d_jha/status/1925212069168910340
- **https://x.com/jxmnop/status/1925224618060587523**

## Masked Next Token Prediction (GPT-BERT / LLM2VEC)
<table>
  <tr>
    <td width="40%"><img src="https://arxiv.org/html/2410.24159v2/x1.png"/></td>
    <td width="60%"><img src="https://private-user-images.githubusercontent.com/12207571/319390512-48efd48a-431b-4625-8e0f-248a442e3839.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyNjYxNzEsIm5iZiI6MTc0ODI2NTg3MSwicGF0aCI6Ii8xMjIwNzU3MS8zMTkzOTA1MTItNDhlZmQ0OGEtNDMxYi00NjI1LThlMGYtMjQ4YTQ0MmUzODM5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDEzMjQzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTEzOTk1NWY4MWU5N2E3NWM1MjExMzkwZjNiMjlhMzQ5OWM1NzlhMGRhYzI4MmMwZmI5NWZhYzQ0ZGI2OThhMzkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.GXRq3cQnXiyD_vYF-4O9HagAM3Fcug7rxjtr8fvGoXU"/></td>
  </tr>
</table>

## mixture of tokenizers
- https://x.com/omouamoua/status/1922934072730403228
- https://github.com/snimu/blog/tree/main/contents/mixture-of-tokenizers
- https://github.com/snimu/blog/blob/main/contents/mixture-of-tokenizers-math/article.md
- https://github.com/snimu/blog/blob/main/contents/mot-scaling/article.md


LATENT
------

## Byte Latent Transformer
![](https://arxiv.org/html/2412.09871v1/extracted/6066458/assets/patching_types.png)
![](https://arxiv.org/html/2412.09871v1/x4.png)
![](https://arxiv.org/html/2412.09871v1/x5.png)
![](https://arxiv.org/html/2412.09871v1/x3.png)
<!-- ![](https://arxiv.org/html/2412.09871v1/extracted/6066458/assets/patching.png) -->

## Chồng chập mọi thú, đầu vào đa độ phân giải và tổng hợp dự đoán từ các lớp trung gian để tránh tấn công đối kháng
https://www.alphaxiv.org/abs/2408.05446 sử dụng intermediate layer features như một dạng latent representations - những feature này ít bị ảnh hưởng bởi adversarial attacks so với final layer, cho phép tạo ra self-ensemble robust hơn.

**Backbone được đóng băng**: Tác giả sử dụng mô hình pretrained (như ResNet152 trên ImageNet) và đóng băng toàn bộ backbone. Chỉ có layer đầu tiên (để nhận 12 kênh) và layer cuối cùng được fine-tune.

**Linear probes độc lập**: Sau đó, họ huấn luyện riêng biệt các linear probes (head tuyến tính) trên activation của từng lớp trung gian. Tác giả mô tả: "we fix a trained network f : X → y and use its intermediate layer activations h₁(X), h₂(X), ··· , hₗ(X) to train separate trained linear probes (affine layers) that map the activation of the layer l into classification logits".

---

Trong LLM, latent representation biến đổi từ cụ thể → trừu tượng qua các layers:
Ví dụ câu "Hôm nay là một ngày đẹp trời" trong BERT

- Layer đầu (1-4):
  Token position, grammar, syntax
  "đẹp" biết nó là tính từ, đứng trước "trời"

- Layer giữa (5-8):
  Semantic relationships, coreference
  "đẹp trời" hiểu là weather condition

- Layer cuối (9-12):
  High-level reasoning, task-specific
  Toàn bộ câu → sentiment positive, talking about weather

Mỗi layer làm giàu thêm representation bằng attention mechanism - kết hợp thông tin từ các tokens khác.

Với GPT:
- layers cuối mạnh về NTP, nó vẫn phải hiểu toàn bộ câu nhưng
  hiểu theo hướng what comes next hơn là what does this mean
- "Các lớp giữa học cách văn bản tự xây dựng dần dần"
- Các lớp đầu giống BERT?

"Transformer Feed-Forward Layers Are Key-Value Memories" (Geva et al., 2021)
"Locating and Editing Factual Knowledge in GPT" (Meng et al., 2022)

---

# Dynamic Latent Representation: Kiến Trúc Tự Tổ Chức cho Multimodal AI

## Core Insight: "Tòa Tháp với Nền Móng Động"

### 1. Vấn đề Multimodal
- Input đa dạng (text/image/audio/video) cần không gian biểu diễn lớn
- Paradox: Cả diversity (input) và abstraction (high-level) đều cần nhiều dimensions

### 2. Kiến Trúc Đề Xuất: Dynamic Dimensional Allocation

**Nguyên lý:** Mỗi token/concept tự quyết định không gian cần thiết
- "chó" → 500 active dims
- "democracy" → 2000 active dims  
- Implement qua sparse activation hoặc attention gating

### 3. MoE như Self-Organizing Pathway

**Breakthrough:** MoE không chỉ là efficiency trick mà là cách token tự chọn representation
```
Token → Router → Expert Visual (2048d)
                → Expert Abstract (3072d)
                → Expert Factual (1024d)
```

### 4. Sinh học & Triết lý
- Parallel với não: specialized circuits cho different concepts
- Mỗi ý tưởng có "essence" riêng, cần xử lý riêng
- Emergence từ self-organization, không forced architecture

**Đề xuất nghiên cứu:** Phát triển "Self-Organizing Multimodal Transformer" với dynamic dimensions + hierarchical MoE cho Vietnamese AI advancement.

## BTL
- https://github.com/facebookresearch/blt
- https://www.alphaxiv.org/abs/2412.09871
![](https://pbs.twimg.com/media/Gshh4eaasAAfHL-?format=jpg)
![](https://pbs.twimg.com/media/Gshkw6easAA7Dzy?format=jpg)
![](https://pbs.twimg.com/media/GshlkjsbcAA70Th?format=png)
![](https://pbs.twimg.com/media/GshqAiObAAAOkQq?format=jpg)
![](https://pbs.twimg.com/media/GshsGeJasAMyBcr?format=jpg)

## EvaByte
- https://hkunlp.github.io/blog/2025/evabyte
- https://github.com/OpenEvaByte/evabyte
- Base model before annealing https://huggingface.co/EvaByte/EvaByte-Phase1

The main difference between BLTs and EvaByte lies in the architecture: BLTs use patchification and propose entropy patching to dynamically group bytes. While this approach adjusts compute allocation based on data complexity and reduces context length, it still relies on external models to determine patch boundaries. The majority of compute ends up focused on patch-level modeling, detached from the byte stream, similar to tokenizer-based models.

In contrast, EvaByte keeps things simple: it directly operates on bytes with a flat Transformer-like model without needing to invoke external modules or group inputs. Empirically, EvaByte achieves better performance than BLTs even with 3-4x fewer training bytes, as shown in the table below. Besides, EvaByte is more flexible and scales easily to multimodal data, while BLTs require retraining or swapping out the auxiliary language model used for entropy patching.

![](https://hkunlp.github.io/assets/img/2025-01-21-evabyte-imgs/comp_to_blt-1400.webp)

## ~~EVA (extended value aggregation) linearized attention~~ <= chưa hỗ trợ varlen
- playground/684138df4cd7dbf747d280d5
- https://github.com/OpenEvaByte/evabyte/blob/main/evabyte_hf/eva_agg_kernel.py
![](https://hkunlp.github.io/assets/img/2025-01-21-evabyte-imgs/arch-1400.webp)
![](https://hkunlp.github.io/assets/img/2025-01-21-evabyte-imgs/attn_sketch-1400.webp)

- Local window attention: Attention trong window cục bộ (như SWA)
- RFA chunks: Compressed representations cho global attention

---

# Logits, Vocab ...
- https://chatgpt.com/share/6847808b-2d30-8003-8500-75ef6485f961
  - logits distill: học phân phối, ko chỉ học nhãn đúng
  - hiệu chỉnh logits (calibration)
- https://chatgpt.com/share/6847ade6-0f7c-8003-82a4-4f418ae6101f

Một mô hình được gọi là **well-calibrated** nếu, ví dụ, trong tất cả các trường hợp nó dự đoán “đáp án A” với xác suất 80%, thì khoảng 80% những trường hợp đó đáp án A thật sự đúng. Tuy nhiên, nghiên cứu chỉ ra rằng các mô hình mạng nơ-ron hiện đại thường không được hiệu chỉnh tốt – chúng có xu hướng quá tự tin vào dự đoán của mình.

Vocab lớn cũng khiến phân phối xác suất đầu ra “loãng” hơn: xác suất được dàn trải trên nhiều khả năng. Ví dụ, ở một mô hình với |V|≈49k, xác suất cao nhất quan sát chỉ ~33.9%, các token khác chia nhau ~66% còn lại.

## n-gram embedding https://www.alphaxiv.org/abs/2501.16975v1

**Kết quả đáng chú ý**: "Using a large input vocabulary, we achieve performance comparable to double-sized baselines with no additional cost" - với từ vựng đầu vào lớn, mô hình 400M tham số đạt hiệu suất tương đương mô hình 1B tham số mà không tốn thêm chi phí ?!?

**mối quan hệ log-linear** giữa kích thước từ vựng đầu vào và training loss: "exponentially increasing the input vocabulary size consistently results in a linear decrease in loss". => Cái này dễ hiểu!

![](https://arxiv.org/html/2501.16975v1/x1.png)

KẾT LUẬN: OE SCALE TUYẾN TÍNH VÀ ỔN ĐỊNH, OD SCALE PHI TUYẾN VÀ PHỤ THUỘC KÍCH THƯỚC MÔ HÌNH.

## Larger Models Deserve Larger Vocabularies
- https://www.alphaxiv.org/overview/2406.16508 (Large Vocabulary Size Improves Large Language Models)
- https://www.alphaxiv.org/code/2407.13623

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F443f26f7-38c8-4e8a-85e6-a528b78f83a0_776x385.png)

Ý chính 3: Thực trạng hiện tại "Most LLMs, however, use insufficient vocabulary sizes. For example, we predict that the optimal vocabulary size of Llama2-70B should have been at least 216K, 7 times larger than its vocabulary of 32K." - Tuy nhiên, hầu hết các LLM sử dụng kích thước từ vựng không đủ. Ví dụ, chúng tôi dự đoán rằng kích thước từ vựng tối ưu của Llama2-70B lẽ ra phải ít nhất 216K, lớn hơn 7 lần so với từ vựng 32K hiện tại.

Scaling laws của vocabulary size mà bài báo phát hiện ra cho thấy kích thước từ vựng tối ưu có mối quan hệ toán học cụ thể với các thành phần khác của mô hình. Luật chính được thể hiện qua công thức `N_v^opt ∝ N_nv^γ` với `γ ≈ 0.83 < 1`, có nghĩa là **các tham số từ vựng nên được mở rộng chậm hơn các tham số phi từ vựng**.

Lý do đằng sau scaling law này là khi đã có không gian embedding đủ phong phú thông qua từ vựng lớn, việc mở rộng các tham số phi từ vựng để học các cấu trúc cú pháp và ngữ nghĩa phức tạp của ngôn ngữ trở nên quan trọng hơn. Do đó, vocabulary size không cần tăng tỷ lệ thuận hoàn toàn với kích thước mô hình mà chỉ cần tăng theo tỷ lệ γ < 1.

---

## STOCHASTOK
https://www.alphaxiv.org/overview/2506.01687

Mỗi seq đầu vào và xác xuất p.
Với p = 0.1 (default), nếu câu có 10 tokens thì sẽ expand 1 lần
Với câu có 20 tokens thì expand 2 lần

!!! Như vậy cũng có thể ngẫu nhiên merge 2 tokens lại để có được phiên bản 2-gram => TKNZ linh hoạt !!!


## Training free token transplantation via OMP (orthogonal matching pursuit)
- https://www.alphaxiv.org/abs/2506.06607

---

# ADAT: Adaptive Tokenizer
- https://proceedings.neurips.cc/paper_files/paper/2024/file/cdf00c97c0cb2cc35179f03363da6c4f-Paper-Conference.pdf
- https://www.alphaxiv.org/abs/2410.03258 ADAPTBPE sửa BPE để tknz tốt hơn cho domain
- https://www.alphaxiv.org/abs/2310.05317 TKNZ riêng cho task, => giảm 60% số tokens
- https://www.alphaxiv.org/abs/2109.07460 ...

Giải pháp – ADAT (Adaptive Tokenizer)

- Khởi tạo từ từ-vựng lớn.
- Huấn luyện LLM, tính loss từng token = hàm kết hợp tần suất & cross-entropy.
- Cắt bỏ token đóng góp thấp, lặp lại → tạo tokenizer “thích ứng” với mô hình.

=> !!! Có thể ADAT ngay trong lúc pre-train !!!

## Scaling LLM Pre-training with Vocabulary Curriculum
- DR https://chatgpt.com/s/dr_68499a6c2550819191a385cfb7d9bfee
- https://ar5iv.labs.arxiv.org/html/2502.17910
- https://www.alphaxiv.org/abs/2502.17910
- Entropy-Guided Vocabulary Updates
![](https://ar5iv.labs.arxiv.org/html/2502.17910/assets/better-scale-vocab-curriculum-1.png)

ban đầu mô hình học xử lý ký tự và các đơn vị nhỏ (giúp nắm chắc cấu trúc cơ bản), về sau dần “nâng cấp” lên các token lớn hơn cho những mẫu phổ biến. Yu và cộng sự cho biết cách làm này giúp mô hình GPT nhỏ đạt bpc (bits-per-character) thấp hơn ~6.7% so với mô hình dùng vocab cố định cùng kích thước. Hơn nữa, khi tăng gấp đôi kích thước vocab, mô hình thích ứng thu được hiệu quả cải thiện cao hơn ~34% so với mô hình truyền thống (tức là tận dụng vocab lớn tốt hơn). Kết quả cũng cho thấy một hệ thống phân cấp token tự nhiên hình thành: các token dài dần xuất hiện để đại diện cho các cụm từ phổ biến, dễ dự đoán, còn những đoạn nội dung khó dự đoán thì vẫn bị phân nhỏ thành token ngắn hơn để mô hình xử lý chi tiết. Điều này khớp với trực giác rằng tokenization động cho phép mô hình phân bổ tài nguyên tính toán hợp lý hơn – dành nhiều “não” hơn cho phần phức tạp, bớt tốn sức cho phần đơn giản.


---

# zip2zip
- https://www.alphaxiv.org/abs/2506.01084v1
a framework that enables LLMs to **dynamically adjust token vocabulary at inference time**, allowing for fewer generated tokens and thus faster inference. zip2zip consists of three key components:
- (1) a tokenizer based on LZW compression that incrementally compresses tokens into reusable "`hypertokens`" on the fly;
- (2) an embedding layer that computes embeddings for newly formed hypertokens at runtime; and
- (3) a causal language modeling variant that trains the model to operate on hypertokenized, compressed sequences.

We show that an **existing LLM can be zip2zip-fied in 10 GPU-hours via parameter-efficient finetuning**. The resulting zip2zip LLMs effectively learn to use hypertokens at inference time, reducing input and output sequence length by 20-60\%, with significant improvements in inference latency.

