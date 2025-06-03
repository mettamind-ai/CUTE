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
