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

---

# HIỆU CHUẨN LOGITS
- survey https://chatgpt.com/share/6848f5c0-5574-8003-83c3-710998a95115
- o3-pro https://chatgpt.com/share/68491c15-4508-8003-87ae-81410849f187

Hiệu chuẩn logit (logit calibration) đề cập đến việc điều chỉnh phân phối xác suất đầu ra của mô hình sao cho độ tự tin dự đoán của mô hình phù hợp với xác suất đúng thực tế. Nói cách khác, nếu mô hình dự đoán một câu trả lời với xác suất 70%, thì trong thực tế câu trả lời đó nên đúng khoảng 70% số lần trong tình huống tương tự. Một mô hình được gọi là được hiệu chuẩn tốt khi xác suất mô hình dự đoán phản ánh đúng xác suất câu trả lời đó thực sự chính xác.

![](https://pbs.twimg.com/media/GtIii5ibMAQ6aNp?format=png&name=large)
![](https://arxiv.org/html/2408.12168v1/extracted/5806746/Figures/top5.png)

Vocab lớn dẫn đến phân phối đầu ra rất sparse (thưa) đặt ra bài toán khó cho hiệu chuẩn. Bởi lẽ, mô hình có xu hướng quá tự tin vào vài lựa chọn hàng đầu (vì được huấn luyện để tối đa xác suất cho token đúng duy nhất) và quá thiếu tự tin cho số đông lựa chọn còn lại (vì hầu hết token bị đẩy xác suất về 0). Nếu token đúng nằm trong top đầu, mô hình có thể dự đoán đúng nhưng dễ overconfident về độ đúng của nó. Ngược lại, nếu câu trả lời đúng thực sự nằm ngoài top-5 hay top-10 (ví dụ mô hình bỏ sót), thì mô hình có thể đã gán xác suất cực thấp cho đáp án đúng đó – một trường hợp underconfidence nghiêm trọng (mô hình hoàn toàn không nhận thức được đáp án đúng tiềm tàng).

Thêm vào đó, số lượng lớp khổng lồ khiến các thước đo hiệu chuẩn truyền thống khó áp dụng trực tiếp. Chẳng hạn, metric ECE thường tập trung vào xác suất của lớp dự đoán cao nhất để so sánh với độ chính xác, nhưng với LLM, ta cũng quan tâm phân phối của các xác suất thấp (vì chúng quyết định mức độ “nghi ngờ” của mô hình với các phương án khác). Full-ECE là một độ đo mới được đề xuất nhằm tính lỗi hiệu chuẩn trên toàn bộ phân phối – về cơ bản xem xét khoảng cách giữa phân phối dự đoán và phân phối thực tế của tất cả các token, chứ không chỉ token đúng/sai.

**ta muốn không chỉ “xác suất token dự đoán” đúng cỡ nào, mà còn muốn phân phối xác suất của mọi token phản ánh tần suất xuất hiện thực sự (ví dụ các token hiếm cũng cần được gán xác suất phù hợp với mức độ hiếm của chúng, thay vì bị triệt tiêu hoàn toàn).**

Hết sức cẩn thận với Label Smoothing (Làm mịn nhãn) <= Điều này ám chỉ rằng label smoothing cần được điều chỉnh cẩn thận; trong một số trường hợp, hậu xử lý bằng temperature scaling sau huấn luyện có thể ổn định và hiệu quả hơn label smoothing.

**Ensemble và Bayesian methods**: Sử dụng nhiều mô hình dự đoán thay vì một thường giúp cải thiện hiệu chuẩn. Bằng cách lấy trung bình dự đoán của một tập hợp mô hình (deep ensemble) hoặc lấy mẫu dropout nhiều lần (Monte Carlo dropout), ta thu được phân phối xác suất “mềm” hơn, phản ánh độ bất định mô hình tốt hơn là dự đoán điểm từ một mô hình đơn lẻ. Các mô hình ensemble thường cho kết quả ít overconfident hơn, vì những lỗi quá tự tin của từng mô hình có xu hướng được trung bình hóa. Nghiên cứu đã chỉ ra ensemble có thể giảm ECE đáng kể so với một mô hình đơn, đặc biệt khi các mô hình thành phần đa dạng (ví dụ khởi tạo khác nhau) – nhờ đó tăng cường độ tin cậy của dự đoán cuối.

__NOTE__: Full‑ECE (2024) Xét toàn bộ xác suất token, không phụ thuộc binning, Tính toán nặng với vocab 50k+
=> Vocab 4k sẽ có lợi thế!

=> Nói cách khác, số lượng lựa chọn đầu ra tăng (ví dụ từ câu hỏi Yes/No sang A/B/C/D) có thể làm giảm độ hiệu chuẩn của mô hình lớn (vì mô hình bị “distract” – phân tán sự tự tin), trong khi mô hình nhỏ đôi khi lại được hưởng lợi khi có ít lựa chọn rõ ràng (giúp chúng định lượng độ tin cậy tốt hơn). Điều này gợi ý rằng mở rộng không gian đầu ra (nhiều lớp hơn) có thể tác động tiêu cực đến hiệu chuẩn, tùy thuộc vào quy mô và kiến trúc mô hình.

## Kỹ thuật hiệu chuẩn gắn vào huấn luyện (Train‑time)
| Phương pháp                             | Cơ chế                                          | Lợi ích                                          |
| --------------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| **LogitNorm**                           | Chuẩn hoá norm logit về hằng số trong loss      | Giảm over-confidence, OOD detection tốt hơn 42%  |
| **Entropy Minimization (EM-INF, 2025)** | Tối ưu logit trong lúc suy diễn để giảm entropy | Hiệu quả đặc biệt ở bài toán khó (AIME math…)    |

```py
# Pseudo‑code: T‑scale cho mô hình mở (PyTorch)
with torch.no_grad():
    val_logits = model(input_ids) # B × L × V
    val_labels = labels

# Optimize T
T = torch.nn.Parameter(torch.ones(1,device='cuda'))
optimizer = torch.optim.LBFGS([T], lr=0.01)
def _nll():
    loss = torch.nn.functional.cross_entropy(val_logits / T.exp(), val_labels)
    optimizer.zero_grad(); loss.backward(); return loss
optimizer.step(_nll)

# Inference
scaled_logits = logits / T.exp()
probs = torch.softmax(scaled_logits, dim=-1)
```

## Thách thức & Hướng mở
- Chuẩn hoá theo miền ứng dụng: Calibration cho câu trả lời “có/không” khác với toán học đa bước; đòi hỏi bộ dữ liệu chuyên dụng.
- Hiệu chuẩn đa phương thức: khi LLM nối với thị giác/âm thanh, logit không đồng nhất.
- Chi phí: consistency cần ≥ 20 mẫu, tốn GPU. Nghiên cứu nén xác suất (distillation) đang được thử nghiệm. https://arxiv.org/abs/2503.15850v2

| Phương pháp                | Granularity | Cần truy cập logit? | Hiệu quả ECE    | Chi phí              |
| -------------------------- | ----------- | ------------------- | --------------- | -------------------- |
| **Scalar T**               | Global      | ✔                   | Trung bình      | Rẻ nhất              |
| **Input‑dependent T**      | Instance    | ✔                   | Tốt (dạng ITS)  | Huấn luyện extra net |
| **ATS (token)**            | Token       | ✔                   | **Tốt nhất**    | +1 ms                |
| **Self‑Consistency**       | Sequence    | ✖                   | Phụ thuộc k mẫu | ×k lần gọi LM        |
| **LogitNorm (train‑time)** | Token       | ✔                   | Tốt + OOD       | Cần retrain          |

Hiệu chuẩn động (adaptive‑T, Adaptive Temperature Scaling – ATS) là biến thể tiên tiến của temperature‑scaling: thay vì dùng một hằng số T duy nhất, ATS học một hàm `T(x, h)` sinh ra **nhiệt độ riêng cho từng token** dựa trên đặc trưng ẩn của chính mô hình. Nhờ vậy, ATS khôi phục độ tin cậy đã suy giảm sau RLHF, giảm 10 – 50% lỗi ECE trên nhiều benchmark mà không làm mất độ chính xác 

Các LLM tiền‑huấn luyện vốn tương đối “trung thực” ở mức token, nhưng khi tiếp tục tinh chỉnh bằng supervised‑fine‑tune rồi RLHF, phân phối logit trở nên sắc nhọn, ECE tăng gấp 2–3 lần. Scalar temperature‑scaling khắc phục một phần, song một T duy nhất không bù được độ lệch khác nhau giữa câu hỏi dễ/khó, token thường/thấp tần, hay vị trí đầu/cuối câu.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
from ats import AdaptiveTemperatureWrapper   # repo của Xie @turn0search4

base = AutoModelForCausalLM.from_pretrained("mistral-7b")
tok  = AutoTokenizer.from_pretrained("mistral-7b")

model = AdaptiveTemperatureWrapper(base, hidden_size=4096, mlp_dim=512)
model.fit(calib_dataset, epochs=2, lr=5e-4)   # chỉ cập nhật head
# Suy diễn
out = model.generate("Why is the sky blue?", max_new_tokens=64)
```

---

## not all logits are you need
- https://alphaxiv.org/abs/2411.07641

![](https://pbs.twimg.com/media/GtJSDg5acAApkD4?format=jpg)

Bài báo giới thiệu phương pháp sampling mới top-nσ để cải thiện chất lượng sinh văn bản của các mô hình ngôn ngữ lớn. Ý tưởng cốt lõi là tác giả phát hiện ra rằng logits trong LLM có cấu trúc hai phần riêng biệt: một vùng nhiễu tuân theo phân phối Gaussian chứa các token không quan trọng, và một vùng thông tin chứa các token có giá trị. "Gaussian distributions often indicate the presence of random noise in a system" - đây là hiện tượng thống kê phổ biến khi có nhiều yếu tố ngẫu nhiên tác động. Quan trọng hơn, tác giả nhấn mạnh đây là khuyết điểm cố hữu của hàm softmax vì nó buộc phải gán giá trị hữu hạn cho tất cả token, kể cả những token không liên quan.

**top-nσ hoạt động trực tiếp trên logits và sử dụng ngưỡng thống kê để lọc token**. Phương pháp này có tính chất bất biến nhiệt độ - tập token được chọn không thay đổi khi điều chỉnh temperature.



## FIRST: Teach A Reliable Large Language Model Through Efficient Trustworthy Distillation
- https://alphaxiv.org/abs/2408.12168v1
