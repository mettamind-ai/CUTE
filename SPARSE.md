Logits bản chất là sparse!

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

## Top-nσ: Not All Logits Are You Need
- https://alphaxiv.org/abs/2411.07641

![](https://pbs.twimg.com/media/GtJSDg5acAApkD4?format=jpg)

Bài báo giới thiệu phương pháp sampling mới top-nσ để cải thiện chất lượng sinh văn bản của các mô hình ngôn ngữ lớn. Ý tưởng cốt lõi là tác giả phát hiện ra rằng logits trong LLM có cấu trúc hai phần riêng biệt: một vùng nhiễu tuân theo phân phối Gaussian chứa các token không quan trọng, và một vùng thông tin chứa các token có giá trị. "Gaussian distributions often indicate the presence of random noise in a system" - đây là hiện tượng thống kê phổ biến khi có nhiều yếu tố ngẫu nhiên tác động. Quan trọng hơn, tác giả nhấn mạnh đây là khuyết điểm cố hữu của hàm softmax vì nó buộc phải gán giá trị hữu hạn cho tất cả token, kể cả những token không liên quan khi inference.

**top-nσ hoạt động trực tiếp trên logits và sử dụng ngưỡng thống kê để lọc token**. Phương pháp này có tính chất bất biến nhiệt độ - tập token được chọn không thay đổi khi điều chỉnh temperature.


### khuyết điểm cốt lõi của softmax:

**"Noise of Silence"**: softmax buộc phải gán xác suất dương cho tất cả token trong vocabulary, kể cả những token hoàn toàn không liên quan đến ngữ cảnh. Lý tưởng thì những token này nên có logit = −∞ (xác suất = 0). Tác giả trích dẫn Miller (2023) và Xiao et al. (2023) cho rằng đây là "inherent flaw" - khuyết điểm cố hữu của hàm softmax, dẫn đến việc tạo ra "distinctive noise pattern" trong không gian logit.

Điều này giải thích tại sao phần lớn vocabulary tạo thành "vùng nhiễu" với phân phối Gaussian, trong khi chỉ một số ít token thực sự có ý nghĩa tạo thành "vùng thông tin".

## Phân phối của "vùng thông tin"
tác giả thừa nhận khó xác định chính xác do số lượng token ít: "Due to the limited number of tokens in this region, it is challenging to make definitive claims about the underlying distribution." Tuy nhiên, tác giả có phát hiện quan trọng thông qua phân tích `min-p sampling`:

**Theorem 3**: "For logits following a uniform distribution, min-p sampling is equivalent to `top-(1 − p)` sampling."

Từ đó tác giả kết luận: "Furthermore, the effectiveness of min-p sampling suggests that the informative region approximately follows a uniform distribution.". Tác giả giải thích thêm rằng "despite min-p's claimed adaptiveness, it essentially performs a static truncation in the logits space" - nghĩa là min-p thực chất chỉ cắt ngưỡng cố định trong không gian logit.

**Đặc điểm của vùng thông tin:**

- Số lượng token ít nhưng chiếm phần lớn probability mass
- Xu hướng `phân phối đều` (uniform distribution)
- Thường xuất hiện khi model có độ tin cậy cao hoặc ở nhiệt độ thấp
- Các token có logit cao hơn `mean + nσ` (thường n=1)

- Top-p/min-p: làm mềm toàn bộ → bao gồm cả noise
- Top-nσ + T: làm mềm chỉ trong vùng sạch

`top-nσ` loại bỏ cứng vùng nhiễu, sau đó temperature làm mềm vùng thông tin.
  1. Top-nσ: Lọc cứng → Set logit của noise tokens = -∞ → chỉ giữ informative region
  2. Temperature: Làm mềm → điều chỉnh exploration trong vùng đã lọc
  - `T > 1`: làm phẳng phân phối (less peaked)
  - `T < 1`: làm nhọn phân phối (more peaked)


**Temperature Invariance** - Tính chất đặc biệt nhất: Theorem 4: "For any temperature T > 0, the nucleus of top-nσ remains invariant."

KẾT LUẬN

Bài báo "Top-nσ: Not All Logits Are You Need" đại diện cho một bước đột phá quan trọng trong lĩnh vực text generation của các mô hình ngôn ngữ lớn. Thay vì tiếp tục cải tiến các phương pháp sampling truyền thống hoạt động trên phân phối xác suất, tác giả đã đi ngược lại nguồn gốc vấn đề bằng cách phân tích trực tiếp cấu trúc của logits.

Phát hiện then chốt là logits trong LLM có cấu trúc hai vùng rõ rệt: vùng nhiễu tuân theo phân phối Gaussian chứa đa số token không liên quan, và vùng thông tin với phân phối đều chứa các token có ý nghĩa. Vùng nhiễu xuất hiện do ba nguyên nhân chính là nhiễu từ dữ liệu huấn luyện, hiệu ứng regularization, và đặc biệt là khuyết điểm cố hữu của hàm softmax khi buộc phải gán giá trị hữu hạn cho tất cả token kể cả những token không liên quan.

Phương pháp top-nσ được đề xuất hoạt động bằng cách lọc cứng các token có logit thấp hơn ngưỡng M - nσ, trong đó M là logit tối đa và σ là độ lệch chuẩn. Điều đặc biệt là tính chất bất biến nhiệt độ - tập token được chọn không thay đổi khi điều chỉnh temperature, cho phép tách biệt hoàn toàn việc kiểm soát kích thước nucleus và mức độ exploration.

Kết quả thực nghiệm trên bốn dataset reasoning cho thấy top-nσ không chỉ vượt trội so với các phương pháp sampling hiện tại mà còn tốt hơn cả greedy decoding, đặc biệt ở nhiệt độ cao khi các phương pháp khác bị suy giảm nghiêm trọng. Điều này thách thức quan niệm truyền thống rằng reasoning tasks cần nhiệt độ thấp, thay vào đó cho thấy exploration có kiểm soát thực sự có thể cải thiện chất lượng.

Ý nghĩa thực tiễn của nghiên cứu vượt ra ngoài phạm vi sampling, mở ra hướng cải tiến architecture và training procedures dựa trên hiểu biết về cấu trúc logit. Tuy nhiên, phương pháp hiện tại chưa thể áp dụng trực tiếp trong pretraining do tính không khả vi, và cũng chưa giải quyết căn bản vấn đề over-confidence của mô hình.

[ logits: giá trị đầu ra thô trước softmax; LLM: mô hình ngôn ngữ lớn; regularization: kỹ thuật điều chuẩn; softmax: hàm chuyển logits thành xác suất; nucleus: tập token được chọn; exploration: khám phá không gian token; reasoning: suy luận; greedy decoding: giải mã tham lam; over-confidence: quá tự tin; architecture: kiến trúc mô hình ]


## FIRST: Teach A Reliable Large Language Model Through Efficient Trustworthy Distillation
- https://alphaxiv.org/abs/2408.12168v1

Bài báo này giải quyết vấn đề mất cân bằng hiệu chuẩn trong các mô hình ngôn ngữ lớn sau khi fine-tuning, khi độ tin cậy dự đoán không phù hợp với độ chính xác thực tế. Tác giả phát hiện hiện tượng "tuning-induced mis-calibration" - mô hình quá tự tin vào token đầu tiên và thiếu tự tin vào các token khác.

Phương pháp FIRST được đề xuất dựa trên hai khám phá quan trọng: "concentrated knowledge" - 95% xác suất tập trung ở top-5 token, và kỹ thuật "trustworthy maximization" sử dụng temperature scaling để hiệu chuẩn lại kiến thức từ mô hình giáo viên.

Kết quả thực nghiệm cho thấy FIRST đạt độ chính xác cao hơn 2.3% và giảm 10% lỗi hiệu chuẩn so với phương pháp truyền thống, đồng thời tiết kiệm đáng kể chi phí lưu trữ (từ 120TB xuống 1.2GB) nhờ chỉ sử dụng top-5 token thay vì toàn bộ phân phối xác suất.

[Fine-tuning: quá trình tinh chỉnh mô hình; ECE (Expected Calibration Error): sai số hiệu chuẩn kỳ vọng; Temperature scaling: kỹ thuật điều chỉnh nhiệt độ để cân bằng phân phối xác suất; Distillation: chưng cất kiến thức từ mô hình lớn sang mô hình nhỏ]

|![](https://pbs.twimg.com/media/GtJiNZxbMAMuvbI?format=jpg&name=medium)|![](https://pbs.twimg.com/media/GtJlU8dbMAElKjF?format=jpg&name=medium)|
|-|-|
|![]()|![]()|

Sau khi chọn top-5 tokens, FIRST thực hiện bước "Trustworthy Maximization" để giải quyết vấn đề teacher model bị "tuning-induced mis-calibration", tức là quá tự tin vào top-1 token và thiếu tự tin vào các token khác.

Phương pháp được sử dụng là Temperature Scaling với công thức `PT(i) = exp(PT(i)/c) / Σj exp(PT(j)/c)`, trong đó c là temperature parameter được tối ưu trên validation set nhằm minimize ECE (Expected Calibration Error). Khi c lớn hơn 1, phân phối sẽ được làm "mềm" để giảm over-confidence của top-1 token, ngược lại khi c nhỏ hơn 1 thì phân phối được làm "cứng" hơn để tăng concentration. Quá trình grid search được áp dụng để tìm ra giá trị c optimal, từ đó thu được teacher knowledge đã được hiệu chuẩn lại và sẵn sàng để transfer sang student model.

[ECE: Expected Calibration Error - sai số hiệu chuẩn kỳ vọng; Grid search: tìm kiếm lưới; Re-calibrate: hiệu chuẩn lại]

Knowledge Matching là bước cuối cùng trong pipeline FIRST, được thực hiện sau khi đã có teacher knowledge đã được hiệu chuẩn lại. Trong bước này, student model được huấn luyện bằng cách sử dụng cùng training data nhưng thay vì áp dụng language modeling loss trên hard labels như phương pháp truyền thống, FIRST sử dụng soft labels từ teacher.

Cụ thể, student model sẽ tạo ra xác suất cho 5 tokens tương ứng với top-5 của teacher, được ký hiệu là PS chứa PS(1), PS(2), ..., PS(5). Sau đó, Kullback-Leibler divergence được sử dụng để đo độ khác biệt giữa phân phối xác suất của teacher PT và student PS theo công thức Loss(y1:N) = ΣN t=1 DKL(PT||PS).

[Kullback-Leibler divergence: phân kỳ Kullback-Leibler để đo độ khác biệt giữa hai phân phối xác suất; Soft labels: nhãn mềm dưới dạng phân phối xác suất; Hard labels: nhãn cứng dạng one-hot]

Grid search trong FIRST là quá trình tìm kiếm temperature parameter c tối ưu để minimize Expected Calibration Error trên validation set. Quá trình này được thực hiện theo hai giai đoạn để đảm bảo độ chính xác cao.

Giai đoạn đầu tiên, các tác giả chia khoảng từ 0 đến 1 thành các bước nhảy 0.1 và tính toán ECE cho từng giá trị temperature. Ví dụ, nếu temperature 0.3 cho ECE thấp nhất, họ sẽ chọn khoảng này để tinh chỉnh thêm. Giai đoạn thứ hai thu hẹp khoảng tìm kiếm và sử dụng bước nhảy nhỏ hơn là 0.02 để xác định chính xác giá trị c optimal.

Khi temperature quá lớn, tất cả top-5 tokens sẽ có xác suất gần bằng nhau khoảng 0.2, dẫn đến mất thông tin quan trọng. Khi temperature bằng 1, xác suất của top-1 token bị nén xuống trong khi các token khác được phóng đại. Ngược lại, temperature 0.1 có thể làm tăng over-confidence của các token đã quá tự tin, dẫn đến hiệu chuẩn tệ hơn. Thực nghiệm cho thấy FIRST với temperature optimal vượt trội hơn hẳn so với các giá trị temperature khác, chứng minh tính hiệu quả của việc chọn temperature phù hợp.

[Expected Calibration Error: sai số hiệu chuẩn kỳ vọng; Validation set: tập dữ liệu kiểm tra; Over-confidence: quá tự tin]

Expected Calibration Error được tính theo công thức **ECE = Σ(M, m=1) |Bm|/n |acc(Bm) - conf(Bm)|** và không cần data ngoài để tham khảo. Quá trình tính toán bắt đầu bằng việc chia confidence thành M bins, thường là 10 khoảng từ 0-0.1, 0.1-0.2 cho đến 0.9-1.0. Với mỗi bin m, chúng ta tính |Bm| là số lượng predictions trong bin đó, acc(Bm) là độ chính xác trung bình trong bin, và conf(Bm) là confidence trung bình trong bin. Cuối cùng, ECE được tính bằng weighted average của sự khác biệt giữa accuracy và confidence.

Việc tính ECE yêu cầu **validation set có sẵn**, model predictions cùng **ground truth**, và confidence scores từ model mà không cần thêm data bên ngoài. Ví dụ, nếu bin 0.8-0.9 có 100 predictions với accuracy 75% và confidence 85%, thì đóng góp của bin này sẽ là (100/total) nhân với |75% - 85%| bằng (100/total) nhân 10%. ECE đo độ lệch giữa confidence và actual performance, trong đó model well-calibrated sẽ có ECE thấp.

[Bins: các khoảng chia; Weighted average: trung bình có trọng số; Well-calibrated: hiệu chuẩn tốt]

---

## Top-nσ: Not All Logits Are You Need: Phân tích chi tiết các hướng ứng dụng từ insights về cấu trúc logit
- https://alphaxiv.org/abs/2411.07641

- **Pre-filtering layers**: Thêm layer chuyên lọc noise trước softmax
- **Attention sink modifications**: Dựa trên insight rằng "model's architectural constraint of assigning finite values to irrelevant tokens"
- **Adaptive vocabulary**: Dynamic vocabulary size dựa trên context

- **Sparse softmax**: Chỉ compute trên informative tokens
- **Learned thresholding**: Học ngưỡng tự động thay vì fixed softmax
- **Hierarchical softmax**: Phân tầng vocabulary theo mức độ liên quan

Ba nguồn nhiễu đã xác định:
- `Training Data Noise`     → Data filtering/weighting  
- `Regularization Effects`  → Adaptive regularization
- `Noise of Silence`        → Modified loss functions

Loss function modifications:
- `Penalty cho noise region`: Thêm term phạt tokens ở vùng nhiễu
- `Focus loss`: Tăng trọng số cho informative tokens
- `Distribution shaping`: Khuyến khích separation rõ ràng hơn


--

Ý nghĩa thực tiễn của nghiên cứu mở ra nhiều hướng phát triển đầy tiềm năng vượt xa phạm vi của các phương pháp sampling truyền thống. Từ hiểu biết sâu sắc về cấu trúc hai vùng trong logits, các nhà nghiên cứu có thể cải tiến kiến trúc mô hình theo những cách hoàn toàn mới.

Về mặt kiến trúc, có thể phát triển các layer tiền xử lý chuyên biệt để lọc bỏ nhiễu trước khi áp dụng softmax, dựa trên nhận thức rằng kiến trúc mô hình hiện tại buộc phải gán giá trị hữu hạn cho các token không liên quan. Điều này có thể dẫn đến những biến thể softmax thưa thớt chỉ tính toán trên các token có ý nghĩa, hoặc các cơ chế học ngưỡng tự động thay thế cho softmax cố định. Thậm chí có thể phát triển softmax phân tầng, phân chia vocabulary theo mức độ liên quan đến ngữ cảnh.

Trong quá trình huấn luyện, việc xác định được ba nguồn gốc chính của nhiễu mở ra khả năng can thiệp có mục tiêu. Nhiễu từ dữ liệu huấn luyện có thể được giải quyết bằng các kỹ thuật lọc và gán trọng số dữ liệu tinh vi hơn. Hiệu ứng của regularization có thể được điều chỉnh thích ứng, trong khi vấn đề noise of silence có thể được xử lý thông qua các hàm loss được thiết kế lại.

Các hàm loss có thể được cải tiến để thêm các thành phần phạt cho vùng nhiễu, tăng trọng số cho các token thuộc vùng thông tin, hoặc khuyến khích sự phân tách rõ ràng hơn giữa hai vùng. Về hiệu quả tính toán, có thể áp dụng gradient clipping chỉ trong vùng thông tin, lan truyền ngược có chọn lọc bỏ qua các token nhiễu, hoặc sử dụng vocabulary động để giảm bộ nhớ cần thiết.

Ví dụ, thay vì huấn luyện trên toàn bộ vocabulary 50 nghìn từ, có thể sử dụng mặt nạ động chỉ giữ lại khoảng 500 token liên quan cho mỗi bước huấn luyện. Tuy nhiên, tác giả cũng thận trọng nhấn mạnh rằng những hiểu biết này có thể đóng góp vào việc cải tiến các quy trình huấn luyện trong nghiên cứu tương lai, chứ vẫn chưa phải là giải pháp đã được triển khai thực tế.

[ logits: giá trị đầu ra thô trước softmax; softmax: hàm chuyển logits thành xác suất; vocabulary: bộ từ vựng; regularization: kỹ thuật điều chuẩn; gradient clipping: cắt ngưỡng gradient; lan truyền ngược: backpropagation ]

--

Dựa trên những hiểu biết sâu sắc từ bài báo về cấu trúc logits, có thể phát triển nhiều phương án thay thế thú vị cho hàm softmax truyền thống. Một trong những hướng tiếp cận đầy hứa hẹn là nhóm **sparsemax và entmax**, thay vì việc chuẩn hóa cưỡng bức của softmax, sparsemax cho phép một số đầu ra thực sự bằng không. Entmax tổng quát hóa ý tưởng này với tham số α để điều chỉnh mức độ thưa thớt, điều này hoàn toàn phù hợp với nhận thức của top-nσ rằng nhiều token nên được "im lặng".

**Hierarchical softmax** mang lại một cách tiếp cận khác biệt bằng cách thay thế cấu trúc vocabulary phẳng bằng cấu trúc cây phân tầng. Ở tầng đầu tiên, hệ thống phân loại thô các nhóm từ như danh từ, động từ, dấu câu, sau đó ở tầng thứ hai mới xác định các token cụ thể trong mỗi danh mục. Cách tiếp cận này không chỉ mang lại lợi ích tính toán với độ phức tạp O(log V) thay vì O(V), mà còn giảm nhiễu bằng cách loại bỏ sớm các danh mục không phù hợp.

[ sparsemax: phiên bản thưa của softmax; entmax: tổng quát hóa sparsemax; hierarchical: phân tầng; mixture: hỗn hợp; adaptive: thích ứng; Gaussian mixture: hỗn hợp Gaussian; cơ chế cổng: gating mechanism; tính thưa thớt: sparsity ]

--

Trong số các phương án thay thế softmax đã thảo luận, một số có thể áp dụng trực tiếp trong pre-training nhờ tính chất khả vi và ổn định tính toán.

**Sparsemax và entmax** là những ứng viên sáng giá nhất vì chúng hoàn toàn khả vi và đã có implementation ổn định. `Sparsemax` đặc biệt phù hợp vì nó tự nhiên tạo ra độ thưa thớt mà không cần ngưỡng cứng, cho phép một số token thực sự có xác suất bằng không. Entmax với tham số α có thể điều chỉnh mức độ thưa thớt theo từng tầng hoặc giai đoạn training, tạo ra sự linh hoạt trong việc kiểm soát phân phối.

`Hierarchical softmax` đã được chứng minh hiệu quả trong pre-training từ thời word2vec và có thể mở rộng cho các mô hình transformer hiện đại. Phương pháp này không chỉ giảm độ phức tạp tính toán từ O(V) xuống O(log V) mà còn tự nhiên tạo ra cấu trúc phân tầng giúp mô hình học được mối quan hệ ngữ nghĩa tốt hơn.


### Hierarchical softmax

Hierarchical softmax hoạt động bằng cách thay thế việc tính toán xác suất trực tiếp trên toàn bộ vocabulary bằng một chuỗi quyết định nhị phân theo cấu trúc cây. Thay vì một softmax duy nhất trên V tokens, mô hình thực hiện log₂(V) quyết định nhị phân từ gốc đến lá, mỗi quyết định có độ phức tạp O(1).

Cơ chế cốt lõi là mỗi node trong cây đại diện cho một quyết định nhị phân với sigmoid activation. Xác suất của một token cụ thể được tính bằng tích của tất cả các quyết định từ gốc đến lá tương ứng. Điều này đảm bảo tổng xác suất của tất cả tokens vẫn bằng 1 mà không cần normalize toàn bộ vocabulary.

Để tạo hierarchy tự động, có nhiều phương pháp tinh vi. Clustering dựa trên embedding là cách phổ biến nhất, sử dụng các thuật toán như K-means hoặc hierarchical clustering trên word embeddings đã học trước. Phương pháp này nhóm các từ có ngữ nghĩa tương tự vào cùng một subtree, giúp mô hình tận dụng được mối quan hệ ngữ nghĩa.

**Trong bối cảnh insights từ bài báo top-nσ, có thể kết hợp logit analysis để tạo hierarchy động. Phân tích phân phối logits theo thời gian để XÁC ĐỊNH CÁC NHÓM TOKENS THƯỜNG CÙNG NẰM TRONG VÙNG INFORMATIVE, sau đó nhóm chúng vào cùng subtree. Điều này tạo ra hierarchy phản ánh thực tế sử dụng của mô hình.**

!!! Ý tưởng hay nhưng engineering complexity >> expected benefits trong bối cảnh pretraining thực tế. !!!

[ hierarchical: phân tầng; sigmoid: hàm sigmoid; normalize: chuẩn hóa; clustering: phân cụm; K-means: thuật toán phân cụm K-means; embedding: vector đại diện; co-occurrence: đồng xuất hiện; syntactic: cú pháp; semantic: ngữ nghĩa; part-of-speech: từ loại; gradient feedback: phản hồi gradient; graph neural networks: mạng neural đồ thị ]

---

- **Pre-filtering layers**: Thêm layer chuyên lọc noise trước softmax
- **Attention sink modifications**: Dựa trên insight rằng "model's architectural constraint of assigning finite values to irrelevant tokens"
- **Adaptive vocabulary**: Dynamic vocabulary size dựa trên context

- **Sparse softmax**: Chỉ compute trên informative tokens
- **Learned thresholding**: Học ngưỡng tự động thay vì fixed softmax
- **Hierarchical softmax**: Phân tầng vocabulary theo mức độ liên quan

Ba nguồn nhiễu đã xác định:
- `Training Data Noise`     → Data filtering/weighting  
- `Regularization Effects`  → Adaptive regularization
- `Noise of Silence`        → Modified loss functions

Loss function modifications:
- `Penalty cho noise region`: Thêm term phạt tokens ở vùng nhiễu
- `Focus loss`: Tăng trọng số cho informative tokens
- `Distribution shaping`: Khuyến khích separation rõ ràng hơn


### Sparsemax => Liger Kernel có

Thuật toán sparsemax hoạt động theo ba bước chính. Đầu tiên, các logits đầu vào được sắp xếp theo thứ tự giảm dần để tạo ra dãy z₁ ≥ z₂ ≥ ... ≥ zₙ. Bước thứ hai là tìm ra ngưỡng τ thông qua việc xác định chỉ số k lớn nhất sao cho điều kiện z_k - (1/k) × (tổng từ z₁ đến z_k - 1) > 0 được thỏa mãn. Ngưỡng τ sau đó được tính bằng công thức (1/k) × (tổng từ z₁ đến z_k - 1).

### Attention sink modifications
là một hướng tiếp cận kiến trúc tinh vi dựa trên hiện tượng đã được quan sát trong các mô hình transformer, nơi attention weights có xu hướng tập trung bất thường vào một số ít tokens đầu tiên trong sequence, ngay cả khi những tokens này không mang ý nghĩa ngữ nghĩa quan trọng. Hiện tượng này được gọi là attention sink và có thể được coi như một dạng structural noise trong attention mechanism.

Một cách tiếp cận là learnable attention filtering, trong đó thêm một **gating layer** trước attention computation để predict tokens nào đáng được attend to. Layer này có thể học được patterns về việc tokens nào thường nằm trong vùng informative based on contextual cues. Ví dụ, trong một context về technology, tokens liên quan đến computer science có likelihood cao hơn là informative so với random common words.

Dynamic attention sparsity là một modification khác, sử dụng statistical properties của attention distribution để dynamically prune những attention connections có weight thấp. Thay vì compute full attention matrix, mechanism này có thể identify top-k most relevant tokens cho mỗi query position, tương tự như spirit của top-nσ nhưng applied ở attention level thay vì output level.

## https://github.com/Akkki28/SparseMax-Transformers
|![](https://pbs.twimg.com/media/GtKZtK_bMAIqQyT?format=png)|![](https://pbs.twimg.com/media/GtKZ8njbMAIsu8V?format=png&name=900x900)|
|-|-|

|![]()|![]()|
|-|-|

## Sparse Attention for Long-Range Transformers https://arxiv.org/html/2406.16747v1#S3

--- 

# sparsemax vs softmax
- survey https://chatgpt.com/share/6849865b-2460-8003-a5fa-1876f0341b77
- https://www.alphaxiv.org/abs/1602.02068 from softmax to sparsemax
- https://www.alphaxiv.org/abs/1905.05702 sparse seq-to-seq models (alpha-entmax)
- https://www.alphaxiv.org/abs/2004.02644 sparse text generation (entmax sampling)
- https://www.alphaxiv.org/abs/2502.12082 adaptive entmax sparse attn (2025 mới)

vocabulary lớn: sparsemax hoạt động như một bộ lọc trên không gian đầu ra lớn, chỉ giữ lại các biến có liên quan và gán 0 cho phần còn lại, từ đó tăng tính diễn giải của mô hình

trong cơ chế chú ý (attention), dùng sparsemax sẽ tạo trọng số chú ý tập trung vào một vài token nguồn quan trọng, bỏ qua hẳn các token ít liên quan – dẫn đến điểm chú ý gọn và dễ hiểu hơn mà vẫn đạt hiệu quả tương đương softmax

Sparsemax còn có một hệ quả thú vị: vì hầu hết các từ nhận xác suất 0, chỉ một số ít từ có xác suất dương, nên kích thước tập mở rộng trong suy luận (inference) sẽ giảm. Trong các mô hình seq2seq như dịch máy, người ta nhận thấy việc phân phối thưa có thể khiến beam search trở nên hiệu quả hơn

Một nghiên cứu về sinh văn bản thưa cho thấy mô hình ngôn ngữ huấn luyện với entmax (một tổng quát chứa sparsemax) cho văn bản sinh ra mạch lạc, ít lặp lại và đa dạng n-gram hơn so với softmax, gần hơn với phân phối ngôn ngữ tự nhiên của con người (sparse text generation)

### Gradient của soft vs sparse

Softmax luôn cho gradient khác 0 với mọi lớp (trừ lớp mục tiêu có gradient âm, các lớp khác có gradient dương tỷ lệ với xác suất dự đoán). Điều này đảm bảo mỗi trọng số đầu ra đều được cập nhật một chút tại mỗi mẫu: ngay cả những lớp không phải mục tiêu vẫn nhận gradient (nhỏ) để giảm xác suất của chúng. Thêm vào đó, việc softmax gắn liền với hàm mất mát log-likelihood (đối với nhãn one-hot) nghĩa là mô hình tối ưu trực tiếp cho xác suất tối đa của dữ liệu huấn luyện – thuận lợi về mặt thống kê (ước lượng hợp lý tối đa).

Đối với sparsemax, có một thách thức ngay lập tức: nếu dùng trực tiếp cross-entropy, bất kỳ mẫu nào mà mô hình gán xác suất 0 cho nhãn đúng sẽ làm loss âm vô cực, không khả thi để huấn luyện. Trên thực tế, tác giả Martins et al. đã định nghĩa một hàm mất mát mới gọi là sparsemax loss, thay thế cho cross-entropy khi dùng sparsemax. Hàm mất mát này được thiết kế sao cho gradient của nó cũng có dạng $p - y$ tương tự softmax nhưng tránh được vấn đề log(0). sparsemax loss được chứng minh là convex và khả vi mọi nơi, liên hệ tới Huber loss trong phân loại.

## sparse text generation
https://alphaxiv.org/abs/2004.02644

## sparse seq-to-seq model: α-entmax, chọn α (1.5) nằm giữa softmax (α=1) và sparsemax (α=2)
https://www.alphaxiv.org/abs/1905.05702

Tác giả đưa ra họ biến đổi α-entmax dựa trên entropy Tsallis (dạng tổng quát của entropy Shannon), bao gồm softmax (α=1) và sparsemax (α=2) như các trường hợp đặc biệt. Với α > 1, các phép biến đổi này tạo ra phân phối thưa.

## Flash Entmax Attention
- https://github.com/deep-spin/adasplash
- https://www.alphaxiv.org/abs/2502.12082
Thuật toán Halley-bisection lai: "Giảm 7 lần số vòng lặp cần thiết để tính α-entmax transformation"

---

# Sparse Attn
- https://huggingface.co/blog/Kseniase/attentions
- 