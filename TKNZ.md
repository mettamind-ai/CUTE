ADAPTIVE & FLEXIBLE TOKENIZATION
--------------------------------

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

Giải pháp – ADAT (Adaptive Tokenizer)

- Khởi tạo từ từ-vựng lớn.
- Huấn luyện LLM, tính loss từng token = hàm kết hợp tần suất & cross-entropy.
- Cắt bỏ token đóng góp thấp, lặp lại → tạo tokenizer “thích ứng” với mô hình.

=> !!! Có thể ADAPT ngay trong lúc pre-train !!!

## Scaling LLM Pre-training with Vocabulary Curriculum
- https://ar5iv.labs.arxiv.org/html/2502.17910
- https://www.alphaxiv.org/abs/2502.17910
- Entropy-Guided Vocabulary Updates
![](https://ar5iv.labs.arxiv.org/html/2502.17910/assets/better-scale-vocab-curriculum-1.png)

ban đầu mô hình học xử lý ký tự và các đơn vị nhỏ (giúp nắm chắc cấu trúc cơ bản), về sau dần “nâng cấp” lên các token lớn hơn cho những mẫu phổ biến. Yu và cộng sự cho biết cách làm này giúp mô hình GPT nhỏ đạt bpc (bits-per-character) thấp hơn ~6.7% so với mô hình dùng vocab cố định cùng kích thước. Hơn nữa, khi tăng gấp đôi kích thước vocab, mô hình thích ứng thu được hiệu quả cải thiện cao hơn ~34% so với mô hình truyền thống (tức là tận dụng vocab lớn tốt hơn). Kết quả cũng cho thấy một hệ thống phân cấp token tự nhiên hình thành: các token dài dần xuất hiện để đại diện cho các cụm từ phổ biến, dễ dự đoán, còn những đoạn nội dung khó dự đoán thì vẫn bị phân nhỏ thành token ngắn hơn để mô hình xử lý chi tiết. Điều này khớp với trực giác rằng tokenization động cho phép mô hình phân bổ tài nguyên tính toán hợp lý hơn – dành nhiều “não” hơn cho phần phức tạp, bớt tốn sức cho phần đơn giản.

---

# OTT: Over Tokenized Transformer
- https://arxiv.org/html/2501.16975v2
- https://www.alphaxiv.org/abs/2501.16975v2

A baseline tokenizer constructs a vocabulary using the three terminal characters defined by the CFG, tokenizing sentences character-wisely, which we refer as a `1-gram tokenizer`. We further define `n-gram tokenizers`, whose vocabulary comprises all `3^n possible combinations of n sequential characters`. We train both larger and smaller GPT-2 models using 1-gram and 3-gram tokenizers, respectively.

- The left panel compares 1-gram and **3-gram tokenizers**, showing that 3-gram improves larger (85M parameters) models but harms smaller (2.4M parameters) ones
- The right panel examines **3-gram usage in encoders and decoders**, revealing consistent gains with 3-gram encoders regardless of model size, while 3-gram decoders degrade performance in smaller models.

|![](https://arxiv.org/html/2501.16975v2/x1.png)|![](https://arxiv.org/html/2501.16975v2/x2.png)|
|-|-|
|![](https://arxiv.org/html/2501.16975v2/x3.png)|![](https://arxiv.org/html/2501.16975v2/x4.png)|

We conclude that, when using large tokenizers, the large input vocabulary is always positive while the large output vocabulary can be negative for smaller models. We hypothesize that the difference lies in their respective roles: the input embedding is responsible for encoding the context into feature embeddings, where a larger vocabulary enhances the representational capacity of the feature mapping, thereby positively impacting the model. In contrast, the output vocabulary determines the granularity of the prediction task. A larger output vocabulary implies more fine-grained supervision signals, which can either be beneficial (e.g., for large models prone to overfitting) or burdensome (e.g., for smaller models suffering from severe underfitting). Motivated by this observation, we extend our exploration to over-tokenized transformers in real-world natural language modeling.

Under MTP-DS architecture, over-encoding enhances the representation capacity of token embeddings and directly participates future token predictions. On the one hand, the future token prediction tasks become easier to learn. On the other hand, the over-encoding can be trained more sufficiently. With these advantages, the integration of the two methods yields greater benefits, even on relatively smaller models.

https://arxiv.org/html/2501.16975v2#S4.SS1.SSS0.Px2
The results are shown in Table 1. We first compare the training loss. At two different model scales, OE-12.8M achieves approximately the same improvement in loss compared to the baseline, despite decreases in the proportion of embedding parameters as the model scales up (i.e., 10× dense parameters for OLMoE-1.3B and 3.7× for OLMoE-7B). However, in terms of downstream evaluation metrics, the performance improvement of OE diminishes. We hypothesize that this reduction is related to the sparse parameters utilized in the MoE architecture, which may overlap with the benefits provided by sparse embedding parameters.

# ADATOK
![](https://pbs.twimg.com/media/GtS0C4ybMAItNaH?format=jpg&name=large)

ADAT bắt đầu với một từ vựng khổng lồ gồm 150 nghìn tokens được tạo ra bằng các thuật toán truyền thống như Unigram hoặc BytePiece. Mục tiêu là thu gọn từ vựng này xuống còn 50 nghìn tokens thông qua năm vòng lặp cắt tỉa liên tiếp.

Trong mỗi vòng lặp, quá trình diễn ra qua bốn bước chính. Đầu tiên, một mô hình LLM nhỏ được khởi tạo ngẫu nhiên và huấn luyện trên 0.3 tỷ tokens với từ vựng hiện tại. Tiếp theo, mô hình này thực hiện inference trên 0.1 tỷ tokens để thu thập dữ liệu về hiệu suất của từng token.

Hệ thống sau đó tính toán hai loại loss quan trọng cho mỗi token. Loss tần suất Unigram LP(xi) được tính bằng công thức LP(V) trừ đi LP(V-xi), phản ánh tầm quan trọng của token theo góc độ thống kê. Loss hiệu suất LLM LM(xi) được tính bằng tổng các giá trị cross-entropy CE(M(xi-1), xi), đánh giá khả năng của token trong việc giúp mô hình dự đoán chính xác.

Hai loại loss này được kết hợp thành điểm số cuối cùng theo công thức L(xi) bằng LP(xi) chia cho λ nhân với logarit của LM(xi) cộng một. Để tăng tính ổn định, hệ thống áp dụng kỹ thuật momentum với công thức L^j_momentum(xi) bằng β nhân L^j-1_momentum(xi) cộng với L^j(xi).

Cuối cùng, các tokens được xếp hạng theo điểm số giảm dần và 20% tokens có điểm thấp nhất sẽ bị loại bỏ. Quá trình này lặp lại qua năm vòng, từ 150 nghìn tokens ban đầu giảm dần xuống 120 nghìn, 96 nghìn, 77 nghìn, 62 nghìn và cuối cùng là 50 nghìn tokens.

Điểm đặc biệt của phương pháp này là việc đánh giá tokens dựa trên cả tần suất xuất hiện lẫn khả năng thực sự giúp mô hình dự đoán tốt hơn, thay vì chỉ dựa vào thống kê tần suất đơn thuần như các phương pháp truyền thống.

