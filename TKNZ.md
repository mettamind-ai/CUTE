ADAPTIVE & FLEXIBLE TOKENIZATION
--------------------------------

## Over Tokenized Transformer and n-gram Embeddings
- https://arxiv.org/html/2501.16975v2
- https://www.alphaxiv.org/abs/2501.16975v2

Phương pháp tokenizer cơ bản (baseline tokenizer) xây dựng vốn từ vựng sử dụng 3 ký tự đầu cuối từ CFG, chia văn bản thành từng ký tự riêng biệt (gọi là tokenizer 1-gram). Ngoài ra, các tokenizer n-gram được định nghĩa dựa trên mọi tổ hợp có thể của $3^n$ chuỗi gồm $n$ ký tự liên tiếp. Nghiên cứu huấn luyện các mô hình GPT-2 kích thước lớn và nhỏ với tokenizer 1-gram và 3-gram.

* Việc sử dụng 3-gram ở encoder luôn cải thiện hiệu suất cho mọi kích thước mô hình. Ngược lại, sử dụng tokenizer 3-gram ở decoder làm giảm hiệu suất trên các mô hình nhỏ.

* Với tokenizer lớn, vocab lớn luôn có tác động tích cực, trong khi logits lớn có thể gây ảnh hưởng tiêu cực lên các mô hình nhỏ.

* Lí do là embedding đầu vào giúp mã hóa ngữ cảnh tốt hơn nhờ khả năng biểu diễn phong phú hơn khi dùng vốn từ lớn, trong khi vốn từ đầu ra lớn làm tăng độ chi tiết của nhiệm vụ dự đoán, có thể tạo gánh nặng đối với các mô hình nhỏ.

|![](https://arxiv.org/html/2501.16975v2/x1.png)|![](https://arxiv.org/html/2501.16975v2/x2.png)|
|-|-|
|![](https://arxiv.org/html/2501.16975v2/x3.png)|![](https://arxiv.org/html/2501.16975v2/x4.png)|

Áp dụng kết luận này, nghiên cứu mở rộng sang transformer “over-tokenized” trong mô hình ngôn ngữ tự nhiên thực tế (MTP-DS). Kết quả cho thấy việc over-encoding làm tăng năng lực biểu diễn của embedding, giúp nhiệm vụ dự đoán token tiếp theo dễ dàng hơn và đạt huấn luyện đầy đủ hơn, qua đó mang lại lợi ích đáng kể ngay cả với các mô hình tương đối nhỏ.

Theo kết quả thực nghiệm (Bảng 1), phương pháp OE-12.8M giảm loss huấn luyện ổn định ở cả hai quy mô mô hình, dù tỉ lệ tham số embedding giảm khi tăng quy mô mô hình. Tuy nhiên, lợi ích của OE với các chỉ số đánh giá nhiệm vụ thực tế lại giảm đi khi mô hình lớn hơn. Lý giải được đề xuất là do sự chồng lấn về lợi ích giữa các tham số sparse trong kiến trúc MoE và các tham số embedding sparse.

**Kết quả đáng chú ý**: "Using a large input vocabulary, we achieve performance comparable to double-sized baselines with no additional cost" - với từ vựng đầu vào lớn, mô hình 400M tham số đạt hiệu suất tương đương mô hình 1B tham số mà không tốn thêm chi phí.

**mối quan hệ log-linear** giữa kích thước từ vựng đầu vào và training loss: "exponentially increasing the input vocabulary size consistently results in a linear decrease in loss". => Cái này dễ hiểu!

KẾT LUẬN: OVER ENCODE SCALE TUYẾN TÍNH VÀ ỔN ĐỊNH, OVER DECODE SCALE PHI TUYẾN VÀ PHỤ THUỘC KÍCH THƯỚC MÔ HÌNH.


## STOCHASTOK
https://www.alphaxiv.org/overview/2506.01687

Mỗi seq đầu vào và xác xuất p.
Với p = 0.1 (default), nếu câu có 10 tokens thì sẽ expand 1 lần
Với câu có 20 tokens thì expand 2 lần

!!! Như vậy cũng có thể ngẫu nhiên merge 2 tokens lại để có được phiên bản 2-gram => TKNZ linh hoạt !!!

Kết hợp OT và STOCHASTOK
------------------------

!!! Ta có thể huấn luyện cho model hiểu từ vựng có độ phân giản mịn hơn (tách 1 token làm 2 tokens) hoặc đô phân giải thô hơn(sử dụng 2-gram như là 1 token) bằng cách ngẫu nhiên tăng hoặc giảm độ phân giải của chuỗi đầu vào !!!

---

## Scaling LLM Pre-training with Vocabulary Curriculum
- https://ar5iv.labs.arxiv.org/html/2502.17910
- https://www.alphaxiv.org/abs/2502.17910
- Entropy-Guided Vocabulary Updates
![](https://ar5iv.labs.arxiv.org/html/2502.17910/assets/better-scale-vocab-curriculum-1.png)

ban đầu mô hình học xử lý ký tự và các đơn vị nhỏ (giúp nắm chắc cấu trúc cơ bản), về sau dần “nâng cấp” lên các token lớn hơn cho những mẫu phổ biến. Yu và cộng sự cho biết cách làm này giúp mô hình GPT nhỏ đạt bpc (bits-per-character) thấp hơn ~6.7% so với mô hình dùng vocab cố định cùng kích thước. Hơn nữa, khi tăng gấp đôi kích thước vocab, mô hình thích ứng thu được hiệu quả cải thiện cao hơn ~34% so với mô hình truyền thống (tức là tận dụng vocab lớn tốt hơn). Kết quả cũng cho thấy một hệ thống phân cấp token tự nhiên hình thành: các token dài dần xuất hiện để đại diện cho các cụm từ phổ biến, dễ dự đoán, còn những đoạn nội dung khó dự đoán thì vẫn bị phân nhỏ thành token ngắn hơn để mô hình xử lý chi tiết. Điều này khớp với trực giác rằng tokenization động cho phép mô hình phân bổ tài nguyên tính toán hợp lý hơn – dành nhiều “não” hơn cho phần phức tạp, bớt tốn sức cho phần đơn giản.


# ADATOK
- https://proceedings.neurips.cc/paper_files/paper/2024/file/cdf00c97c0cb2cc35179f03363da6c4f-Paper-Conference.pdf
![](https://pbs.twimg.com/media/GtS0C4ybMAItNaH?format=jpg&name=large)

ADAT bắt đầu với một từ vựng khổng lồ gồm 150 nghìn tokens được tạo ra bằng các thuật toán truyền thống như Unigram hoặc BytePiece. Mục tiêu là thu gọn từ vựng này xuống còn 50 nghìn tokens thông qua năm vòng lặp cắt tỉa liên tiếp.

Trong mỗi vòng lặp, quá trình diễn ra qua bốn bước chính. Đầu tiên, một mô hình LLM nhỏ được khởi tạo ngẫu nhiên và huấn luyện trên 0.3 tỷ tokens với từ vựng hiện tại. Tiếp theo, mô hình này thực hiện inference trên 0.1 tỷ tokens để thu thập dữ liệu về hiệu suất của từng token.

Hệ thống sau đó tính toán hai loại loss quan trọng cho mỗi token. Loss tần suất Unigram LP(xi) được tính bằng công thức LP(V) trừ đi LP(V-xi), phản ánh tầm quan trọng của token theo góc độ thống kê. Loss hiệu suất LLM LM(xi) được tính bằng tổng các giá trị cross-entropy CE(M(xi-1), xi), đánh giá khả năng của token trong việc giúp mô hình dự đoán chính xác.

Hai loại loss này được kết hợp thành điểm số cuối cùng theo công thức L(xi) bằng LP(xi) chia cho λ nhân với logarit của LM(xi) cộng một. Để tăng tính ổn định, hệ thống áp dụng kỹ thuật momentum với công thức L^j_momentum(xi) bằng β nhân L^j-1_momentum(xi) cộng với L^j(xi).

Cuối cùng, các tokens được xếp hạng theo điểm số giảm dần và 20% tokens có điểm thấp nhất sẽ bị loại bỏ. Quá trình này lặp lại qua năm vòng, từ 150 nghìn tokens ban đầu giảm dần xuống 120 nghìn, 96 nghìn, 77 nghìn, 62 nghìn và cuối cùng là 50 nghìn tokens.

Điểm đặc biệt của phương pháp này là việc đánh giá tokens dựa trên cả tần suất xuất hiện lẫn khả năng thực sự giúp mô hình dự đoán tốt hơn, thay vì chỉ dựa vào thống kê tần suất đơn thuần như các phương pháp truyền thống.

---

=> 

# EVOTOK: Where vocabularies evolve with learning (kết hợp Vocab Curriculum và ADATOK)
https://www.alphaxiv.org/abs/2410.04335v1?conversation_id=684b9e2201b4f61b63a7ab65

Cách tiếp cận đề xuất bắt đầu với một từ vựng trung bình khoảng 100 nghìn tokens, một điểm cân bằng tốt giữa độ bao phủ và hiệu quả, với mục tiêu cuối cùng là thu gọn xuống 50 nghìn tokens. Thay vì phải trải qua hai giai đoạn riêng biệt như các phương pháp trước đó, quy trình mới này thực hiện song song hai thao tác sau mỗi epoch huấn luyện.

Đầu tiên là thao tác loại bỏ tokens kém hiệu quả. Hệ thống tính toán điểm số cho tất cả tokens theo công thức L(xi) bằng F(LP(xi), LM(xi)) rồi loại bỏ 20% tokens có điểm số thấp nhất, tương đương 20 nghìn tokens. Đồng thời, hệ thống cũng thực hiện thao tác thêm tokens mới bằng cách áp dụng entropy-guided merging trên corpus để tìm các sequences thỏa mãn điều kiện H(st|s1:t−1) nhỏ hơn H(st−1|s1:t−2) và H(st|s1:t−1) nhỏ hơn ngưỡng epsilon, sau đó thêm vào 10% tokens mới tốt nhất, tương đương 10 nghìn tokens.

Kết quả sau mỗi iteration là từ vựng giảm dần từ 100 nghìn xuống 90 nghìn tokens, tạo nên một quá trình hội tụ có kiểm soát. Cách tiếp cận này mang lại nhiều ưu điểm vượt trội so với các phương pháp trước đó.

Về mặt hiệu quả, việc chỉ cần một lần huấn luyện cho cả hai thao tác giúp giảm 50% chi phí tính toán so với approach riêng lẻ. Từ vựng có thể thích ứng liên tục với quá trình học của mô hình, loại bỏ những tokens đã trở nên lỗi thời và bổ sung những tokens phù hợp với giai đoạn học hiện tại.

Quá trình hội tụ được kiểm soát chặt chẽ thông qua việc điều chỉnh tỷ lệ loại bỏ và thêm mới để đạt được kích thước mục tiêu, ví dụ từ 100 nghìn giảm dần qua các mốc 90 nghìn, 80 nghìn, 70 nghìn, 60 nghìn và cuối cùng là 50 nghìn tokens. Chất lượng được đảm bảo khi tokens mới được tạo ra dựa trên các entropy patterns hiện tại của mô hình, trong khi tokens cũ bị loại bỏ dựa trên performance thực tế.

---

**Cuối cùng làm thế nào để khởi tạo 1 bộ vocab mới theo ý mình từ 1 vocab / model sẵn có?**

=>

# Training free token transplantation via OMP (orthogonal matching pursuit)
- https://www.alphaxiv.org/abs/2506.06607

...