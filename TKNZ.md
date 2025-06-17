LVOT: LLM-based Vocabulary Optimization for Tokenization
--------------------------------------------------------

Đầu tiên, hãy đọc một loạt các bài báo gần đây về những ý tưởng và tiến bộ trong việc xây dựng vocab, áp dụng trong tokenization.

## Paper1: Over Tokenized Transformer and n-gram Embeddings
- https://arxiv.org/html/2501.16975v2
- https://www.alphaxiv.org/abs/2501.16975v2

Phương pháp tokenizer cơ bản (baseline tokenizer) xây dựng vốn từ vựng sử dụng 3 ký tự đầu cuối từ CFG, chia văn bản thành từng ký tự riêng biệt (gọi là tokenizer 1-gram). Ngoài ra, các tokenizer n-gram được định nghĩa dựa trên mọi tổ hợp có thể của $3^n$ chuỗi gồm $n$ ký tự liên tiếp. Nghiên cứu huấn luyện các mô hình GPT-2 kích thước lớn và nhỏ với tokenizer 1-gram và 3-gram.

* Việc sử dụng 3-gram ở encoder luôn cải thiện hiệu suất cho mọi kích thước mô hình. Ngược lại, sử dụng tokenizer 3-gram ở decoder làm giảm hiệu suất trên các mô hình nhỏ. **Note**: over decode hay dự đoán n-gram ở đầu ra đơn giản là dùng MTP (multi token prediction), tức là sử dụng nhiều hơn 1 head để dự đoán nhiều gram (token) một lúc.

* Embeddings lớn luôn có tác động tích cực, trong khi logits lớn có thể gây ảnh hưởng tiêu cực lên các mô hình nhỏ.

* Lí do là embedding đầu vào giúp mã hóa ngữ cảnh tốt hơn nhờ khả năng biểu diễn phong phú hơn khi dùng vốn từ lớn, trong khi vốn từ đầu ra lớn làm tăng độ chi tiết của nhiệm vụ dự đoán, có thể tạo gánh nặng đối với các mô hình nhỏ.

|![](https://arxiv.org/html/2501.16975v2/x1.png)|![](https://arxiv.org/html/2501.16975v2/x2.png)|
|-|-|
|![](https://arxiv.org/html/2501.16975v2/x3.png)|![](https://arxiv.org/html/2501.16975v2/x4.png)|

Áp dụng kết luận này, nghiên cứu mở rộng sang transformer “over-tokenized” trong mô hình ngôn ngữ. Kết quả cho thấy việc over-encoding (tăng embedding thông qua kết hợp token embedding với n-gram embeddings - với những n-gram mà có gram sau cùng là token đang xét) làm tăng năng lực biểu diễn của embedding, giúp nhiệm vụ dự đoán token tiếp theo dễ dàng hơn và đạt huấn luyện đầy đủ hơn, qua đó mang lại lợi ích đáng kể ngay cả với các mô hình tương đối nhỏ.

Theo kết quả thực nghiệm (paper's table 1), phương pháp OE-12.8M giảm loss huấn luyện ổn định ở cả hai quy mô mô hình, dù tỉ lệ tham số embedding giảm khi tăng quy mô mô hình. Tuy nhiên, lợi ích của OE với MoE (mixture of expert) không nhiều như dense model. Lý giải được đề xuất là do sự chồng lấn về lợi ích giữa các tham số sparse trong kiến trúc MoE và các tham số embedding sparse.

**Kết quả đáng chú ý**: "Using a large input vocabulary, we achieve performance comparable to double-sized baselines with no additional cost" - với từ vựng đầu vào lớn, mô hình 400M tham số đạt hiệu suất tương đương mô hình 1B tham số mà không tốn thêm chi phí.

**mối quan hệ log-linear** giữa kích thước từ vựng đầu vào và training loss: "exponentially increasing the input vocabulary size consistently results in a linear decrease in loss".

KẾT LUẬN: OVER ENCODE SCALE TUYẾN TÍNH VÀ ỔN ĐỊNH, OVER DECODE SCALE PHI TUYẾN VÀ PHỤ THUỘC KÍCH THƯỚC MÔ HÌNH.

## Paper2: STOCHASTOK
https://www.alphaxiv.org/overview/2506.01687

Mỗi seq đầu vào và xác xuất p.
Với p = 0.1 (default), nếu câu có 10 tokens thì sẽ expand 1 lần
Với câu có 20 tokens thì expand 2 lần

!!! Như vậy cũng có thể ngẫu nhiên merge 2 tokens lại để có được phiên bản 2-gram => TKNZ linh hoạt !!!

**=> Kết hợp OT và STOCHASTOK**

!!! Ta có thể huấn luyện cho model hiểu từ vựng có độ phân giản mịn hơn (tách 1 token làm 2 tokens) hoặc đô phân giải thô hơn(sử dụng 2-gram như là 1 token) bằng cách ngẫu nhiên tăng hoặc giảm độ phân giải của chuỗi đầu vào !!!

## Paper3: VocabCurr - Scaling LLM Pre-training with Vocabulary Curriculum
- https://ar5iv.labs.arxiv.org/html/2502.17910
- https://www.alphaxiv.org/abs/2502.17910
- Entropy-Guided Vocabulary Updates
![](https://ar5iv.labs.arxiv.org/html/2502.17910/assets/better-scale-vocab-curriculum-1.png)

ban đầu mô hình học xử lý ký tự và các đơn vị nhỏ (giúp nắm chắc cấu trúc cơ bản), về sau dần “nâng cấp” lên các token lớn hơn cho những mẫu phổ biến. Yu và cộng sự cho biết cách làm này giúp mô hình GPT nhỏ đạt bpc (bits-per-character) thấp hơn ~6.7% so với mô hình dùng vocab cố định cùng kích thước. Hơn nữa, khi tăng gấp đôi kích thước vocab, mô hình thích ứng thu được hiệu quả cải thiện cao hơn ~34% so với mô hình truyền thống (tức là tận dụng vocab lớn tốt hơn). Kết quả cũng cho thấy một hệ thống phân cấp token tự nhiên hình thành: các token dài dần xuất hiện để đại diện cho các cụm từ phổ biến, dễ dự đoán, còn những đoạn nội dung khó dự đoán thì vẫn bị phân nhỏ thành token ngắn hơn để mô hình xử lý chi tiết. Điều này khớp với trực giác rằng tokenization động cho phép mô hình phân bổ tài nguyên tính toán hợp lý hơn – dành nhiều “não” hơn cho phần phức tạp, bớt tốn sức cho phần đơn giản.

## Paper4: ADATOK
- https://proceedings.neurips.cc/paper_files/paper/2024/file/cdf00c97c0cb2cc35179f03363da6c4f-Paper-Conference.pdf
![](https://pbs.twimg.com/media/GtS0C4ybMAItNaH?format=jpg&name=large)

ADAT bắt đầu với một từ vựng khổng lồ gồm 150 nghìn tokens được tạo ra bằng các thuật toán truyền thống như Unigram hoặc BytePiece. Mục tiêu là thu gọn từ vựng này xuống còn 50 nghìn tokens thông qua năm vòng lặp cắt tỉa liên tiếp.

Trong mỗi vòng lặp, quá trình diễn ra qua bốn bước chính. Đầu tiên, một mô hình LLM nhỏ được khởi tạo ngẫu nhiên và huấn luyện trên 0.3 tỷ tokens với từ vựng hiện tại. Tiếp theo, mô hình này thực hiện inference trên 0.1 tỷ tokens để thu thập dữ liệu về hiệu suất của từng token.

Hệ thống sau đó tính toán hai loại loss quan trọng cho mỗi token. Loss tần suất Unigram LP(xi) được tính bằng công thức `LP(V) - LP(V-xi)`, phản ánh tầm quan trọng của token theo góc độ thống kê. Loss hiệu suất LLM `LM(xi)` được tính bằng tổng các giá trị cross-entropy `CE(M(xi-1), xi)`, **đánh giá khả năng của token trong việc giúp mô hình dự đoán chính xác**. Hai loại loss này được kết hợp thành điểm số cuối cùng theo công thức `L(xi) = LP(xi) / λ * logarit(LM(xi)) + 1`.

Cuối cùng, các tokens được **xếp hạng theo điểm số giảm dần và 20% tokens có điểm thấp nhất sẽ bị loại bỏ**. Quá trình này lặp lại qua năm vòng, từ 150 nghìn tokens ban đầu giảm dần xuống 120 nghìn, 96 nghìn, 77 nghìn, 62 nghìn và cuối cùng là 50 nghìn tokens.

**Điểm đặc biệt của phương pháp này là việc đánh giá tokens dựa trên cả tần suất xuất hiện lẫn khả năng thực sự giúp mô hình dự đoán tốt hơn, thay vì chỉ dựa vào thống kê tần suất đơn thuần như các phương pháp truyền thống.**

---

CẤU TRÚC DỮ LIỆU ĐỂ EMBED VÔ HẠN TOKENS
---------------------------------------

Mô tả bài toán: tôi có một số lượng rất lớn L tokens ti với i = 1..L, và số đếm count_ti tương đương được thống kê từ tập dữ liệu huấn luyện (training data). Vấn đề là tôi chỉ được phép lưu trữ S embeddings với S nhỏ hơn L rất nhiều. Tôi dự định tạo một hàm hash để mapping L tokens vào S embedding slots, sao cho các tokens có count lớn sẽ ít bị trùng lặp hơn những tokens có count thấp. Nói cách khác là luôn ưu tiên các tokens có count lớn hơn được sử dụng embeddings riêng, và các tokens có counts thấp sẽ buộc phải dùng chung nhiều hơn.

Câu hỏi: có cấu trúc dữ liệu hay hàm hashing nào phù hợp cho đề bài của tôi?

o3-pro: https://chatgpt.com/share/684e6fa5-ef94-8003-988d-13d8fdf2b118
![](https://pbs.twimg.com/media/Gte4qHwb0AIIomp?format=png&name=large)

```py Thuật toán “Frequency‑Aware Slotting”
N = 4*(1024**3)						# N = 4M là tổng số embeddings
K = 2*(1024**3)  					# K = 2M là số tokens cho tầng 1, ko va chạm
B = N - K 							# Số embeddings còn lại
									# dim=1024, bfloat16 		=>  4K bytes / feat vector
E1 = Embedding(K, dim)				# full dim					=>  8GB
E2 = Embedding(B, dim//2)			# half dim (concat => full) =>  4GB
									# E1 + E2					=> 12GB RAM 
def slot(token_id):  				# dim=2048 					=> 24GB RAM
    if token in < K:
        return E[token_id]      	# Level 1: no collision
    else: 							# 
        a = murmur32(token_id)%B 	# băm 1 lần xác xuất va chạm 1/B   => 1/2M
        b = farmhash64(token_id)%B 	# băm 2 lần xác xuất va chạm 1/B^2 => 1/4B
        return concat(E2[a], E2[b]) # Level 2: 1/B va chạm 1 phần, 1/B^2 va chạm toàn phần
```
* `B = S − K` là kích thước bảng hash.
* concat E[h1] và E[h2]
* Tăng K → giảm va chạm

---

## Paper5: VEGAD - Vocabulary Expansion via GrADients
- https://www.alphaxiv.org/abs/2410.01188
|![](https://ar5iv.labs.arxiv.org/html/2410.01188/assets/x2.png)|![](https://ar5iv.labs.arxiv.org/html/2410.01188/assets/x3.png)|
|-|-|
**trực giác**: "các nhóm token thể hiện gradient lớn hơn trong các instance lĩnh vực được coi là quan trọng hơn đối với nhiệm vụ và nên được tích hợp vào từ vựng như các thuật ngữ chuyên lĩnh vực". **NOTE**: với VEGAD các candidate words đã được đề xuất từ trước, VEGAD chỉ việc chọn "mọi chuyện trở nên đơn giản" là chọn cái nào có sự ảnh hưởng cao nhất, mà ảnh hưởng ở đây đo bằng grad_score. **Với Paper3 VocabCurr thì họ phải tự đề xuất các candidates dựa trên token logits / loss.**

Tính gradient cho cả hai thành phần quan trọng:
- Embedding layer: G^embed = ∂L_lm/∂α
- Language modeling head layer: G^lmhead = ∂L_lm/∂β

Tác giả phát hiện rằng "tầng language modeling head cũng quan trọng đặc biệt đối với các nhiệm vụ sinh văn bản", điều mà logits đơn thuần không thể capture được. Trong phần ablation study, khi loại bỏ gradient của LMHead layer, "đối với các bộ dữ liệu yêu cầu sinh văn bản, 'w/o LMHead' bị giảm đáng kể". Điều này chứng minh gradient từ LMHead layer mang thông tin quan trọng mà logits thông thường không có.


Ma trận β ∈ R^(L×C) là một trick kỹ thuật rất thông minh mà tác giả sử dụng để tính gradient cho từng token riêng biệt. Trong ký hiệu này, L đại diện cho độ dài của chuỗi (length of sequences) và C là kích thước từ vựng vanilla (size of vanilla vocabulary), β chính là ma trận phụ trợ được thiết kế để tính gradient.

Thông thường, công thức tính logits trong language modeling là ŷ = h × LMHead^T, trong đó h ∈ R^(L×d) là hidden states từ transformer, LMHead ∈ R^(C×d) là ma trận language modeling head, và ŷ ∈ R^(L×C) là logits cho mỗi position và mỗi token trong vocab. Tuy nhiên, để có thể tính gradient riêng cho từng token tại từng vị trí, tác giả đã modify công thức này thành ŷ = β ⊗ (h × LMHead^T).

Điểm quan trọng là β ∈ R^(L×C) được "filled with 1", tức là toàn bộ các giá trị trong ma trận đều bằng 1. Vì vậy, khi thực hiện element-wise multiplication, kết quả `β ⊗ (h × LMHead^T)` vẫn bằng `h × LMHead^T`, không làm thay đổi giá trị logits ban đầu. Tuy nhiên, trick này cho phép họ tính được `G^lmhead = ∂L_lm/∂β`, tức là gradient của loss function theo ma trận β.

Khi tính `∂L_lm/∂β`, tác giả thu được gradient cho từng vị trí và từng token, nghĩa là `G^lmhead[i,j] = ∂L_lm/∂β[i,j]` cho biết token j tại position i ảnh hưởng như thế nào đến loss. Ví dụ, với chuỗi "口服 降压 药" có độ dài L=3 và vocab size C=50000, **ma trận β sẽ có `3 hàng` (mỗi hàng cho một position) và `50000 cột` (mỗi cột cho một token trong vocab), tất cả đều có giá trị 1**.

1. Gradient ≠ Độ Khó Đoán
Xét ví dụ từ bài báo:

- Gradient CAO: 痔疮|Hemorrhoids, 腰椎|Lumbar spine, 甲亢|Hyperthyroidism
- Gradient THẤP: 院去, 下用, 等情, 下才, 本是, 来后...

Các từ có **gradient cao là thuật ngữ y học có ý nghĩa**, `không nhất thiết "khó đoán"`. Ngược lại, các fragment có gradient thấp như "院去", "下用" có thể khó đoán hơn nhưng không quan trọng cho domain.

2. Mục Tiêu Khác Nhau
- `Logits`: Phản ánh "token nào khó dự đoán trong context hiện tại"
- `Gradient`: Phản ánh "token nào quan trọng cho việc tối ưu hóa loss trên batch data"

3. Làm thế nào để tính gradient trên toàn bộ dataset?
VEGAD Có Cơ Chế Accumulation
```py
for (X, Y) ∈ D do  # Lặp qua TOÀN BỘ dataset
    x, y ← GetInputOutput(X, Y)
    # Tính gradient cho batch này
    Calculate G^embed, G^lmhead by Equation 8  
    # TÍCH LŨY gradient
    Gw = Gw + ||∑G^embed||2 + ||∑G^lmhead||1
end for
```
- Mỗi step: Gradient chỉ từ batch hiện tại
- VEGAD strategy: "Gw = Gw +" - tích lũy gradient qua nhiều batches
- Kết quả cuối: Gradient được aggregate từ toàn bộ training data

=> GRADIENT (SAU ACCUMULATION) PHẢN ÁNH "TOKEN NÀO QUAN TRỌNG CHO VIỆC TỐI ƯU HÓA LOSS TÍCH LŨY QUA QUÁ TRÌNH TRAINING", 
TRONG KHI LOGITS CHỈ PHẢN ÁNH ĐỘ KHÓ ĐOÁN TẠI TỪNG PREDICTION CỤ THỂ.

4. Tính grad score cho các word candidates

- Khởi tạo: Gwi ← 0 cho tất cả words trong candidate vocabulary
- Loop qua dataset: for (X, Y) ∈ D do
- Accumulate: Gw = Gw + ||∑G^embed||2 + ||∑G^lmhead||1
- Kết quả: Mỗi word có một gradient score tổng hợp

5. Từ Token Gradient → Word Gradient
```
Token sequence: [降, 压, 药] 
→ Word: "降压药" (antihypertensive drugs)
→ Gradient: Sum of individual token gradients
```
- Gw = Gw + ||∑(q=i to j) G^embed_q||2 + ||∑(q=i-1 to j-1) G^lmhead_q||1
- ||...||2: Áp dụng L2 norm (Euclidean norm)
- ||...||1: Áp dụng L1 norm (Manhattan norm)

Word "降压药" gồm 3 tokens: [降, 压, 药] tại positions [5,6,7]
```
G_embed_sum = G^embed_5 + G^embed_6 + G^embed_7
G_lmhead_sum = G^lmhead_4 + G^lmhead_5 + G^lmhead_6  # position shift
G_降压药 += ||G_embed_sum||2 + ||G_lmhead_sum||1

L2 norm: "Magnitude" của vector gradient - đo "strength" tổng thể
L1 norm: Tổng absolute values - less sensitive to outliers
```

6. Tại Sao Cần Position Shift?
```
Input:  [CLS] 口服  降   压   药   [SEP]
         0    1    2    3    4     5
         ↓    ↓    ↓    ↓    ↓
Target: 口服  降   压   药   [SEP]  
         0    1    2    3    4
```
- Position 0 trong input ([CLS]) → dự đoán "口服" (position 0 trong output)
- Position 1 trong input (口服) → dự đoán "降" (position 1 trong output)
- Position 2 trong input (降) → dự đoán "压" (position 2 trong output)

Position shift đảm bảo chúng ta capture cả hai khía cạnh:
- `Understanding` (embeddings): Gradient khi model "hiểu" token đó trong input
- `Generation` (lm_head): Gradient khi model "sinh ra" token đó trong output

## IMPORTANT
Paper5 cũng chỉ ra rằng “VEGAD+2-gram” outperforms VEGAD, lý lo là vì 2-gram tối ưu hơn cho việc giảm gradient của toàn bộ training dataset. => Tiếp tục mở rộng ra 3-gram, ... n-gram sẽ giúp tối ưu cho việc giảm gradient hơn nữa. Đây là một hướng đi mà paper5 chưa phám khá.

|![](https://pbs.twimg.com/media/GtilNRna4AAuW8z?format=jpg)|![](https://pbs.twimg.com/media/GtimanPboAAToM8?format=jpg)|
|-|-|


VÍ DỤ VỀ KẾT HỢP Ý TƯỞNG N-GRAM EMBEDDINGS TRONG Paper1 với GRADIENT BASED SCORE TRONG Paper5
---------------------------------------------------------------------------------------------

Hiện tại VEGAD có một vấn đề lớn. Nó hoạt động theo kiểu "làm sau", nghĩa là trước tiên huấn luyện mô hình với từ đơn, sau đó mới chọn ra những cụm từ tốt và khởi tạo lại embedding cho chúng. Cách này tạo ra vấn đề "khởi đầu lạnh" - những cụm từ mới không có nền tảng gì để bắt đầu học.

Ý tưởng của bạn lấy cảm hứng từ Over-Tokenized Transformer, nhưng áp dụng theo cách thông minh hơn. Thay vì chỉ đơn giản cộng các embedding lại, bạn đề xuất sử dụng những trọng số có thể học được cho từng mức độ cụm từ. Ví dụ như embedding cuối cùng sẽ là tổng hợp từ embedding từ đơn, cộng với embedding cụm hai từ nhân với trọng số alpha hai, cộng với embedding cụm ba từ nhân với trọng số alpha ba.

Cách tiếp cận này có nhiều ưu điểm tuyệt vời. Đầu tiên, mô hình có thể tự động phát hiện những cụm từ hữu ích ngay trong quá trình huấn luyện, thay vì phải đợi đến sau. VEGAD gradient sẽ hướng dẫn việc điều chỉnh các trọng số này, tạo ra cơ chế tự động đánh trọng số tầm quan trọng của các cụm từ.

Thứ hai, không còn vấn đề khởi đầu lạnh nữa vì các embedding cụm từ được khởi tạo từ những từ đơn ngay từ đầu, tạo nền tảng ý nghĩa và cải tiến dần qua các tín hiệu gradient. Thứ ba, mô hình học theo cấp bậc - đồng thời hiểu từ đơn cơ bản, nhận biết mẫu cụm hai từ, và hình thành khái niệm cụm ba từ.

Về mặt thiết kế, quá trình khởi tạo sẽ tạo embedding cụm từ từ các kết hợp từ đơn thông qua hàm tổng hợp có thể học được. Quá trình huấn luyện sẽ sử dụng phương pháp tính gradient kiểu VEGAD để xác định trọng số. Điều này giải quyết ba vấn đề lớn: VEGAD từ việc chọn lọc sau chuyển thành học trực tuyến, vấn đề bùng nổ từ vựng của Over-Tokenized được kiểm soát bằng cắt tỉa dẫn dắt gradient, và nhiễu từ cụm từ thuần túy được giảm thiểu thông qua ràng buộc tổ hợp.

Cách triển khai có thể bao gồm một module phân cấp với embedding cơ sở, trọng số cụm từ có thể học, và các tầng tổng hợp cho từng mức cụm từ. Phương pháp này kết hợp những điểm mạnh từ cả hai cách tiếp cận: lợi ích mở rộng từ vựng từ Over-Tokenized và cơ chế lựa chọn thông minh từ VEGAD.

Tuy nhiên vẫn còn những thách thức cần giải quyết, bao gồm thiết kế hàm tổng hợp để kết hợp từ đơn thành cụm từ, tính toán gradient để truyền ngược qua cấu trúc phân cấp, và hiệu quả bộ nhớ để xử lý không gian cụm từ lớn. Dù vậy, đây có thể là một đóng góp quan trọng vì bạn đã xác định được hướng nghiên cứu đầy hứa hẹn kết hợp sức mạnh của cả hai phương pháp.

YÊU CẦU CỤ THỂ
--------------

1. Đọc kỹ nội dung toàn bộ papers được liệt kê ở trên bao gồm paper1 tới paper5, tìm kiếm thêm các thông tin về chúng và các nội dung liên quan trên Internet để có cái nhìn tổng quát nhất về lĩnh vực đang nghiên cứu.

2. Xem xét ví dụ về cách kết hợp idea từ 2 hoặc nhiều papers mà người dùng đưa ra.

3. Dựa trên nội dung tìm hiểu được, hãy kết hợp nhiều ý tưởng, phương hướng, cách làm, thuật toán từ các papers tìm hiểu được để đưa ra những ý tưởng và phương pháp mới khiến cho việc xây dụng vocab áp dụng trong tokenization của LLM trở nên hiệu quả hơn trong cả pretraining và inference.

---

# LVOT: Phương Pháp Tối Ưu Hóa Từ Vựng Dựa Trên LLM

Sau khi phân tích kỹ lưỡng 5 bài báo nghiên cứu về tokenization, tôi nhận thấy mỗi phương pháp đều có những điểm mạnh riêng nhưng cũng tồn tại những hạn chế nhất định. LVOT (LLM-based Vocabulary Optimization for Tokenization) là framework tổng hợp nhằm kết hợp những ưu điểm và khắc phục những nhược điểm của các phương pháp hiện có.

## 1. Tokenization Phân Cấp Thích Ứng (HAT)

Ý tưởng đầu tiên xuất phát từ việc kết hợp ba phương pháp: Over-Tokenized, STOCHASTOK và VEGAD. Thay vì cố định sử dụng một cấp độ token duy nhất, hệ thống sẽ linh hoạt chuyển đổi giữa các cấp độ khác nhau.

Trong quá trình huấn luyện, với mỗi đoạn văn bản, hệ thống sẽ ngẫu nhiên chọn sử dụng token đơn, cụm 2 từ, 3 từ hoặc thậm chí 4 từ. Điều này giúp mô hình học được cách biểu diễn ở nhiều độ phân giải khác nhau. Điểm đặc biệt là các trọng số kết hợp giữa các cấp độ được học tự động thông qua gradient, không cần thiết lập thủ công.

Khi inference, hệ thống sẽ tự động chọn cấp độ token phù hợp nhất dựa trên ngữ cảnh. Ví dụ, với thuật ngữ y khoa "viêm phổi cấp tính", hệ thống có thể nhận ra đây là một khái niệm hoàn chỉnh và sử dụng token 4-gram thay vì tách thành các từ đơn lẻ.

## 2. Tiến Hóa Từ Vựng Dựa Trên Gradient (GGVE)

Phương pháp thứ hai kết hợp ba ý tưởng từ VEGAD, ADATOK và VocabCurr để tạo ra một hệ thống từ vựng có khả năng tiến hóa liên tục trong quá trình huấn luyện.

Hệ thống bắt đầu với một từ vựng lớn, sau đó dần dần điều chỉnh dựa trên ba yếu tố chính. Thứ nhất là gradient - những token nào có gradient lớn sẽ được coi là quan trọng và được ưu tiên giữ lại. Thứ hai là entropy - hệ thống sẽ theo dõi độ khó dự đoán của các token để quyết định khi nào cần mở rộng hoặc thu gọn từ vựng. Thứ ba là hiệu suất thực tế của mô hình khi sử dụng các token khác nhau.

Điểm đột phá là thay vì cố định từ vựng sau khi xây dựng, hệ thống cho phép từ vựng thay đổi động trong suốt quá trình huấn luyện. Những token mới hữu ích có thể được thêm vào, trong khi những token ít được sử dụng sẽ bị loại bỏ.

## 3. Từ Vựng Vô Hạn Tiết Kiệm Bộ Nhớ (MEIV)

Vấn đề lưu trữ embedding cho hàng triệu token là một thách thức lớn. MEIV giải quyết vấn đề này bằng cách phân chia không gian lưu trữ thành ba cấp độ.

Cấp độ đầu tiên dành cho những token quan trọng nhất - chúng có embedding riêng với kích thước đầy đủ. Cấp độ thứ hai cho những token phổ biến vừa phải - mỗi token được biểu diễn bằng cách kết hợp hai embedding kích thước một nửa. Cấp độ thứ ba cho những token hiếm - sử dụng bốn embedding kích thước một phần tư.

Điều thú vị là hệ thống có một mạng neural nhỏ đóng vai trò "bộ định tuyến", tự động quyết định mỗi token thuộc cấp độ nào dựa trên đặc điểm của nó. Cách tiếp cận này cho phép xử lý số lượng token gần như vô hạn mà chỉ cần bộ nhớ hữu hạn.

## 4. Mạng Tokenization Đa Độ Phân Giải (MRTN)

MRTN là sự kết hợp toàn diện của ba phương pháp trên, tạo thành một hệ thống hoàn chỉnh. Khi xử lý văn bản, hệ thống không chỉ đơn giản tách thành token mà còn xem xét ngữ cảnh để quyết định độ phân giải phù hợp.

Ví dụ, khi gặp cụm từ "machine learning", trong ngữ cảnh kỹ thuật, hệ thống có thể giữ nguyên như một token. Nhưng trong ngữ cảnh giải thích cho người mới, nó có thể tách thành "machine" và "learning" để dễ hiểu hơn.

Hệ thống cũng có khả năng meta-learning - học cách học. Nó không chỉ học cách tokenize tốt hơn mà còn học cách điều chỉnh chiến lược tokenization cho phù hợp với từng loại dữ liệu.

## 5. Tokenization Thích Ứng Theo Lĩnh Vực (DAT)

Phương pháp cuối cùng tập trung vào việc tạo ra các "chuyên gia" tokenization cho từng lĩnh vực cụ thể. Khi hệ thống nhận ra đang xử lý văn bản y khoa, nó sẽ kích hoạt module chuyên biệt đã được huấn luyện trên dữ liệu y khoa.

Mỗi chuyên gia lĩnh vực không chỉ có từ vựng riêng mà còn có cách hiểu và xử lý văn bản khác nhau. Chuyên gia y khoa sẽ nhận biết "viêm phổi" là một khái niệm, trong khi chuyên gia văn học có thể xem đó là hai từ riêng biệt.

## Ưu Điểm Tổng Thể của LVOT

LVOT mang lại nhiều cải tiến quan trọng. Thứ nhất, nó linh hoạt hơn các phương pháp hiện tại - có thể thích ứng với nhiều loại văn bản và ngữ cảnh khác nhau. Thứ hai, nó hiệu quả về mặt bộ nhớ - có thể xử lý từ vựng cực lớn mà không cần lưu trữ tất cả. Thứ ba, nó học liên tục - không ngừng cải thiện trong quá trình sử dụng.

Quan trọng nhất, LVOT tạo ra một framework thống nhất, cho phép kết hợp những ý tưởng tốt nhất từ các nghiên cứu hiện có trong khi khắc phục những hạn chế của từng phương pháp riêng lẻ. Đây là bước tiến quan trọng hướng tới việc xây dựng hệ thống tokenization thông minh và hiệu quả hơn cho các mô hình ngôn ngữ lớn.
