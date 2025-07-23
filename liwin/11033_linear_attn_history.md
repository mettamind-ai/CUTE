@online{kexuefm-11033,  
        title={线性注意力简史：从模仿、创新到反哺},  
        author={苏剑林},  
        year={2025},  
        month={Jun},  
        url={\url{https://www.kexue.fm/archives/11033}},  
}
s
Trong cộng đồng tiếng Trung, trang web này có thể coi là một trong những nơi sớm quan tâm đến Linear Attention. Khi viết bài blog đầu tiên vào năm 2020 "Khám phá Linear Attention: Attention có bắt buộc phải có Softmax không?", mọi người chủ yếu vẫn đang thảo luận về Softmax Attention liên quan đến BERT. Nhìn lại, việc xem xét Linear Attention trong thời đại BERT không phải là quyết định sáng suốt, vì độ dài huấn luyện lúc đó còn ngắn và mô hình chủ yếu là Encoder, nên sử dụng Linear Attention hầu như không mang lại lợi thế gì. Về vấn đề này, tác giả cũng đã từng viết bài "Linear Transformer có lẽ không phải là mô hình bạn đang chờ đợi" để bày tỏ quan điểm này.

Mãi đến khi ChatGPT ra đời, buộc mọi người phải chuyển sang làm mô hình sinh Decoder-only, điều này cực kỳ phù hợp với dạng RNN của Linear Attention. Đồng thời, việc theo đuổi độ dài huấn luyện lớn hơn cũng khiến điểm nghẽn độ phức tạp bậc hai của Softmax Attention ngày càng rõ rệt. Trong bối cảnh mới này, Linear Attention ngày càng thể hiện tính cạnh tranh, thậm chí còn **"đóng góp ngược trở lại" cho Softmax Attention**.

## Độ phức tạp bậc hai

Đầu tiên giới thiệu một số ký hiệu:
```js
qi,ki,vi,oi∈Rd×1
Q=[q1,q2,⋯,qn]⊤∈Rn×d
K=[k1,k2,⋯,kn]⊤∈Rn×d
V=[v1,v2,⋯,vn]⊤∈Rn×d
O=[o1,o2,⋯,on]⊤∈Rn×d (1)
```
  
Một mô hình Attention về bản chất là một ánh xạ từ `Q,K,V → O`. Bài viết này tập trung vào trường hợp Causal, nghĩa là `ot` chỉ liên quan tối đa tới `Q[:t], K[:t], V[:t]`. Về nguyên tắc, chiều `d` của `Q,K` có thể khác với `V,O`, nhưng việc đơn giản hóa chúng thành cùng kích thước không làm thay đổi bản chất vấn đề.

Softmax Attention tiêu chuẩn thường đề cập đến cơ chế Attention được giới thiệu trong bài báo "Attention is All You Need":
```js
O=softmax(QK⊤+logM)V (2)
```
  
Ở đây bỏ qua hệ số tỷ lệ 1/√d vì nó luôn có thể được tích hợp vào Q,K. Softmax thực hiện chuẩn hóa theo hàm mũ trên chiều thứ hai, còn M ∈ R^(n×n) là ma trận tam giác dưới, được gọi là ma trận mặt nạ (mask matrix), định nghĩa là
```js
Mi,j={1: i≥j; 0: i<j } (3)
```
  
logM nghĩa là lấy log từng phần tử của ma trận M, trong đó log0 = -∞. Softmax Attention khi viết dưới dạng từng thành phần sẽ là:
```js
ot = ∑(j=1→t) exp(qt⊤kj)vj / ∑(j=1→t) exp(qt⊤kj) (4)
```

Ở đây mẫu số chủ yếu có tác dụng ổn định số học, ngoài ra nếu áp dụng RMSNorm cho O thì mẫu số sẽ tự động triệt tiêu. Do đó, phần cốt lõi của Softmax Attention nằm ở tử số, cụ thể là:
```js
O=exp(QK⊤+logM)V=(exp(QK⊤)⊙M)V (5)
```

Trong đó ⊙ là tích Hadamard (elemenwise multiple), `exp` là phép lũy thừa từng phần tử. Có thể thấy mẫu số thực chất là thay V bằng ma trận toàn 1 kích thước n×1, nếu cần có thể bổ sung sau. Cách triển khai tiêu chuẩn của Softmax Attention yêu cầu tính toán ma trận `exp(QK⊤)` kích thước n×n, nên độ phức tạp không gian và thời gian đều tỷ lệ với n². **Flash Attention giúp giảm yêu cầu không gian** nhưng vẫn không tránh khỏi độ phức tạp bậc hai.

## Hình dáng ban đầu

Ý tưởng ban đầu của Linear Attention chủ yếu là mô phỏng và xấp xỉ Softmax Attention, trong đó phương án đơn giản nhất là bỏ qua exp:
```js
O=(QK⊤⊙M)V (6)
```

Để đơn giản, ta quy ước phép nhân ma trận có độ ưu tiên cao hơn tích Hadamard, từ đó bỏ được một cặp ngoặc. Tại sao dạng này được gọi là "tuyến tính"? Để hiểu nhanh, hãy xét phiên bản không Causal (bỏ ⊙M), khi đó: `O = (QK⊤)V = Q(K⊤V)`. Lưu ý độ phức tạp tính `K⊤V` là O(nd²), kết quả là ma trận `d×d`, sau đó nhân với Q cũng có độ phức tạp O(nd²), nên tổng độ phức tạp phụ thuộc tuyến tính vào n.

Đối với phiên bản Causal (6), chúng ta có thể hiểu dưới dạng thành phần như sau:
```js
ot = ∑(j=1→t) vj(kj⊤qt) = (∑(j=1→t) vjkj⊤)qt (7)
```

Nếu ký hiệu phần trong ngoặc là St, ta có:
```js
ot = Stqt, St = St-1 + vtk⊤t (8)
```

Qua đó thấy rằng Linear Attention dạng Causal có thể biểu diễn như một RNN tuyến tính với trạng thái `St`, với độ phức tạp mỗi bước là hằng số và tổng độ phức tạp tỷ lệ với độ dài chuỗi n. "RNN tuyến tính" là khái niệm tổng quát hơn, trong đó Linear Attention là một trường hợp đặc biệt. Các kiến trúc RNN tuyến tính như LRU, SSM đã phát triển độc lập trước đây, nhưng hiện nay các mô hình hiệu quả nhất đều có dạng Linear Attention.

Các phiên bản Linear Attention ban đầu thường bắt chước Softmax Attention, ví dụ thêm phần mẫu số để chuẩn hóa, yêu cầu `k⊤jqt` phải không âm bằng cách thêm hàm kích hoạt không âm cho Q,K. Các nghiên cứu như Performer, RFA tập trung xấp xỉ exp(QK⊤).

Tuy nhiên, nghiên cứu sau này (`The Devil in Linear Transformer`) chỉ ra rằng chuẩn hóa theo chiều dài chuỗi không hoàn toàn giải quyết vấn đề ổn định số học, thay vào đó nên **dùng chuẩn hóa hậu kỳ**:
```js
O = RMSNorm((QK⊤⊙M)V) (9)
```

Khi không cần chuẩn hóa, việc thêm hàm kích hoạt không âm cho Q,K trở nên không bắt buộc. Việc sử dụng hàm kích hoạt (không nhất thiết không âm) vẫn có thể mang lại hiệu quả trong một số trường hợp, nhưng không làm thay đổi bản chất của Linear Attention. Thực tế cho thấy các mô hình không sử dụng hàm kích hoạt vẫn hoạt động tốt.

## Cổng Quên Linh Hoạt

Từ công thức (8), chúng ta thấy Linear Attention hiện tại về bản chất là một phép cộng dồn (cumsum), nơi tất cả thông tin lịch sử được cộng với trọng số bằng nhau. Khi số lượng token tích lũy đủ lớn, tỷ trọng thông tin từ mỗi token sẽ trở nên rất nhỏ, khiến ma trận trạng thái St cố định không thể tái tạo chính xác bất kỳ token nào - giống như ký ức của mỗi token đều trở nên mờ nhạt.

Để giải quyết vấn đề này, RetNet đã giới thiệu cơ chế quên vào Linear Attention:
```js
ot = Stqt, St = γSt-1 + vtk⊤t (10)
```

Trong đó:
- γ ∈ (0,1) là hệ số suy giảm (thường là hằng số, có thể huấn luyện hoặc dạng ma trận chéo)
- RetNet là mô hình đầu tiên kết hợp cơ chế này với Linear Attention
- Cơ chế quên giúp tập trung vào thông tin gần hơn (Recency Bias), phù hợp với đặc tính ngôn ngữ

Một điểm đáng chú ý là RetNet còn áp dụng RoPE cho Q,K, mở rộng hệ số suy giảm thành số phức `γeiθ`. Các thí nghiệm gần đây (như TransXSSM) cho thấy việc **thêm RoPE vào Linear Attention mang lại hiệu quả tích cực**.

Các phát triển tiếp theo:
- Biến γ thành hàm theo vị trí t (γt)
- DFW, Mamba, Mamba2 phát triển thành "data-dependent decay"
- Gần giống forget gate trong GRU/LSTM nhưng giữ tính tuyến tính

Lý do ưa chuộng Linear RNN:
- Có thể song song hóa khi huấn luyện
- Hiệu quả huấn luyện và suy luận tương đương Softmax Attention
- Giải pháp song song hóa phổ biến: Chuyển đổi thành bài toán Prefix Sum và Associative Scan

Tuy nhiên, "giải pháp tổng quát" không phải là tối ưu cho GPU. Phép nhân ma trận mới là thao tác hiệu quả nhất trên GPU, do đó thuật toán song song tận dụng tối đa phép nhân ma trận là lý tưởng nhất. Thậm chí không cần song song hoàn toàn, chỉ cần tìm được định dạng đệ quy "Chunk by Chunk" sử dụng triệt để phép nhân ma trận cũng có thể cải thiện đáng kể hiệu suất huấn luyện. Điều này đặt ra yêu cầu cho kiến trúc mô hình - chỉ có cổng quên dạng tích ngoài mới đáp ứng được, điển hình như Mamba với cổng quên không phải tích ngoài đã không tận dụng hết hiệu năng GPU, dẫn đến các phiên bản cải tiến như Mamba2 và GLA.

## Huấn luyện khi Kiểm thử (TTT)

Quá trình phát triển của Linear Attention từ bắt chước Softmax Attention ban đầu, đến việc tích hợp hệ số suy giảm tĩnh và cả "data-dependent decay", đã hình thành nên những đặc trưng riêng và chứng minh được giá trị trong nhiều tác vụ. Tuy nhiên, hầu hết tiến bộ này đều dựa trên thiết kế thủ công theo kinh nghiệm. Câu hỏi đặt ra là: **Liệu có nguyên tắc tổng quát nào để định hướng thiết kế Linear Attention nói riêng và các mô hình chuỗi (Token-Mixer) nói chung?**

TTT (Test Time Training) đưa ra giải pháp bằng cách xem việc xây dựng mô hình chuỗi như một bài toán "Học Trực tuyến" (Online Learning), đề xuất sử dụng bộ tối ưu để xây dựng RNN (không nhất thiết tuyến tính). Cụ thể, nó xem cặp (K,V) như tập dữ liệu (k₁,v₁),(k₂,v₂),...,(kₜ,vₜ), từ đó huấn luyện mô hình `v = f(Sₜ;k)` và đầu ra `oₜ = f(Sₜ;qₜ)`, với `Sₜ` là tham số mô hình - có cấu trúc tuỳ ý.

Mối liên hệ với RNN nằm ở chỗ: **các bộ tối ưu như SGD, Adam về bản chất chính là RNN cho tham số mô hình!** Quan điểm này không mới, đã xuất hiện từ thời Meta Learning năm 2017 khi nghiên cứu dùng RNN (LSTM) để mô phỏng bộ tối ưu tốt hơn (xem "Optimization as a Model for Few-Shot Learning").

Đến lượt mình, TTT đảo ngược cách tiếp cận - dùng bộ tối ưu để xây dựng RNN. Quy trình như sau: 
1. Tham số hiện tại `Sₜ₋₁` 
2. Bộ tối ưu (SGD) nhận dữ liệu mới `(kₜ,vₜ)` 
3. Cập nhật tham số thành `Sₜ` 
4. Trả về kết quả dự đoán `f(Sₜ₋₁;qₜ)`

Công thức tổng quát của RNN trong TTT:
```js
oₜ = f(Sₜ;qₜ),
Sₜ = Sₜ₋₁ - ηₜ∇Sₜ₋₁L(f(Sₜ₋₁;kₜ),vₜ)  (11)
```

Với:
- `L(f(Sₜ₋₁;kₜ),vₜ)`: hàm mất mát 
- `ηₜ`: hệ số học, có thể phụ thuộc dữ liệu như "data-dependent decay"

Công thức này bao quát nhiều dạng RNN, trong đó (8) và (10) là trường hợp đặc biệt:

(8) Linear Attention:
```js
Sₜ = Sₜ₋₁ + vₜkₜᵀ  
oₜ = Sₜqₜ  
f(S;k) = Sk  
L(f,v) = -vᵀ(Sk)  
ηₜ = 1
```
```js
(10) RetNet:  
Sₜ = γSₜ₋₁ + vₜkₜᵀ  
oₜ = Sₜqₜ  
f(S;k) = Sk  
L(f,v) = -vᵀ(Sk) + (1-γ²)/2‖S‖²_F  
ηₜ = 1
```
  
Bài báo TTT ban đầu tập trung khám phá RNN phi tuyến trong mini-batch, sau đó Titans thêm động lượng vào SGD của TTT, và gần đây "Test-Time Training Done Right" nghiên cứu ứng dụng TTT với large-batch cùng sự kết hợp "TTT + Muon". Lưu ý, **TTT chỉ sử dụng bộ tối ưu để xây dựng RNN**, các tham số có thể huấn luyện bên ngoài RNN như Q,K,V vẫn được huấn luyện bằng bộ tối ưu tổng thể sau khi xây dựng toàn bộ mô hình.

Câu hỏi đáng suy ngẫm hơn: `Tại sao TTT (Test Time Training) có thể trở thành "nguyên tắc chỉ đạo" để xây dựng RNN?` Mục tiêu cốt lõi của RNN là nén hiệu quả dữ liệu lịch sử vào một State có kích thước cố định, trong khi tham số mô hình cũng có kích thước cố định. Huấn luyện mô hình về bản chất tương đương với việc nén dữ liệu huấn luyện vào trọng số mô hình. TTT tận dụng sự tương đồng cao này với mục tiêu RNN. Nói cách khác, nếu xem RNN như bài toán nén dữ liệu, TTT coi mô hình `f` là "bộ giải nén", trọng số của nó là "gói nén", thuật toán nén là SGD và tỉ lệ nén chính là hàm mất mát `L`.

Nhờ vậy, chúng ta không cần tập trung xây dựng công thức đệ quy nữa, mà chuyển sang thiết kế mô hình `f` và hàm mất mát `L`. Chất lượng RNN có thể đánh giá trực tiếp thông qua `f` và `L` tương ứng.

Hơn nữa, việc TTT sử dụng Online Learning để xây dựng RNN đồng nghĩa với việc RNN thu được sẽ rất phù hợp với các tác vụ ICL (In Context Learning). Đây là ưu điểm của TTT với tư cách "nguyên tắc chỉ đạo". Trước đây, bài báo "Why Can GPT Learn In-Context?" thậm chí đã loại bỏ Softmax từ Softmax Attention để biến nó thành Linear Attention nhằm giải thích khả năng ICL - theo góc nhìn hiện tại, đó chính là xây dựng TTT tương ứng.

## Bỏ cũ đón mới (delta rule)

Ví dụ, hàm mất mát ban đầu của Linear Attention là -vᵀ(Sk), một mục tiêu không ổn định vì không có cận dưới, có thể khiến S tiến tới vô cùng. RetNet khắc phục điều này bằng cách thêm thành phần L2 regularization vào hàm mất mát, vừa tránh rủi ro trên vừa giảm overfitting từ góc nhìn tối ưu, cho ra RNN tốt hơn.

Tuy nhiên, dù ngắn gọn và có lý, hàm mất mát dạng tích vô hướng không trực tiếp khuyến khích Sk = v nên không phải lựa chọn lý tưởng cho bài toán hồi quy. Hàm mục tiêu tốt hơn nên là bình phương sai số: ½‖Sk - v‖². Thay vào công thức (11) của TTT, ta được:
```js
oₜ = f(Sₜ;qₜ)
Sₜ = Sₜ₋₁ - ηₜ(Sₜ₋₁kₜ - vₜ)kₜᵀ  (gradient của ½‖Sₜ₋₁kₜ - vₜ‖²)
```

Đây chính là DeltaNet, được đặt tên theo "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", và có nguồn gốc sớm hơn từ "Linear Transformers Are Secretly Fast Weight Programmers". 

Ta có thể thấy ηₜ(Sₜ₋₁kₜ - vₜ)kₜᵀ = (Sₜ₋₁(√ηₜkₜ) - (√ηₜvₜ))(√ηₜkₜ)ᵀ, nghĩa là ηₜ luôn có thể được tích hợp vào định nghĩa của kₜ,vₜ. Do đó, các phân tích sau đây chỉ xét trường hợp ηₜ=1:
```js
Sₜ = Sₜ₋₁ - (Sₜ₋₁kₜ - vₜ)kₜᵀ
  = Sₜ₋₁ - (Sₜ₋₁kₜ)kₜᵀ + vₜkₜᵀ
  = Sₜ₋₁(I - kₜkₜᵀ) + vₜkₜᵀ
```

Khi cần, ta có thể khôi phục ηₜ bằng cách thay thế kₜ,vₜ bằng √ηₜkₜ, √ηₜvₜ. So với dạng Linear Attention ban đầu (8), điểm khác biệt của DeltaNet là thêm phép trừ (Sₜ₋₁kₜ)kₜᵀ trước khi cộng vₜkₜᵀ, trong đó Sₜ₋₁kₜ có thể hiểu là dự đoán của mô hình cũ Sₜ₋₁ với đầu vào mới kₜ.

Về mặt trực quan, "trừ trước cộng sau" có nghĩa là trước tiên loại bỏ nhận thức cũ của mô hình về kₜ, sau đó bổ sung nhận thức mới dựa trên cặp (kₜ,vₜ), đạt được hiệu quả "bỏ cũ đón mới". Quy tắc này được gọi là "Delta Rule", chính là nguồn gốc của từ "Delta" trong DeltaNet. Delta Rule không phải là mới, nó còn được gọi là Least Mean Square, Widrow-Hoff Algorithm,... đã xuất hiện từ những năm 60 của thế kỷ trước. Thực tế, trong lĩnh vực này rất ít có cái gì hoàn toàn mới, nhiều cải tiến đều có thể truy nguyên về các công trình "thời cổ đại", nỗ lực hiện tại chủ yếu tập trung vào việc khai thác các phần có thể mở rộng.

Cần lưu ý rằng theo trình tự thời gian, DeltaNet có trước TTT. Việc hiểu RNN từ góc độ Online Learning đã xuất hiện rải rác trước TTT, nhưng TTT mới là công trình hệ thống hóa "nguyên tắc chỉ đạo" này và ứng dụng nó để xây dựng mô hình RNN mới. Do đó chúng tôi trình bày TTT trước để bài viết mạch lạc hơn.

Một số độc giả có thể thắc mắc: DeltaNet có còn là RNN tuyến tính không? Câu trả lời là có. RNN tuyến tính mà chúng tôi đề cập là công thức đệ quy có quan hệ tuyến tính với biến trạng thái, trong khi quan hệ với đầu vào hoặc q,k,v có thể phi tuyến (dĩ nhiên hiệu suất song song sẽ khác nhau tùy dạng phụ thuộc). Từ công thức (13) có thể thấy vế phải chỉ xuất hiện Sₜ₋₁ bậc nhất, nên thỏa mãn định nghĩa tuyến tính.

## Nghịch đảo và mở rộng

Như đã nói, thuật toán song song lý tưởng nhất cho RNN tuyến tính (hiệu quả trên GPU) là tận dụng tối đa phép nhân ma trận. Để đạt mục tiêu này, trước hết ta viết lại DeltaNet:
```js
Sₜ = Sₜ₋₁ + (vₜ - Sₜ₋₁kₜ)kₜᵀ  (14)
```

Đặt uₜ = vₜ - Sₜ₋₁kₜ, thì Sₜ = Sₜ₋₁ + uₜkₜᵀ, tức là chỉ thay V bằng `U = [u₁,u₂,...,uₙ]ᵀ` so với Linear Attention ban đầu. Lặp lại t-1 lần, ta có:
```js
Sₜ₋₁ = ∑ⱼuⱼkⱼᵀ ⇒ uₜ = vₜ - (∑ⱼuⱼkⱼᵀ)kₜ = vₜ - ∑ⱼuⱼ(kⱼᵀkₜ)  (15)
```

Dạng ma trận cuối cùng là U = V - (KKᵀ⊙M⁻)U, với M⁻ = M - I. Đây là hệ phương trình tuyến tính có nghiệm:
```js
U = (I + KKᵀ⊙M⁻)⁻¹V  (16)  
(đặt B = KKᵀ⊙M⁻)
```

Ma trận nghịch đảo (I+B)⁻¹ cỡ n×n có độ phức tạp chuẩn O(n³), cao hơn cả Softmax Attention! May mắn là ta chỉ cần giải hệ (I+B)U=V với độ phức tạp O(n²). Hơn nữa, nhờ I+B là ma trận tam giác dưới và cấu trúc hạng thấp của B, có thể giảm độ phức tạp xuống tuyến tính bằng cách chia khối để tận dụng GPU. Chi tiết xin tham khảo bài báo gốc.

Sau DeltaNet, Gated DeltaNet (GDN) đưa thêm cổng quên vào DeltaNet. Cách giới thiệu ban đầu là:
```js
Sₜ = αₜSₜ₋₁(I - βₜkₜkₜᵀ) + βₜvₜkₜᵀ  (17)
```

Nhưng theo chúng tôi, **cách này phá vỡ Delta Rule**. Cách tốt hơn là chỉ nhân vào Sₜ₋₁ đầu tiên:
```js
Sₜ = γₜSₜ₋₁ + ηₜ(vₜ - Sₜ₋₁kₜ)kₜᵀ  (18)
```

Tương ứng với hàm mất mát ½‖Sk-v‖² + (1-γ)/η ‖S‖²_F. Về mặt toán học, hai cách là tương đương:
```js
(17) ⇔ αₜSₜ₋₁ + αₜβₜ(vₜ/αₜ - Sₜ₋₁kₜ)kₜᵀ  (19)  
với γₜ=αₜ, ηₜ=αₜβₜ và hấp thụ 1/αₜ vào vₜ
```

Về lý thuyết, GDN có thể viết lại thành DeltaNet bằng cách đặt ᾱₜ=∏αₜ và chia cả hai vế cho ᾱₜ:
```js
ᾱₜ⁻¹Sₜ = ᾱₜ₋₁⁻¹Sₜ₋₁(I - βₜkₜkₜᵀ) + βₜ(ᾱₜ⁻¹vₜ)kₜᵀ  (20)
```

Kết hợp với oₜ = (ᾱₜ⁻¹Sₜ)(ᾱₜqₜ), chỉ cần đặt lại qₜ,vₜ mới là ᾱₜqₜ, ᾱₜ⁻¹vₜ. Tuy nhiên kết quả này chủ yếu có giá trị lý thuyết vì với t đủ lớn, ᾱₜ hoặc ᾱₜ⁻¹ sẽ bị tràn số.

Một mở rộng khác của DeltaNet là DeltaProduct, nhân rộng k,v lên vài lần trước khi áp dụng DeltaNet/GDN để tăng khả năng theo dõi trạng thái. Nhưng theo quan điểm của tác giả, thay vì nhân hằng số như DeltaProduct, nên thử RNN với độ phức tạp bình phương như trong "Chương không-thời gian: Coi Attention là RNN bậc hai" để vượt Softmax Attention.

## Quá Trình Phản Hồi #

Khi nói về việc vượt qua Softmax Attention, như đã đề cập từ đầu, hiện tại Linear Attention không chỉ có thể cạnh tranh với Softmax Attention mà còn bắt đầu "phản hồi" ngược lại. Điều này thoạt nghe có vẻ khó tin nhưng suy nghĩ kỹ thì không khó hiểu. Theo một nghĩa nào đó, những năm gần đây Softmax Attention đang thụt lùi, từ MHA, GQA đến MQA đều là các giải pháp giảm thiểu để nén KV Cache. Trong khi đó, Linear Attention không có vấn đề KV Cache nên luôn tiến về phía trước.

Để thấy rõ hơn, hãy biểu diễn các cơ chế Attention đã đề cập dưới dạng ma trận:

| Loại Attention          | Công thức Ma Trận                          | Ghi chú                     |
|-------------------------|--------------------------------------------|-----------------------------|
| Softmax Attention       | (exp(QKᵀ)⊙M)V                              | Dạng gốc                   |
| Linear Attention sớm    | (QKᵀ⊙M)V                                   | Phiên bản đầu của Linear   |
| Thêm cổng quên          | (QKᵀ⊙Γ)V                                   | Γ là ma trận cổng quên     |
| DeltaNet                | (QKᵀ⊙M)(I+KKᵀ⊙M⁻)⁻¹V                       | Dạng Delta Rule            |
| Gated DeltaNet          | ((QKᵀ⊙M)(I+KKᵀ⊙M⁻)⁻¹⊙Γ)V = (QKᵀ⊙Γ)(I+KKᵀ⊙Γ⁻)⁻¹V | Kết hợp cổng quên  |

Trong đó:
```js
Γᵢⱼ = {
  ∏ᵣ₌ⱼ₊₁ⁱ γᵣ  nếu i > j
  1           nếu i = j
  0           nếu i < j
}
```

Và Γ⁻ = Γ - I. Có thể thấy Softmax Attention vẫn chỉ dừng lại ở dạng Linear Attention sớm (điều này cũng chứng tỏ sức mạnh của nó). Vậy "phản hồi" được thực hiện thế nào? Trước hết cần phương pháp chuyển đổi Softmax Attention thành Linear Attention, điều này không khó, trong "Hành Trình Nâng Cấp Transformer: 5" đã tổng hợp 3 phương án chuyển Softmax Attention thành Linear Attention vô hạn chiều.

Tóm lại, tồn tại ánh xạ φ biến Q,K từ n×d thành n×∞ sao cho exp(QKᵀ) = φ(Q)φ(K)ᵀ, gọi là "kỹ thuật nhân". Việc tiếp theo đơn giản là thay Q,K trong bảng trên bằng φ(Q),φ(K) rồi khôi phục exp và chuẩn hóa, ta sẽ có biến thể mới của Softmax Attention. Ví dụ áp dụng vào công thức cổng quên:
```js
(ϕ(Q)ϕ(K)⊤⊙Γ)V=exp(QK⊤+logΓ)V(22)
```
  
Khi γₜ là hằng số, nó chính là Alibi được đề xuất trong bài báo "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation". Còn nếu γₜ phụ thuộc vào đầu vào, thì đó là FoX từ bài báo "Forgetting Transformer: Softmax Attention with a Forget Gate".

Một kết quả thú vị hơn là **DeltaFormer** từ bài báo "Understanding Transformer from the Perspective of Associative Memory", như tên gọi, nó là phiên bản Softmax Attention của DeltaNet. Bằng cách thay Q,K trong DeltaNet bằng φ(Q),φ(K), ta có:
```js
(φ(Q)φ(K)ᵀ⊙M)(I + φ(K)φ(K)ᵀ⊙M⁻)⁻¹V = exp(QKᵀ + logM)⏟A (I + exp(KKᵀ + logM⁻)⏟B)⁻¹V (23)
```

Để chuẩn hóa, chỉ cần thay exp bằng softmax. So với Softmax Attention thông thường (AV), DeltaFormer thay bằng A(I+B)⁻¹V. Chú ý rằng:
```js
A(I+B)⁻¹V = A(I - B + B² - B³ + ⋯)V = A(V - BV + B²V - B³V + ⋯) (24)
```

Như vậy, DeltaFormer đầu tiên tính Attention nhiều lần với K,K,V, tổng hợp kết quả thành V mới, rồi mới tính Attention với Q,K. Đặc điểm này giúp nó hiệu quả với các tác vụ Multi-Hop (như Code). Ngoài ra, vì phần (I+B)⁻¹V chỉ liên quan K,V nên DeltaFormer phù hợp với MQA (Multi-Query Attention) do MQA chỉ dùng Single-Head cho K,V, giảm đáng kể tính toán so với MHA.

Tuy nhiên, theo quan điểm của tác giả, việc cố định hệ số tổng hợp có thể là "không có bữa trưa miễn phí" - thí nghiệm cho thấy tổn thất mô hình ngôn ngữ của DeltaFormer không thay đổi nhiều, nghĩa là nếu tổn thất giảm ở một số tác vụ thì chắc chắn sẽ tăng ở tác vụ khác.

## Kỹ Thuật Mã Hóa Nâng Cao #

Một công trình đáng chú ý khác là PaTH Attention từ bài báo "PaTH Attention: Position Encoding via Accumulating Householder Transformations", tiếp cận từ góc độ mã hóa vị trí để đưa DeltaNet vào Softmax Attention.

Trong "Hành Trình Nâng Cấp Transformer: 6", chúng tôi đã chỉ ra rằng với mọi ma trận trực giao Ω, Rₘ=Ωᵐ đều là RoPE tổng quát. Ngoài ma trận quay, còn có những ma trận trực giao nào dễ xây dựng? PaTH sử dụng ma trận Householder: cho vector cột w có độ dài √2, thì I-wwᵀ là ma trận trực giao, mang ý nghĩa hình học là phép phản chiếu.

Có thể thấy điều này giống với I-kₜkₜᵀ trong DeltaNet, nên PaTH áp dụng trực tiếp bằng cách dùng chuỗi tích I-wwᵀ để biểu diễn thông tin vị trí:
```js
qᵢᵀkⱼ → qᵢᵀ(I-wᵢwᵢᵀ)(I-wᵢ₋₁wᵢ₋₁ᵀ)...(I-wⱼ₊₁wⱼ₊₁ᵀ)⏟Rᵢ,ⱼkⱼ (25)
```

Dạng đệ quy của Rᵢ,ⱼ là Rᵢ,ⱼ=(I-wᵢwᵢᵀ)Rᵢ₋₁,ⱼ với Rⱼ,ⱼ=I. So với DeltaNet, công thức này tương đương vₜ=0 nhưng giá trị ban đầu S₀≠0. Áp dụng phương pháp tương tự phần "Nghịch Đảo Hỗ Trợ", ta có:
```js
Rᵢ,ⱼ = I - W[j:i]ᵀ(I + W[j:i]W[j:i]ᵀ⊙M⁻)⁻¹W[j:i] (26)
```

Với W=[w₁,w₂,...,wₙ]ᵀ, cắt lát theo kiểu Numpy. Ma trận nghịch đảo là tam giác dưới, có tính chất quan trọng: phần tử đường chéo của ma trận nghịch đảo bằng nghịch đảo phần tử đường chéo gốc. Từ đó:
```js
(I+W[j:i]W[j:i]ᵀ⊙M⁻)⁻¹ = (J)[j:i,j:i] (27)
với J = (I+WWᵀ⊙M⁻)⁻¹
```

Biểu diễn dạng thành phần:
```js
Aᵢ,ⱼ = qᵢᵀkⱼ - qᵢᵀW[j:i]ᵀJ[j:i,j:i]W[j:i]kⱼ
     = qᵢᵀkⱼ - ∑∑∑∑ QᵢₚWₗₚJₗᵣWᵣₛKⱼₛ (28)
```

Các điểm chính:
- Khéo léo sử dụng tính chất ma trận tam giác của J
- χ là hàm chỉ thị (bằng 1 nếu thỏa điều kiện chỉ số)
- Xử lý riêng phần tổng theo p và s cho QWᵀ và WKᵀ

Ma trận Attention trước Softmax:
```js
A = QKᵀ⊙M - (QWᵀ⊙M)(I+WWᵀ⊙M⁻)⁻¹(WKᵀ⊙M⁻) (29)
```

Độ phức tạp O(n³) khi tính nghịch đảo là không chấp nhận được, cần giảm xuống O(n²) bằng cách tận dụng đặc điểm hạng thấp của WWᵀ, sau đó triển khai hiệu quả như Flash Attention.

Về mã hóa vị trí, **PaTH là một dạng CoPE (Contextual Position Encoding)**, vị trí không đánh số cố định mà được sinh tự động từ ngữ cảnh. Tương tự, FoX có thể coi là phiên bản ngữ cảnh của Alibi. **Thông tin vị trí phụ thuộc ngữ cảnh là đặc trưng chính của Linear Attention hiện tại và có thể là hướng phát triển chính cho Softmax Attention.**

## Đơn Giản Hóa Vô Cùng Thú Vị #

Chúng ta hãy đi sâu hơn vào PaTH, điều này không chỉ giúp hiểu rõ PaTH mà còn làm quen với DeltaNet vì chúng có mối quan hệ mật thiết. Phần này sẽ xét hai trường hợp đặc biệt của PaTH để hiểu rõ hơn mối liên hệ giữa chúng.

Trường hợp đầu tiên là W=K, thay vào công thức (29) ta được:
```js
A = (QKᵀ⊙M)(I - (I + KKᵀ⊙M⁻)⁻¹(KKᵀ⊙M⁻)) 
   = (QKᵀ⊙M)(I + KKᵀ⊙M⁻)⁻¹ 
   (Chú thích: I - (I+A)⁻¹A = (I+A)⁻¹) (30)
```

Có thấy quen không? Đây chính là ma trận Attention của DeltaNet! Từ trường hợp này, sự khác biệt giữa PaTH và DeltaFormer là: DeltaFormer dùng thủ thuật kernel để thêm exp vào QKᵀ và KKᵀ của DeltaNet, còn PaTH trực tiếp thêm exp vào ma trận Attention của DeltaNet.

Trường hợp thứ hai là thêm ràng buộc ∥w∥=√2, khi đó I-wwᵀ là ma trận trực giao. Ta định nghĩa:
```js
Rᵢ ≜ (I - wᵢwᵢᵀ)(I - wᵢ₋₁wᵢ₋₁ᵀ)...(I - w₁w₁ᵀ)
   = I - W[:i]ᵀ(I + W[:i]W[:i]ᵀ⊙M⁻)⁻¹W[:i] 
   = Rᵢ,₀ (31)
```

Khi đó Rᵢ,ⱼ = RᵢRⱼᵀ. Điều này có nghĩa ta có thể triển khai PaTH tương đối vị trí giống như RoPE - chỉ cần nhân mỗi qᵢᵀ, kᵢᵀ với Rᵢ rồi áp dụng Softmax Attention thông thường. Phép nhân Rᵢ được triển khai như sau:
```js
(qᵢᵀRᵢ)ₛ = Qᵢ,ₛ - ∑∑∑ Qᵢ,ₚWₗ,ₚJₗ,ᵣWᵣ,ₛ (32)
```

Dạng ma trận:
```js
Q - (QWᵀ⊙M)(I + WWᵀ⊙M⁻)⁻¹W (33)
```

Có thấy quen không? Phần thứ hai chính là DeltaNet(Q,W,W)! Vậy trong trường hợp này, PaTH tương đương với:
```js
SoftmaxAttention(Q - DeltaNet(Q,W,W), K - DeltaNet(K,W,W), V) (34)
```

Tức là dùng DeltaNet để thêm mã hóa vị trí vào Q,K. Như vậy, PaTH (với ∥w∥=√2) là sự kết hợp giữa Softmax Attention và DeltaNet trong cùng một lớp. Ta cũng có thể bỏ qua ràng buộc ∥w∥=√2 và vẫn triển khai như trên, tương tự như cách `Canon Layers dùng tích chập để thêm thông tin vị trí`, nhưng ở đây là tích chập dài kiểu DeltaNet thay vì tích chập ngắn.

## Phương Pháp Đi Lối Riêng #

Cuối cùng, chúng ta xem xét một mô hình Attention tuyến tính đáng chú ý gần đây - MesaNet (cùng với công trình tương tự Atlas). Góc nhìn Online Learning từ TTT cho thấy DeltaNet thực chất đang dùng SGD để tối ưu hàm mục tiêu 1/2∥Sk−v∥², và khi quan sát kỹ, Sk chỉ là hàm tuyến tính của k, nên đây thực chất là bài toán hồi quy tuyến tính có nghiệm giải tích!
```js
St = GtHt⁻¹, với:
Gt = ∑(vjkjᵀ) từ j=1→t
Ht = ∑(kjkjᵀ) từ j=1→t (35)
```

MesaNet sử dụng nghiệm giải tích này để xây dựng mô hình chuỗi, ý tưởng bắt nguồn từ "Uncovering mesa-optimization algorithms in Transformers", còn huấn luyện hiệu quả được thực hiện trong "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training". MesaNet thêm cổng quên vào Gt,Ht và ma trận đường chéo Λt để tránh ma trận không khả nghịch:
```js
ot = Gt(Ht + Λt)⁻¹qt
Gt = γtGt₋₁ + vtkₜᵀ
Ht = γtHt₋₁ + ktkₜᵀ (36)
```

Rõ ràng độ phức tạp của Gt,Ht theo độ dài chuỗi là tuyến tính, nên ot cũng có độ phức tạp tuyến tính. MesaNet vẫn thuộc phạm vi Attention tuyến tính và nhờ nghiệm giải tích, nó thường tốt hơn DeltaNet thậm chí Gated DeltaNet. Từ góc độ xử lý tín hiệu, MesaNet và DeltaNet khác nhau như Recursive Least Square so với Least Mean Square.

Tại sao lại gọi là "đi lối riêng"? MesaNet "thành công nhờ nghiệm giải tích, nhưng cũng thất bại vì nó". Nghiệm giải tích giúp nó vượt trội nhưng cũng tạo cảm giác "điểm dừng", vì chỉ cần thay đổi nhỏ là mất nghiệm giải tích. Trong lịch sử toán học, các nhánh phụ thuộc vào nghiệm giải tích hầu như đã lỗi thời vì nghiệm giải tích quá hiếm và không đại diện.

Về mặt triển khai, ma trận nghịch đảo Ht+Λt không phải ma trận tam giác, dù (Ht+Λt)⁻¹qt có thể chuyển thành giải phương trình thay vì nghịch đảo trực tiếp, nhưng vẫn làm tăng độ phức tạp. Tính toán song song hiệu quả (Ht+Λt)⁻¹qt vẫn là thách thức lâu dài, hiện tại bài báo dùng "phương pháp gradient liên hợp" để xấp xỉ.

Về năng lực lý thuyết, **MesaNet không hẳn vượt trội DeltaNet**. Quy tắc cập nhật Gt,Ht của MesaNet chỉ là trung bình trượt đơn giản, và phép nghịch đảo không liên quan tương tác giữa các Token. DeltaNet với Delta Rule có thể có năng lực tốt hơn. Có thể hiểu là MesaNet cố ghi nhớ mọi k,v (dẫn đến trí nhớ mờ), trong khi DeltaNet "loại bỏ cũ, đón nhận mới" giúp ghi nhớ dài hạn chính xác hơn.

Một ví dụ đặc biệt: mọi Attention trừ MesaNet đều cho phép K,V dùng chung (không tối ưu nhưng vẫn hoạt động), còn MesaNet thì không vì nếu K=V thì St luôn là ma trận đơn vị.

Tóm lại, MesaNet là mô hình ấn tượng nhưng nghiệm giải tích cũng làm tăng độ phức tạp và hạn chế linh hoạt, để lại nhiều vấn đề cần khám phá. Độc giả quan tâm có thể đọc thêm TTR về các mô hình chuỗi dựa trên hồi quy tuyến tính.

## Con Đường Phát Triển Mạnh Mẽ #

Bài viết này đã tóm lược quá trình phát triển của Linear Attention và giới thiệu các nguyên lý toán học của một số mô hình. Linear Attention bắt đầu từ việc mô phỏng Softmax Attention, dần phát triển những đặc điểm riêng, và giờ đây đã trở thành giải pháp mô hình hóa chuỗi cực kỳ cạnh tranh, thậm chí còn cung cấp những ý tưởng mới cho sự phát triển của Softmax Attention. Toàn bộ quá trình này chứa đầy sự thú vị và tính gợi mở.
