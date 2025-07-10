
## Grouped-Tied Attention (GTA)

a) Biểu đồ giá trị suy biến cho thấy sự suy giảm mạnh trong bộ nhớ đệm khóa, khi **hầu hết phương sai được nắm bắt bởi một vài hướng chính**, do đó `các khóa nằm trong không gian con hạng thấp và rất dư thừa` (Saxena et al., 2024). Hiệu ứng này còn rõ rệt hơn khi áp dụng RoPE, khi các khóa co lại thành một không gian con nhỏ hơn nữa (Yu et al., 2024a; Sun et al., 2024).

b) Nhiều nghiên cứu cho thấy việc áp dụng RoPE cho một phần chiều đầu vẫn giữ được độ chính xác, nên `xoay toàn bộ chiều rộng cải thiện chất lượng rất ít` (Black et al., 2022; Barbero et al., 2025).

=> Vì các khóa vốn có hạng thấp và chỉ một phần mỗi đầu cần xoay cho mã hóa vị trí, nên chúng ta chỉ cần xoay một phần head dim cho positional embedding; **các kênh không xoay còn lại, thường nằm trong không gian con hạng thấp và dư thừa, có thể được chia sẻ hoặc liên kết với các trạng thái giá trị**.

GQA đã giảm bộ nhớ đệm KV và truyền dữ liệu bằng cách cho nhiều đầu truy vấn chia sẻ một đầu KV riêng biệt, và nó mở rộng hiệu quả trên nhiều thiết bị. Dựa trên thiết kế nhóm của GQA, kết hợp với những hiểu biết về hạng thấp và RoPE từng phần ở trên, chúng tôi đề xuất Grouped-Tied Attention (GTA), **liên kết KV thành một trạng thái duy nhất**, và `áp dụng RoPE một phần`, tất cả nhằm mục đích giảm kích thước bộ nhớ đệm KV trong khi vẫn `giữ được chất lượng`.

Trong GQA, mỗi nhóm query head chia sẻ một đầu KV riêng biệt. GTA tiến xa hơn bằng cách liên kết các tham số chiếu khóa và giá trị để tạo ra một trạng thái duy nhất gọi là KV liên kết, có hình dạng khớp với vector khóa hoặc vector giá trị đơn lẻ. Đường dẫn giá trị sử dụng toàn bộ chiều của trạng thái KV liên kết, trong khi đường dẫn khóa chỉ tái sử dụng nửa đầu làm phần không xoay.

Thành phần RoPE còn lại của khóa đến từ một phép chiếu một đầu riêng biệt của trạng thái ẩn, được broadcast qua tất cả các nhóm và nối với phần không xoay để tạo thành vector khóa đầy đủ. Thực nghiệm cho thấy việc **áp dụng RoPE cho phần chia sẻ làm giảm chất lượng**, do đó phần liên kết không bao giờ bị xoay. Sau các bước này, các trạng thái truy vấn, khóa và giá trị được định nghĩa như sau:
```
Q        ∈ R^(ctxlen, head_q, dim)
KV, K, V ∈ R^(ctxlen, head_kv, dim)
K_RoPE   ∈ R^(ctxlen, 1, dim/2)
K_Nope   ∈ R^(ctxlen, head_kv, dim/2)
K_Nope = KV[..., : dim/2]
V      = KV
K      = concat( K_NoPE, broadcast(K_RoPE, head_kv) )
```
Bằng cách liên kết các trạng thái KV, chúng ta tải một trạng thái duy nhất vào bộ nhớ on-chip, tái sử dụng nó cho cả khóa và giá trị, và chia sẻ nó qua một nhóm nhỏ các đầu truy vấn. Việc tái sử dụng này giảm chuyển dữ liệu bộ nhớ, **tăng gần gấp đôi cường độ tính toán và giảm một nửa dung lượng bộ nhớ đệm KV** so với phiên bản GQA tương ứng với cùng số nhóm. GQA-4 biểu thị 4 đầu khóa và giá trị riêng biệt, trong khi GTA-4 biểu thị 4 đầu KV liên kết. Các thí nghiệm cho thấy độ bất định (xem 5.1.1) và hiệu suất trong các tác vụ downstream (xem 5.1.2) vẫn tương đương với phiên bản GQA tương ứng.

---

Trên các bài kiểm tra, với mô hình medium 433M (trong Bảng 3), GLA-2 đạt độ chính xác trung bình cao nhất là 55.4%, `GLA-2 vượt nhẹ MLA` ở mức 54.9%. Ở quy mô mô hình large 876M (Bảng 4), `GTA-4 nhỉnh hơn GLA-2` (GLA-2 cho độ chính xác trung bình 57.5%, chỉ thấp hơn GTA-4 0.1 điểm và tương đương GQA-4), cho thấy việc nhóm hoặc liên kết không làm giảm chất lượng. Xu hướng này tiếp tục ở quy mô XL 1.471B, nơi GLA-2 đạt 60.0% so với 59.1% của MLA, trong khi `GTA-4 và GQA-4 cùng đạt 60.2%` (Bảng 5). Các kết quả này khẳng định các biến thể tiết kiệm phần cứng của chúng tôi **bảo toàn hoặc cải thiện hiệu suất tác vụ downstream từ quy mô medium tới XL**. Chi tiết đánh giá downstream xem Phụ lục B.2.2. Các thí nghiệm ablation ở quy mô nhỏ và medium xem Phụ lục B.3.

Chúng tôi đánh giá hiệu suất zero-shot trên các benchmark tiêu chuẩn: SciQ (Welbl et al., 2017), OpenBookQA (Mihaylov et al., 2018), ARC-Easy subset (Yadav et al., 2019), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2020) và MMLU (Hendrycks et al., 2021).

| Attn   | Winogrande |   SciQ   |   PiQA   | OpenBookQA |   MMLU   | HellaSwag | Arc Easy |   Avg    |
|--------|------------|----------|----------|------------|----------|-----------|----------|----------|
| GLA𝑞-2 |   55.2     |   84.9   | **70.5** |   35.6     |   25.2   |  _47.9_   | **66.3** |  _55.1_  |
| GQA-4  |   53.8     |  _85.7_  |   69.7   |   36.2     |   25.4   |   46.3    |   64.6   |   54.5   |
| GTA-4  |   54.2     |   85.5   |   69.0   |   34.0     |  _25.9_  |   46.8    |   64.2   |   54.2   |
| MQA    |  _55.5_    |   84.6   |   69.5   |  _37.0_    | **26.2** |   45.9    |   60.5   |   54.2   |
| GLA-2  | **56.7**   |   84.1   |  _70.3_  | **37.2**   | **26.2** | **48.2**  |  _65.3_  | **55.4** |
| MLA    |   54.5     | **86.1** |   70.2   |   36.8     |   25.1   |   47.2    |   64.2   |   54.9   |
| MHA    |   55.2     |   84.8   |   69.3   |   35.0     |   25.5   |   46.2    |   63.0   |   54.1   |

*Bảng 3*: Đánh giá các mô hình `433M`. Độ chính xác cao nhất được in đậm, cao thứ hai được gạch chân.
