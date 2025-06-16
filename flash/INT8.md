sagefwd sử dụng phép zero centering `K - mean(K)` để loại bỏ outliers trong K matrix. "Note that such a transformation does not change the attention score P, because for any query q, we have `softmax(q(K - mean(K))⊤) = softmax(qK⊤ - q · mean(K)) = softmax(qK⊤)`."

Giải thích chi tiết:

- Trong phép nhân QK⊤: `q × (K - mean(K))⊤ = q × K⊤ - q × mean(K)⊤`
- Trong softmax: `q × mean(K)⊤` tạo ra một hằng số được trừ khỏi tất cả các score trong cùng một hàng
- Tính chất softmax: `softmax([a₁, a₂, ..., aₙ]) = softmax([a₁-c, a₂-c, ..., aₙ-c])` với c là hằng số bất kỳ

Vì sao có tính chất này:
- Vì trong công thức softmax: `exp(ai-c)/∑exp(aj-c) = exp(ai)×exp(-c)/[∑exp(aj)×exp(-c)] = exp(ai)/∑exp(aj)`
- Hệ số `exp(-c)` bị triệt tiêu ở cả tử và mẫu.

|![](https://ar5iv.labs.arxiv.org/html/2410.02367/assets/x2.png)|![](https://ar5iv.labs.arxiv.org/html/2410.02367/assets/x3.png)|
|-|-|

- Nếu giữ SWA với 4k window thì không có sự khác biệt về tốc độ giữa attn và linear
- Nếu dùng flash-attn @ 4k thậm chí tăng tốc do tối ưu với phần cứng

|![](https://pbs.twimg.com/media/GtjZnudbMAABsCF?format=jpg&name=medium)|![](https://pbs.twimg.com/media/GtjanVZbUAAwep9?format=jpg&name=medium)|
|-|-|
|![](https://pbs.twimg.com/media/GtjbjYLaMAAwdaF?format=jpg&name=medium)|![](https://pbs.twimg.com/media/GtjckTObEAADPln?format=png&name=medium)|
|![](https://pbs.twimg.com/media/Gtjjm9naoAEyzNI?format=jpg&name=medium)|![](https://pbs.twimg.com/media/GtjlGsFakAAsJ9Y?format=jpg&name=medium)|
|![](https://pbs.twimg.com/media/GtjlGsFakAAsJ9Y?format=jpg&name=medium)|![](https://pbs.twimg.com/media/GtjmMyZbQAAXiHz?format=png&name=medium)|
|![]()|![]()|

**Lượng tử hóa thích ứng**: SageAttention triển khai bốn phiên bản kernel khác nhau với sự đánh đổi tốc độ-độ chính xác khác nhau và "propose a method to `select the fastest attention implementation for each layer while preserving accuracy`". Hệ thống sẽ chọn kernel phù hợp nhất cho từng lớp **dựa trên độ tương đồng cosine**.

