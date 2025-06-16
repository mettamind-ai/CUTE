sagefwd sử dụng phép zero centering `K - mean(K)` để loại bỏ outliers trong K matrix. "Note that such a transformation does not change the attention score P, because for any query q, we have `softmax(q(K - mean(K))⊤) = softmax(qK⊤ - q · mean(K)) = softmax(qK⊤)`."

Giải thích chi tiết:

- Trong phép nhân QK⊤: `q × (K - mean(K))⊤ = q × K⊤ - q × mean(K)⊤`
- Trong softmax: `q × mean(K)⊤` tạo ra một hằng số được trừ khỏi tất cả các score trong cùng một hàng
- Tính chất softmax: `softmax([a₁, a₂, ..., aₙ]) = softmax([a₁-c, a₂-c, ..., aₙ-c])` với c là hằng số bất kỳ

Vì sao có tính chất này:
- Vì trong công thức softmax: `exp(ai-c)/∑exp(aj-c) = exp(ai)×exp(-c)/[∑exp(aj)×exp(-c)] = exp(ai)/∑exp(aj)`
- Hệ số `exp(-c)` bị triệt tiêu ở cả tử và mẫu.

