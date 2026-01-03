# An Attention-Free Transformer
https://arxiv.org/pdf/2105.14103.pdf

aft là một biến thể hiệu quả của tfm nhờ loại bỏ nhân ma trận trong self-attn. Trong tầng aft, key và value được kết hợp với một tập các position bias huấn luyện được, kết quả sau đó được nhân với query element-wise. aft giữ được sự tương tác giữa 2 điểm bất kỳ trong context, đó cũng là thế mạnh của self-attn.

Với multi-head attn, với mỗi head $i$  $f_{i}(X) = softmax(\frac{Q_{i} K_{i}^T} {\sqrt(d_k)}) V_i$ với $X \in R^{T \times d}$,
$$Q _ { i } = X W _ { i } ^ { Q } , K _ { i } = X W _ { i } ^ { K } , V _ { i } = X W _ { i } ^ { V }$$
khi đó $$W _ { i } ^ { Q } \in R ^ { d \times d _ { k } } , W _ { i } ^ { K } \in R ^ { d \times d _ { k } } , W _ { i } ^ { V } \in R ^ { d \times d _ { v } }$$, $d_{k}$ và $d_{v}$ là chiều của key và value. MHA ghép nối đầu ra của $h$ attn heads dọc theo chiều của channel, và được kết quả là một vector đặc trưng có chiều $hd_{v}$. Ta có thể giả thiết $d_{k} = d_{v}$ và $h = \frac{d}{d_{k}}$: có nghĩa là $Q,K,V$ có cùng số chiều và chiều của đầu ra bằng chiều của đầu vào.

Với aft, bước đầu tiên cũng chuyển hóa tuyến tính $X$ thành $Q = X W ^ { Q } , K = X W ^ { K } , V = X W ^ { V }$, và sau đó thực hiện: $Y = f(X)$;
$$Y _ { t } = \sigma _ { q } ( Q _ { t } ) \odot \frac { \sum _ { t ^ { \prime } = 1 } ^ { T } \exp ( K _ { t ^ { \prime } } + w _ { t , t ^ { \prime } } ) \odot V _ { t ^ { \prime } } } { \sum _ { t ^ { \prime } = 1 } ^ { T } \exp ( K _ { t ^ { \prime } } + w _ { t , t ^ { \prime } } ) }$$ với:
- $\odot$ là element-wise product.
- $\sigma_{q}$ là biến đổi phi tuyến tính sigmoid áp dụng vào Query.
![](files/aft-00.jpg)

__Giải thích bằng lời__: Với mỗi target position $t$, aft thực hiện một weighted avg of values, kết quả sau đó được kết hợp với query bằng element-wise multiplication. Cụ thể, weighting (trọng số) là một cách kết hợp đơn giản của keys và learned pair-wise position bias. Như vậy: $w_{t,t'}$ là bias của từng cặp vị trí (ám chỉ $t$ và $t'$).
![](files/aft-01.jpg)

## Công thức chung là vậy (aft-full), giờ ta đi sâu vào các biến thể của aft

### aft-local

Trong nhiều ứng dụng locality (tính địa phương) là một inductive bias (thiên kiến quy nạp) đã được chỉ ra bởi CNNs và tfm. Chúng tôi nhận thấy rằng khi train ViT (một tfm tiêu chuẩn) có xu hướng thể hiện extensive local attn patterns.
![](files/aft-02.jpg)
Hình minh họa attn maps trên cho thấy strong local patterns (được thể hiện bởi hình dáng) đặc biệt là ở các tầng thấp. Điều này là động lực cho aft-local, khi mà chúng ta chỉ áp dụng a learned set of relative position bias locally:
$w_{t,t} = w_{t,t}$ with $|(t - t')| < s$ còn lại bằng 0, với s <= T là local window size. Khác với local tfm, aft-local vẫn giữ được các liên kết toàn cục bất kể kích cỡ cửa sổ s. Các thử nghiệm đã chứng minh tính hiệu quả của thiết kế này.
