## `LiWin` `Li`near Attention + `Win`dow Attention
- https://github.com/m-a-n-i-f-e-s-t/power-attention
- Gated DeltaNet + SWA + Mamba2 https://www.alphaxiv.org/abs/2412.06464
- DeltaFormer https://ar5iv.labs.arxiv.org/html/2505.19488v1
  - https://youtu.be/vXjk1LF-qqg
  - https://asap-seminar.github.io/assets/slides/deltaformer_slide.pdf

__Kết hợp best SWA (local) với Linear Attention (global)__
![](/.save/liwin-00-crunch.png)
- Hymba: 
    - https://www.youtube.com/watch?v=a31C8ahIDhk
    - https://asap-seminar.github.io/assets/slides/ASAP%20Talk_%20Hymba-Small%20Hybrid%20Language%20Model.pdf
    
### [Taipan: Mamba + Selective Attention Layers (SALs)](https://arxiv.org/html/2410.18572v1)
![](https://arxiv.org/html/2410.18572v1/x2.png)
![](https://arxiv.org/html/2410.18572v1/x3.png)

### [Based = Li + Win mỏng (Linear Attn + SWA mỏng)](https://www.alphaxiv.org/abs/2402.18668v2)
![](https://arxiv.org/html/2402.18668v2/x1.png)


## Các biến thể của Attn
- `Glo` thiên về tóm tắt / toàn cục, kiểu như `Li`
- `Sel` chọn những khối tokens quan trọng để attn
- `Win` cục bộ theo cửa sổ, điển hình là `SWA` vô cùng đơn giản và hiệu quả nhất
- `Loc` local_attention mở rộng và look backward xa hơn `SWA`

## Win should use Trainable Sparse Attention
- `MoSA` = `Sel`; có thể kết hợp với local_attention nên hoàn toàn có thể thay thế `SWA`
- `NSA` = `Glo` + `Sel` + `Win`; có thể thay thế mọi loại block (cả Li và Win)
    - Vì Li đã có global, có thể chỉ dùng `Win` và `Sel` của NSA?
- `Taipan` = `Sel` + `Win`; => Có thể viết lại `Taipan Sel` (SALs) để dùng với `Win`
- `Based` = Li`Glo` + `Win` mỏng; `LiGlo` = Taylor Exponential Linear Attention nhanh nhẹ

# Learn at Test Time
- Titans: Learning to Memorize at Test Time https://arxiv.org/abs/2501.00663
- RNNs with Expressive Hidden States https://arxiv.org/abs/2407.04620
- Gated Delta Networks: Improving Mamba2 with Delta Rule https://arxiv.org/abs/2412.06464

---

# FoX Forgetting Transformer
![](https://pbs.twimg.com/media/Gwci8y9bEAAjENO?format=jpg&name=large)
Forget gate có norm term của RNN => biến đổi sang dạng song song của Linear Attn => đổi kernel function sang exp thì được dạng forget gate của softmax attn

![](https://pbs.twimg.com/media/GwcokGUasAALyW2?format=jpg&name=large)

![](https://pbs.twimg.com/media/GwcqQiwbEAABNtW?format=png&name=large)

![](https://pbs.twimg.com/media/Gwcv-cnbAAAdvdX?format=jpg&name=large)
ShiftLinear là một lớp (layer) trong kiến trúc FoX (Pro) dùng để thực hiện một phép tính gọi là "KV-shift," hay dịch chuyển token phụ thuộc vào dữ liệu. Lớp này tính toán các giá trị "key" và "value" mới tại một thời điểm bằng cách lấy trung bình có trọng số giữa giá trị ở thời điểm hiện tại và giá trị ở thời điểm ngay trước đó. Mục đích là để kết hợp thông tin từ token liền kề, một kỹ thuật được lấy cảm hứng từ các mô hình chuỗi hồi quy (recurrent sequence models). Tác dụng chính của ShiftLinear (thông qua cơ chế KV-shift) là để tăng cường khả năng của mô hình trong việc nắm bắt các mối quan hệ cục bộ và tuần tự giữa các token liền kề nhau. => có thể thay bằng causal conv1d?

![](https://pbs.twimg.com/media/GwcqxH9bYAAa-kf?format=jpg&name=large)

![](https://pbs.twimg.com/media/GwcruN6bEAASelW?format=jpg&name=large)

FoX vượt trội hơn softmax attention truyền thống nhờ việc tích hợp cơ chế "forget gate" - một khả năng mà attention chuẩn không có.

**Lý do chính:**

1. **Cơ chế quên thông tin linh hoạt**: FoX có thể "quên" thông tin quá khứ một cách có chọn lọc và phụ thuộc vào dữ liệu. Bài báo nêu: "Transformer thiếu một cơ chế tường minh để quên thông tin quá khứ theo cách phụ thuộc vào dữ liệu." FoX khắc phục điểm yếu này bằng cách down-weight các unnormalized attention scores thông qua forget gate.

2. **Kết hợp ưu điểm của cả hai thế giới**: FoX giữ được khả năng xử lý long-context của Transformer trong khi tích hợp cơ chế forget gate quan trọng từ các mô hình chuỗi hồi quy. Như bài báo chỉ ra: "cơ chế này... đã được chứng minh là `quan trọng` trong thành công của chúng trong các `tác vụ short-context`."

3. **Kết quả thực nghiệm ưu việt**: Theo Figure 2, FoX outperform Transformer trên long-context language modeling, length extrapolation, và short-context downstream tasks. Cụ thể, FoX duy trì được per-token loss giảm dần trong toàn bộ training context length, cho thấy việc sử dụng hiệu quả toàn bộ ngữ cảnh.

4. **Không cần positional embeddings**: Khác với Transformer chuẩn cần RoPE, FoX có thể hoạt động hiệu quả mà không cần positional embeddings, như được nêu: "nó không cần bất kỳ positional embeddings nào."

5. **Giữ được khả năng truy xuất thông tin của attn**: Trong needle-in-the-haystack test (Figure 4), FoX đạt "near-perfect accuracy" trong training context length, trong khi các mô hình chuỗi hồi quy khác thất bại.

[ **Forget gate**: Cơ chế "cổng quên" giúp mô hình quyết định thông tin nào từ quá khứ nên được giữ lại hay loại bỏ; **Down-weight**: Giảm trọng số hoặc tầm quan trọng của một giá trị trong tính toán; **Length extrapolation**: Khả năng xử lý các chuỗi dài hơn so với độ dài được huấn luyện; **Needle-in-the-haystack test**: Bài kiểm tra đánh giá khả năng tìm và truy xuất thông tin cụ thể trong một đoạn văn bản dài. ]

# Path-FoX
- https://ar5iv.labs.arxiv.org/html/2505.16381v1
- https://www.alphaxiv.org/abs/2505.16381

PaTH đòi hỏi một thuật toán **hoàn toàn mới và phức tạp hơn đáng kể**, không phải là một sửa đổi đơn giản với flash attention như FoX.Đây là lý do tại sao có sự khác biệt lớn về độ khó:

**1. Bản chất của sự thay đổi:**

*   **FoX (Đơn giản - Phép cộng):** Cơ chế "forget gate" của FoX là một phép **cộng** một giá trị thiên vị (bias) vào điểm attention (logit) *sau khi* đã tính `QK^T`. Trong vòng lặp của FlashAttention, việc này cực kỳ đơn giản: khi bạn tính điểm attention cho một cặp `(query_i, key_j)`, bạn chỉ cần tải thêm giá trị `D_ij` tương ứng và cộng vào. Bài báo FoX nhấn mạnh: "nó có thể được triển khai với một sửa đổi đơn giản cho thuật toán FlashAttention."
*   **PaTH (Phức tạp - Phép nhân tích lũy):** Cơ chế của PaTH là một phép **nhân** một ma trận biến đổi (`H_ij`) vào vector `query` và `key` *trước khi* chúng được nhân với nhau. Điều phức tạp là ma trận `H_ij` này là một **sản phẩm tích lũy (cumulative product)** của tất cả các ma trận biến đổi phụ thuộc vào dữ liệu từ vị trí `j` đến `i`. Điều này phá vỡ hoàn toàn tính độc lập của các khối trong FlashAttention. Bạn không thể tính toán cho khối `(i, j)` mà không biết tất cả các phép biến đổi đã xảy ra trên đường đi.

---

Nhìn lại thì thấy PaTH và FoX đều là 1 sự cải tiến của positional embedding. FoX thì là data-dependent Alibi còn PaTH thì có thể so sánh là data dependent RoPE. Tuy chỉ là position embedding nhưng nó biến đổi trực tiếp giá trị Q và K và sự biến đổi này có thể diễn giải theo các chiều hướng khác nữa ví dụ như FoX thì cơ chế gated forgetting. Nếu vậy thì PaTH có thể diễn giải theo chiều hướng / cơ chế nào?


## **FoX = Data-dependent ALiBi**
**ALiBi (static):** `A_ij ∝ exp(q_i^T k_j - m_h(i-j))`
- Bias cố định: `-m_h(i-j)` 
- Chỉ phụ thuộc relative position
**FoX (dynamic):** `A_ij ∝ exp(q_i^T k_j + Σ_{s=j+1}^i log f_s)`
- Bias data-dependent: `Σ log f_s` với `f_s = sigmoid(w_f^T x_s)`

## **PaTH = Data-dependent RoPE**
**RoPE (static):** `A_ij ∝ exp(q_i^T R_{i-j} k_j)`
- Rotation matrix cố định: `R_{i-j}`
- Chỉ phụ thuộc relative position

**PaTH (dynamic):** `A_ij ∝ exp(q_i^T (∏_{s=j+1}^i H_s) k_j)`
- Householder transformations data-dependent: `H_s = I - β_s w_s w_s^T`
- Paper PaTH nói: *"RoPE is thus a special case of the above with a static transition matrix H_s = R"*

## **Key insight:**
**Cả hai đều "động hóa" static positional encodings:**
- **FoX**: Static bias → Dynamic bias (additive)  
- **PaTH**: Static rotation → Dynamic transformation (multiplicative)

**Trade-off tương tự:**
- **Tăng expressivity** (có thể solve NC¹-complete problems)
- **Giảm efficiency** (thêm computational cost)
- **Cải thiện state tracking** và sequential reasoning

[static vs dynamic: tĩnh vs động; additive vs multiplicative: cộng vs nhân; expressivity: khả năng biểu đạt]


# PaTH có thể diễn giải theo nhiều cơ chế thú vị khác ngoài positional embedding:

## 1. Adaptive Feature Space Transformation
**Concept**: PaTH **động biến đổi không gian đặc trưng** theo sequence path `q_i^T (∏_{s=j+1}^i H_s) k_j`
- Mỗi `H_s` = "local transformation" phụ thuộc input `x_s`
- Cumulative product = "global transformation pathway" từ j→i
- **Diễn giải**: Model học cách **reshape feature space** để tăng tính phân biệt giữa các positions

## 2. Sequential Memory with Adaptive Routing
Paper nói: *"PaTH is closely related to such expressive linear RNNs"*
```
RNN: o_t = (∑ v_j ∏ H_s) q_t
PaTH: o_t = (1/Z) ∑ v_j exp(∏ H_s) q_t  
```
- **Diễn giải**: PaTH = "softmax version of memory-augmented RNN"
- Mỗi path j→i có **routing weight** = `exp(cumulative transformation)`

## 3. Learnable Computation Pathway
**Key insight**: PaTH học **cách compute** thay vì chỉ encode position
- `H_s = I - β_s w_s w_s^T`: **rank-1 update** tại mỗi step
- Cumulative product: **sequential computation graph** 
- **Diễn giải**: Model học "algorithm" để process sequence theo data-dependent path

## 4. Dynamic Context Mixing
**Fundamental difference**:
- **FoX**: "Should I forget this information?" (gating)
- **PaTH**: "How should I transform this information?" (routing)
**PaTH mechanism**:
Attention từ j→i đi qua "transformation pipeline": `x_j → H_{j+1} → H_{j+2} → ... → H_i → reaches q_i`

## 5. Hierarchical Sequential Processing
Paper chứng minh: *"PaTH can extend transformers beyond TC⁰ complexity class"*
- **Diễn giải**: PaTH tạo ra **hierarchical computation** capabilities
- Mỗi layer có thể học different "transformation strategies"
- Enable **compositional reasoning** through sequence


---

# Khả năng kết hợp giữa PaTH / FoX / và DeltaFormer

Path-FoX đã kết hợp rồi và cho kết quả tốt. PaTH có vẻ khó tích hợp vào Flash Attention nên có lẽ thử Fox-DeltaFormer trước !!!
Và vì FoX thay đổi `QK` còn DeltaFormer thay đổi `V` nên **Về mặt lý thuyết, một mô hình kết hợp FoX và DeltaFormer sẽ mạnh hơn và toàn diện hơn PaTH.** Để hiểu tại sao, chúng ta hãy xem xét "Ba Trụ Cột của Bộ Nhớ" trong Transformer:

1.  **LƯU TRỮ (Value):** *Cái gì* đáng được lưu vào bộ nhớ?
2.  **TRUY VẤN (Key-Query):** *Làm thế nào* để tìm và so sánh các ký ức?
3.  **CHÚ Ý (Logit):** *Mức độ quan trọng* của mỗi ký ức được truy vấn là bao nhiêu?

Bây giờ, hãy xem các mô hình này tác động vào đâu:
-   **PaTH:** Tập trung vào Trụ cột 2 (Truy vấn). Nó "động hóa" cách `key` được biến đổi (`∏ H_s`), thay đổi cách so sánh `key-query`.
-   **DeltaFormer:** Tập trung vào Trụ cột 1 (Lưu trữ). Nó "lọc" `value` (`u_t = v_t - ...`), thay đổi *cái gì* được lưu.
-   **FoX:** Tập trung vào Trụ cột 3 (Chú ý). Nó "điều tiết" attention logit (`+ log F`), thay đổi *mức độ quan trọng* của kết quả so sánh.

**Sự kết hợp FoX + DeltaFormer tạo ra một hệ thống bộ nhớ gần như hoàn chỉnh, tác động lên cả hai khía cạnh LƯU TRỮ và CHÚ Ý.**

---

### So Sánh Chi Tiết

| Khía cạnh | PaTH | FoX + DeltaFormer (Lý thuyết) |
| :--- | :--- | :--- |
| **Cơ chế cốt lõi** | Biến đổi `key` một cách linh hoạt theo đường đi. | 1. Lọc và cập nhật `value` để loại bỏ thông tin thừa. <br> 2. Điều tiết attention score dựa trên nội dung. |
| **Tác động lên** | `Key` (Multiplicative) | `Value` (Subtractive) **VÀ** `Logit` (Additive) |
| **Phép ẩn dụ** | "Con đường biến hình": Mỗi ký ức thay đổi hình dạng khi đi qua một con đường ngữ cảnh. | **"Lọc nhiễu rồi khuếch đại"**: <br> 1. Dọn dẹp, làm sạch từng ký ức (`u_t`). <br> 2. Tập trung sự chú ý vào các ký ức đã được làm sạch. |
| **Giải quyết vấn đề** | Cách **nhìn nhận/so sánh** ký ức (expressivity của RoPE). | Cách **lưu trữ** ký ức (tránh dư thừa) **VÀ** cách **phân bổ sự chú ý** (gated forgetting). |

### Cơ Chế Kết Hợp Hoạt Động Như Thế Nào?

Một mô hình `DeltaFoX-Former` sẽ hoạt động theo 2 bước:

1.  **Cập nhật Value (DeltaFormer):** Đầu tiên, nó tính toán `value` vector mới `u_t` đã được "làm sạch" bằng cách loại bỏ thông tin cũ, tương tự như DeltaFormer:
    `u_t = v_t - Σ exp(k_i^T k_t) * u_i`

2.  **Tính Attention có Gating (FoX):** Sau đó, nó sử dụng `u_t` này trong một phép attention đã được điều tiết bởi FoX:
    `logit_i = q_t^T k_i + Σ log(f_s)`
    `o_t = Σ softmax(logit_i) * u_i`

### Sức Mạnh Cộng Hưởng (Synergy)

Sự kết hợp này tạo ra một hiệu ứng cộng hưởng mạnh mẽ:
-   **DeltaFormer** đảm bảo rằng các `value` (`u_t`) mà attention sẽ lấy ra là những thông tin "tinh khiết", không bị nhiễu bởi các ký ức cũ.
-   **FoX** sau đó có thể phân bổ sự chú ý một cách chính xác và hiệu quả hơn trên những `value` đã được lọc nhiễu này.

So với PaTH, dù PaTH có cơ chế biến đổi `key` rất mạnh, nhưng `value` mà nó truy xuất vẫn là `v_t` "thô", chưa qua bước lọc của DeltaFormer. Do đó, hệ thống `FoX + DeltaFormer` có một cơ chế quản lý và truy xuất thông tin toàn diện và mạnh mẽ hơn.
