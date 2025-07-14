# H-Net: Học chuỗi phân cấp không cần Tokenizer

Tài liệu này tóm tắt những ý tưởng chính từ bài báo H-Net và các chi tiết triển khai thú vị được tìm thấy trong mã nguồn.

- **Bài báo:** [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/html/2507.07955v1)
- **Mã nguồn:** `hnet/*.py`

<table><tr>
<td width="45%"><img src="https://github.com/goombalab/hnet/raw/main/assets/code.gif"></td>
<td width="55%"><img src="https://raw.githubusercontent.com/goombalab/hnet/refs/heads/main/assets/arch.png"></td>
</tr></table>

## Các khái niệm cốt lõi (từ bài báo)

1.  **Học End-to-End trực tiếp từ Byte:** H-Net loại bỏ hoàn toàn bước tiền xử lý token hóa với một bộ từ vựng cố định (như BPE). Thay vào đó, nó học trực tiếp từ các byte thô, cho phép mô hình tự xây dựng các biểu diễn của riêng mình.

2.  **Gộp chuỗi động (Dynamic Chunking - DC):** Đây là cơ chế trung tâm. Mô hình học cách phân đoạn một chuỗi thành các "đoạn" (chunk) có độ dài thay đổi dựa trên sự tương đồng về nội dung. Quá trình này phụ thuộc vào ngữ cảnh, linh hoạt hơn token cố định.

3.  **Xử lý phân cấp:** Kiến trúc có nhiều tầng. Tầng thấp nhất xử lý các byte thô để tạo ra các chunk. Các tầng cao hơn xử lý các biểu diễn của những chunk này, cho phép mô hình học các cấu trúc phức tạp và các phụ thuộc xa.

4.  **Hiệu suất và Độ bền vững:** Bài báo cho thấy H-Net vượt trội hơn các mô hình Transformer dựa trên token có cùng quy mô. Bản chất xử lý ở cấp độ byte giúp nó bền vững trước các lỗi chính tả, từ hiếm và dữ liệu đa ngôn ngữ.

## Phân tích triển khai chính (Kết nối giữa Code và Lý thuyết)

### 1. Cấu trúc phân cấp đệ quy (`hnet/hnet.py`)
Bản chất phân cấp của H-Net được triển khai một cách thanh lịch bằng cách sử dụng đệ quy.
- Class `HNet` chứa một module `main_network`.
- Trong quá trình khởi tạo, nếu tầng hiện tại không phải là tầng cuối cùng (`is_innermost` là False), `self.main_network` sẽ trở thành một thực thể khác của `HNet` với `stage_idx` được tăng lên.
- Vòng đệ quy này dừng lại ở tầng trong cùng, nơi `main_network` là một mô hình `Isotropic` (một chuỗi các khối xử lý tuần tự). Thiết kế này phản ánh hoàn hảo cấu trúc phân cấp lý thuyết.

### 2. Cơ chế Gộp chuỗi động (`hnet/dynamic_chunking.py`)
Đây là phần hiện thực hóa ý tưởng cốt lõi của bài báo.
- **`RoutingModule`**: Dự đoán ranh giới của chunk. Nó tính toán độ tương đồng cosine (cosine similarity) giữa các trạng thái ẩn của hai token liền kề (`t` và `t+1`). Độ tương đồng thấp (khoảng cách lớn) cho thấy một ranh giới tự nhiên, dẫn đến `boundary_prob` cao.
- **`ChunkLayer`**: Một lớp đơn giản nhưng hiệu quả, sử dụng `boundary_mask` từ `RoutingModule` để lọc chuỗi, chỉ chuyển các trạng thái ẩn tại các vị trí ranh giới lên tầng phân cấp tiếp theo.
- **`DeChunkLayer`**: Phần phức tạp nhất, thực hiện việc "trải phẳng" chuỗi đã xử lý. Nó sử dụng một phép quét giống như EMA (Trung bình động hàm mũ) để lan truyền thông tin từ các biểu diễn của chunk trở lại chuỗi có độ dài ban đầu. **Một điểm thú vị là nó tái sử dụng một cách thông minh kernel `mamba_chunk_scan_combined` để thực hiện thao tác này một cách hiệu quả.**

### 3. Huấn luyện một quyết định "Cứng" (`hnet/hnet.py`)
Việc quyết định một ranh giới chunk là một lựa chọn rời rạc, không khả vi. Vấn đề này được giải quyết bằng **Bộ ước tính truyền thẳng (Straight-Through Estimator - STE)**.
- Class `STE` được định nghĩa để hoạt động như một hàm đồng nhất (identity function) trong quá trình lan truyền ngược (`backward(ctx, grad_output): return grad_output`).
- Điều này "đánh lừa" bộ tối ưu hóa bằng cách cho phép gradient đi qua điểm quyết định cứng như thể nó là một hàm liên tục, giúp `RoutingModule` có thể được huấn luyện end-to-end. Hàm `residual_func` đã áp dụng kỹ thuật này.

### 4. Kiến trúc linh hoạt, lai (Hybrid) (`hnet/config_hnet.py`, `hnet/block.py`)
Mô hình không bị trói buộc vào một loại khối xử lý duy nhất. `HNetConfig` cho phép định nghĩa một `arch_layout` để thiết kế các kiến trúc rất linh hoạt.

**Giải thích `arch_layout`:**
- **`M`**: Viết tắt của khối **Mamba**.
- **`T`**: Viết tắt của khối **Attention** (Transformer).
- **Số liền sau (ví dụ `6` trong `M6`)**: Là số lần lặp lại của khối đó. `M6` nghĩa là 6 khối Mamba nối tiếp nhau.
- **Cấu trúc lồng nhau `[..., [...], ...]`**: Thể hiện các tầng phân cấp. Phần tử ở giữa `[...]` là tầng xử lý sâu hơn.

**Ví dụ 1: `["M6", ["M12"], "M6"]`**
Đây là kiến trúc 2 tầng, chỉ dùng Mamba:
1.  **Tầng 0 (Encoder - `M6`)**: Dữ liệu đầu vào đi qua 6 khối Mamba.
2.  **Gộp chuỗi (Chunking)**: Dữ liệu được nén lại thành một chuỗi ngắn hơn.
3.  **Tầng 1 (Xử lý sâu - `M12`)**: Chuỗi ngắn này được xử lý bởi 12 khối Mamba.
4.  **Trải phẳng (De-chunking)**: Dữ liệu được giải nén về độ dài ban đầu.
5.  **Tầng 0 (Decoder - `M6`)**: Dữ liệu cuối cùng đi qua 6 khối Mamba nữa.

**Ví dụ 2: `["T4M2", ["M10T2"], "M2T4"]`**
Đây là một kiến trúc lai (hybrid) phức tạp hơn, kết hợp cả Attention và Mamba:
1.  **Tầng 0 (Encoder - `T4M2`)**: Dữ liệu đầu vào đi qua 4 khối Attention, rồi tiếp tục qua 2 khối Mamba.
2.  **Gộp chuỗi (Chunking)**.
3.  **Tầng 1 (Xử lý sâu - `M10T2`)**: Chuỗi ngắn được xử lý bởi 10 khối Mamba, rồi tiếp tục qua 2 khối Attention.
4.  **Trải phẳng (De-chunking)**.
5.  **Tầng 0 (Decoder - `M2T4`)**: Dữ liệu cuối cùng đi qua 2 khối Mamba, rồi kết thúc bằng 4 khối Attention.

Thiết kế này cho phép kết hợp sức mạnh của cả hai loại khối: Attention giỏi trong việc nắm bắt các mối quan hệ phức tạp, trong khi Mamba lại rất hiệu quả trong việc xử lý các chuỗi dài.

---
## Sơ đồ hành trình: Từ Byte đến Byte

Để hiểu rõ cách H-Net hoạt động, hãy cùng theo dõi toàn bộ hành trình của dữ liệu, từ lúc là byte đầu vào cho đến khi dự đoán ra byte tiếp theo. Ta sẽ dùng kiến trúc `["M6", ["M12"], "M6"]` làm ví dụ.

### Hành trình đi vào (Encoding & Chunking)
1.  **Byte đầu vào** -> `nn.Embedding` -> Tạo ra vector biểu diễn ban đầu `h_0`.
2.  `h_0` đi qua **Encoder** của tầng ngoài (`M6`) -> Tạo ra `h_encoded`. Vector này chứa thông tin ngữ cảnh cục bộ.
3.  `h_encoded` được `RoutingModule` phân tích để tạo ra `boundary_mask` (mặt nạ ranh giới).
4.  `ChunkLayer` dùng `boundary_mask` để nén `h_encoded` thành một chuỗi ngắn hơn `h_chunked`.
5.  `h_chunked` được đưa vào tầng trong cùng (`M12`) để xử lý. Kết quả là `z`, một vector chứa thông tin ngữ cảnh ở mức độ rất cao và trừu tượng.

### Hành trình đi ngược ra (De-chunking & Decoding)
`z` không trực tiếp dự đoán byte. Nó bắt đầu một hành trình đi ngược ra để làm giàu thông tin cho các tầng bên ngoài.

6.  **DeChunkLayer (Dùng EMA)**: Lớp này nhận `z` và `boundary_mask`. Nó dùng cơ chế EMA để "trải" thông tin trừu tượng trong `z` ra lại thành một chuỗi có độ dài đầy đủ, gọi là `h_dechunked`.
7.  **Kết nối phần còn lại (Residual Connection)**: Đây là bước cực kỳ quan trọng. Mô hình kết hợp `h_dechunked` (thông tin trừu tượng, mượt mà) với `h_encoded` (thông tin chi tiết, nguyên bản được giữ lại từ trước). Việc này cho phép mô hình có được cả hai: cái nhìn tổng quan từ tầng sâu VÀ chi tiết cụ thể từ tầng nông.
8.  **Decoder của tầng ngoài (`M6`)**: Vector kết hợp ở trên tiếp tục đi qua các khối xử lý cuối cùng của tầng ngoài (`M6`) để hòa trộn hai nguồn thông tin lại với nhau. Kết quả là `h_final`.
9.  **Chiếu ra Byte (LM Head)**: **Chỉ bây giờ**, `h_final` mới được đưa vào lớp `lm_head`. Lớp này chiếu `h_final` thành một vector `logits` có 256 chiều (tương ứng với 256 giá trị byte).
10. **Dự đoán**: Một hàm `softmax` được áp dụng lên `logits` để tạo ra phân phối xác suất, và byte có xác suất cao nhất được chọn làm dự đoán.

> **Ghi chú quan trọng về việc hòa trộn thông tin:**
> Bước 7 (Kết nối phần còn lại) là nơi sức mạnh của kiến trúc phân cấp tỏa sáng. Nó hòa trộn hai nguồn thông tin thiết yếu:
> - **Thông tin "Từ trên xuống" (`h_dechunked`):** Mang ngữ cảnh trừu tượng, tổng quan từ tầng xử lý sâu. Giống như "cái nhìn chiến lược".
> - **Thông tin "Từ dưới lên" (`h_encoded`):** Mang chi tiết cục bộ, nguyên bản từ tầng nông. Giống như "báo cáo tại hiện trường".
> Việc kết hợp này cho phép mô hình có được cả **cái nhìn tổng quan** và **chi tiết cụ thể** để đưa ra dự đoán cuối cùng chính xác nhất.

---
## Phân tích sâu: Cơ chế Gộp chuỗi động (`hnet/dynamic_chunking.py`)

File `dynamic_chunking.py` là trái tim của H-Net. Nó bao gồm 3 thành phần chính hoạt động như một dây chuyền:

1.  **`RoutingModule`**: **Người Ra Quyết Định** - Quyết định xem vị trí nào nên là ranh giới của một "chunk".
2.  **`ChunkLayer`**: **Người Thực Thi** - Dựa trên quyết định, nó thực sự lọc và tạo ra chuỗi ngắn hơn.
3.  **`DeChunkLayer`**: **Người Tái Tạo** - Sau khi chuỗi ngắn được xử lý, nó sẽ "trải" thông tin trở lại chuỗi dài ban đầu.

### 1. `RoutingModule` - Người Ra Quyết Định
**Mục đích:** Trả lời câu hỏi: "Tại mỗi vị trí trong chuỗi, xác suất để vị trí này là điểm kết thúc của một chunk là bao nhiêu?"

- **Ý tưởng cốt lõi:** Nếu hai token liền kề nhau có ý nghĩa rất khác nhau, thì khả năng cao giữa chúng là một ranh giới. Ngược lại, nếu chúng rất giống nhau (ví dụ: "c" và "h" trong "ch"), chúng nên thuộc cùng một chunk.
- **Triển khai trong code:**
    1.  Nó sử dụng hai lớp `Linear` (`q_proj_layer`, `k_proj_layer`) để chiếu (project) các vector `hidden_state` của mỗi token.
    2.  Sau đó, nó tính **độ tương đồng cosine (cosine similarity)** giữa mỗi token và token ngay sau nó.
    3.  **Bước quan trọng nhất:** Nó chuyển đổi độ tương đồng (từ -1 đến 1) thành xác suất ranh giới (từ 0 đến 1) bằng công thức `(1 - cos_sim) / 2`.
        - Nếu `cos_sim` = 1 (rất giống nhau) -> xác suất = 0 (không phải ranh giới).
        - Nếu `cos_sim` = -1 (rất khác nhau) -> xác suất = 1 (chắc chắn là ranh giới).
- **Điểm thông minh:**
    - Nó **ép xác suất của token đầu tiên luôn là 1.0**. Điều này đảm bảo rằng mọi chuỗi luôn bắt đầu bằng một chunk.
    - Kết quả cuối cùng là `boundary_mask` - một chuỗi boolean (`True`/`False`) cho biết vị trí nào được chọn làm ranh giới.

### 2. `ChunkLayer` - Người Thực Thi
**Mục đích:** Nhận vào `boundary_mask` và chuỗi đầy đủ, sau đó tạo ra một chuỗi mới chỉ chứa các token được đánh dấu là ranh giới.

- Lớp này cực kỳ đơn giản nhưng hiệu quả. Nó sử dụng **lập chỉ mục bằng mảng boolean (boolean array indexing)**.
- Dòng code cốt lõi là `hidden_states[boundary_mask]` sẽ tạo ra một tensor mới, chỉ chứa các token từ `hidden_states` mà tại đó `boundary_mask` có giá trị là `True`.
- Kết quả là một chuỗi ngắn hơn nhiều, sẵn sàng để được đưa vào tầng xử lý sâu hơn của H-Net.

### 3. `DeChunkLayer` - Người Tái Tạo
**Mục đích:** Sau khi tầng sâu hơn đã xử lý chuỗi ngắn, lớp này cần phải "phân phối" thông tin đó trở lại chuỗi có độ dài ban đầu.

- **Ý tưởng cốt lõi:** Mỗi token trong chuỗi gốc sẽ nhận thông tin từ chunk gần nhất mà nó thuộc về, được thực hiện bằng một phép toán tương tự như **Trung bình động hàm mũ (EMA)**.
- **Triển khai thông minh:**
    1.  Thay vì tự viết một kernel EMA, các tác giả đã **tái sử dụng kernel `mamba_chunk_scan_combined`** từ Mamba2, một kernel được tối ưu hóa ở mức độ rất cao.
    2.  Họ thiết lập các tham số đầu vào cho kernel Mamba (`A`, `dt`, `b`, `c`) theo một cách đặc biệt để nó thực hiện chính xác phép toán EMA, mô phỏng hành vi phân rã theo hàm mũ.
    3.  Sau khi kernel Mamba thực hiện phép quét, nó sử dụng `torch.gather` để "cắm" các giá trị đã được làm mịn này trở lại đúng vị trí của chúng trong chuỗi có độ dài ban đầu.

---
## Phân tích sâu: `DeChunkLayer` và vai trò của EMA

`DeChunkLayer` là một trong những thành phần thông minh và quan trọng nhất của H-Net. Nó giải quyết một bài toán khó: Làm thế nào để "trải" hoặc "lan tỏa" thông tin từ một vài vector biểu diễn chunk (đã được xử lý ở tầng sâu) ra tất cả các byte con mà nó đại diện một cách mượt mà?

#### Vấn đề cần giải quyết

- **Đầu vào:** Một chuỗi ngắn gồm các vector `z` của chunk (ví dụ: 3 vector cho 3 chunk `[Học]`, `[chuỗi]`, `[phân cấp]`).
- **Đầu ra cần tạo:** Một chuỗi dài có cùng độ dài với chuỗi byte ban đầu (ví dụ: 18 vector cho 18 ký tự), trong đó mỗi byte đều có một vector biểu diễn mới, đã được làm giàu thông tin.

Một cách tiếp cận đơn giản là sao chép vector của chunk cho tất cả các byte con. Tuy nhiên, cách này tạo ra các "vách đá" thông tin đột ngột ở ranh giới chunk và làm mất thông tin về vị trí tương đối của byte bên trong chunk.

#### Giải pháp thanh lịch với EMA (Trung bình động hàm mũ)

`DeChunkLayer` sử dụng một cơ chế tương tự EMA để giải quyết vấn đề này một cách hoàn hảo. Hãy tưởng tượng EMA như một cách "làm mịn" hoặc "lan tỏa" giá trị theo thời gian.

**Cơ chế hoạt động:**

1.  **Khi gặp một ranh giới chunk (ví dụ, byte 'c' trong "chuỗi"):**
    - Xác suất ranh giới `boundary_prob` (đóng vai trò là cổng `alpha` trong EMA) sẽ cao.
    - Công thức cập nhật sẽ ưu tiên mạnh mẽ cho giá trị mới (vector `z` của chunk `[chuỗi]`). Vector của byte 'c' sẽ gần như là vector của chunk này.

2.  **Khi ở bên trong một chunk (ví dụ, byte 'h' sau 'c'):**
    - `boundary_prob` (`alpha`) sẽ rất thấp.
    - Công thức sẽ ưu tiên mạnh mẽ cho giá trị cũ (vector của byte 'c' ngay trước đó). Vector của 'h' sẽ là một phiên bản "sao chép có giảm nhẹ" từ vector của 'c'. Quá trình này tiếp tục cho các byte tiếp theo, tạo ra một dòng chảy thông tin.

**Kết quả của việc dùng EMA:**
- **Chuyển tiếp mượt mà:** Thay vì một "vách đá", thông tin từ một chunk sẽ "phai" hoặc "lan tỏa" dần qua các byte bên trong nó.
- **Tạo nhận thức về vị trí:** Do hiệu ứng "phai" dần này, vector của byte ở đầu chunk sẽ hơi khác một chút so với vector của byte ở cuối chunk. Điều này giúp mô hình giữ lại được thông tin vị trí tương đối bên trong chunk.
