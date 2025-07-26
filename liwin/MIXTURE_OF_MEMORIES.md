# MoM: Linear Sequence Modeling with Mixture-of-Memories
- https://ar5iv.labs.arxiv.org/html/2502.13685
- playground/6882feed6a6471d23d20471a

## Chú Ý Tuyến Tính (since 2020)

Để giảm độ phức tạp thời gian của cơ chế attention trong Transformer, nhiều kỹ thuật tối ưu hóa đã được đề xuất. Linear Transformers của Katharopoulos và cộng sự (2020) thay thế cơ chế softmax attention bằng tích vô hướng của các ánh xạ đặc trưng φ(·):

$$o_t = \frac{\sum_{i=1}^{n} \phi(q_i)\phi(k_i)^T v_i}{\sum_{i=1}^{n} \phi(q_i)\phi(k_i)^T},$$

trong đó $q_t, k_t, v_t \in \mathbb{R}^d$. Sự hiện diện của mẫu số có thể dẫn đến sự bất ổn định về số học theo Qin và cộng sự (2024b) và ánh xạ đặc trưng có thể sử dụng hàm đồng nhất, nên có thể bỏ qua để đơn giản hoá. Từ góc độ bộ nhớ, công thức này cũng có thể được viết dưới dạng hồi quy:

$$M_t = M_{t-1} + k_t^T v_t, \quad o_t = q_t M_t.$$

Điều này cho thấy linear attention có thể hoạt động như một lớp hồi quy tuyến tính với trạng thái ẩn có giá trị ma trận M mà chúng tôi gọi là trạng thái bộ nhớ và đầu ra được tạo ra bằng cách truy vấn trạng thái bộ nhớ M. Điều này đại diện cho sự nén tối ưu của thông tin chuỗi, cô đọng toàn bộ chuỗi thành một trạng thái bộ nhớ duy nhất.

Dựa trên các khái niệm nền tảng về linear attention và góc nhìn bộ nhớ, một số tiến bộ gần đây đã tập trung vào việc tối ưu hóa cấu trúc bộ nhớ, bao gồm cập nhật có cổng của Yang và cộng sự (2023); Qin và cộng sự (2024c, d) và mở rộng dung lượng bộ nhớ của Peng và cộng sự (2024); Qin và cộng sự (2024d).

---

Các mô hình chuỗi tuyến tính nén toàn bộ dữ liệu chuỗi thành một trạng thái bộ nhớ có kích thước cố định. Mặc dù có nhiều nỗ lực để giảm thiểu mất mát thông tin—như giới thiệu cơ chế cổng và sử dụng điều khiển chính xác hơn đối với các sửa đổi bộ nhớ — sự suy giảm trong quá trình nén này là không thể tránh khỏi. Việc mở rộng dung lượng bộ nhớ đã được chứng minh có thể giảm thiểu vấn đề này ở một mức độ nào đó, với các nghiên cứu chỉ ra rằng `tăng dung lượng bộ nhớ có thể nâng cao hiệu suất` mô hình.

Tuy nhiên, các phương pháp trước đây chỉ đơn giản tăng kích thước của trạng thái RNN, về cơ bản là mở rộng một trạng thái bộ nhớ duy nhất, gặp khó khăn trong việc nắm bắt toàn bộ phổ thông tin trong một chuỗi hoàn chỉnh. Chúng tôi đề xuất rằng khó khăn này phát sinh vì **thông tin chuỗi thường có nhiều khía cạnh**, và một bộ nhớ duy nhất được mở rộng có thể không có khả năng nắm bắt đồng thời nhiều khía cạnh của dữ liệu. Các đầu vào giới thiệu thông tin mới hoặc trực giao có thể gây nhiễu với nội dung bộ nhớ hiện có khi sử dụng bộ nhớ chung. Thay vì loại bỏ các đầu vào này thông qua cơ chế cổng hoặc ghi đè trạng thái bộ nhớ hiện có, có thể hiệu quả hơn khi xem xét các chiến lược thay thế cho phép bảo tồn thông tin đa dạng mà không có sự can thiệp.

## 3.2 MoM: Hỗn hợp các Bộ nhớ (Mixture-of-Memories)

Để giải quyết thách thức được nêu ở trên, chúng tôi đề xuất một phương pháp mới được lấy cảm hứng từ các cơ chế sinh học để mã hóa bộ nhớ đa mục như dao động theta-gamma Lisman và Jensen (2013), và các khái niệm từ Mixture-of-Experts (MoE) Shazeer và cộng sự (2017), trong đó các chuyên gia khác nhau xử lý các token cụ thể. Trong phương pháp này, chúng tôi tận dụng nhiều trạng thái bộ nhớ, mỗi trạng thái được cập nhật có chọn lọc bởi các đầu vào khác nhau. Điều này làm tăng dung lượng bộ nhớ và cho phép mô hình giữ lại các phần thông tin đa dạng bằng cách lưu trữ các loại đầu vào khác nhau trong các trạng thái bộ nhớ riêng biệt.

Trong framework của chúng tôi, các trạng thái bộ nhớ hoạt động tương tự như các chuyên gia trong MoE. Tuy nhiên, thay vì dựa vào các mạng hoàn toàn riêng biệt, các mô-đun này là các trạng thái RNN riêng lẻ được nhúng trong một cơ chế hồi quy tuyến tính. Thiết kế này cho phép cô lập các cập nhật bộ nhớ trong khi đồng thời quản lý các loại thông tin khác nhau. Điều quan trọng cần lưu ý là MoM khác biệt cơ bản với MoE truyền thống, như chúng tôi sẽ thảo luận trong Phụ lục A. Hình 1 cung cấp cái nhìn tổng quan về kiến trúc MoM. Dưới đây, chúng tôi giới thiệu cấu trúc của lớp MoM và giải thích cách kiến trúc lấy cảm hứng từ sinh học này được triển khai trong bối cảnh mô hình hóa chuỗi tuyến tính.

![](https://ar5iv.labs.arxiv.org/html/2502.13685/assets/x1.png)

**Hình 1: MoM.framework** Mỗi token đầu vào kích hoạt và cập nhật có chọn lọc `top-k` trạng thái bộ nhớ, giữ nguyên các trạng thái bộ nhớ không được kích hoạt để tránh sự can thiệp từ đầu vào hiện tại. Ngoài ra, chúng tôi giới thiệu một bộ nhớ chung được kích hoạt liên tục.

### 3.2.1 Mạng Định Tuyến (Router Network)

Chúng tôi sử dụng một bộ định tuyến để phân bổ các đầu vào cho các trạng thái bộ nhớ khác nhau. Sử dụng khái niệm top-k, mỗi token được định tuyến đến top-k bộ nhớ dựa trên điểm số quan trọng của nó. Cụ thể, chúng tôi sử dụng một lớp tuyến tính đơn giản để tạo ra các điểm số này cho mỗi token đầu vào. Sau khi áp dụng hàm softmax, chúng tôi chọn top-k điểm số và chuẩn hóa chúng.

$$\text{scores}_t = \text{TopK}(\text{softmax}(x_t W_g)) \in \mathbb{R}^k,$$

$$g_t = \frac{\text{scores}_t}{\sum \text{scores}_t} \in \mathbb{R}^k,$$

trong đó $x_t \in \mathbb{R}^d$, $k$ là số top-k, $W_g \in \mathbb{R}^{d \times M}$ là trọng số có thể học, $g_t$ là điểm số quan trọng được chuẩn hóa của đầu vào $x_t$.

### 3.2.2 Mô-đun Bộ Nhớ Hồi Quy Tuyến Tính

Sau mạng định tuyến, đầu vào $x_t$ được hướng đến top-k mô-đun hồi quy tuyến tính, có nghĩa là top-k bộ nhớ được kích hoạt trong khi các bộ nhớ khác vẫn không hoạt động. Đối với mỗi mô-đun bộ nhớ được kích hoạt, được chỉ mục bởi $m$, chúng tôi thực hiện các thao tác sau:

1. **Phép Chiếu Khóa và Giá Trị**: Chúng tôi chiếu đầu vào $x_t$ thành $k_t^m$ và $v_t^m$ sử dụng $W_k^m$ và $W_v^m$:

   $$k_t^m = x_t W_k^m, \quad v_t^m = x_t W_v^m \in \mathbb{R}^d,$$

   trong đó $W_k^m$, $W_v^m$ là các trọng số chiếu có thể học cho $k$, $v$ của mô-đun bộ nhớ thứ $m$.

2. **Cập Nhật Bộ Nhớ**: Chúng tôi cập nhật trạng thái bộ nhớ được kích hoạt sử dụng $k_t^m$, $v_t^m$:

   $$M_t^m = M_{t-1}^m + (k_t^m)^T v_t^m \in \mathbb{R}^{d \times d}.$$

Phương trình trên đại diện cho dạng đơn giản nhất của cập nhật bộ nhớ để rõ ràng. Phương pháp của chúng tôi linh hoạt và không phụ thuộc vào một cơ chế cập nhật bộ nhớ cụ thể. Để nâng cao hiệu suất, chúng tôi có thể kết hợp các cơ chế như cổng quên Sun và cộng sự (2023):

$$M_t^m = \gamma M_{t-1}^m + (k_t^m)^T v_t^m \in \mathbb{R}^{d \times d},$$

trong đó $\gamma$ là cổng quên hằng số.

Tổng quát hơn, phương pháp của chúng tôi có thể được điều chỉnh để kết hợp các phương pháp cập nhật bộ nhớ khác nhau được đề xuất trong các công trình trước đây. Mô tả chi tiết về các phương pháp này được cung cấp trong Bảng 1.

| Phương pháp | Quy tắc Cập nhật Bộ nhớ |
|-------------|-------------------------|
| LA | $M_t = M_{t-1} + k_t^T v_t$ |
| Lightning | $M_t = \gamma M_{t-1} + k_t^T v_t$ |
| RetNet | $M_t = \gamma M_{t-1} + k_t^T v_t$ |
| HGRN2 | $M_t = (a_t^T 1)M_{t-1} + (1 - a_t)^T v_t$ |
| GLA | $M_t = (a_t^T 1)M_{t-1} + k_t^T v_t$ |
| Mamba2 | $M_t = \alpha_t M_{t-1} + \beta_t k_t^T v_t$ |
| DeltaNet | $M_t = (I - k_t^T k_t)M_{t-1} + \beta_t k_t^T v_t$ |
| G-DeltaNet | $M_t = \alpha_t(I - k_t^T k_t)M_{t-1} + \beta_t k_t^T v_t$ |
| TTT | $M_t = M_{t-1} + \beta_t \nabla l(M_{t-1}; k_t, v_t)$ |
| Titan | $M_t = \alpha_t M_{t-1} + \beta_t \nabla l(M_{t-1}; k_t, v_t)$ |

**Bảng 1: Quy tắc Cập nhật Bộ nhớ.** Chúng tôi chứng minh rằng một số mô hình chuỗi tuyến tính hiện tại có thể được xem như các mô hình hồi quy về mặt cập nhật bộ nhớ, trong đó $\alpha_t, \beta_t \in (0,1)$ là hệ số tỷ lệ phụ thuộc dữ liệu, $a_t$ là vector phụ thuộc dữ liệu, và $\gamma$ là hằng số độc lập với dữ liệu.

**Trộn Bộ nhớ (Memory Mixing):** Sau khi cập nhật các trạng thái bộ nhớ được kích hoạt, chúng tôi thực hiện tổng có trọng số của các trạng thái bộ nhớ này sử dụng điểm số quan trọng thu được từ Phương trình (6).

$$\widetilde{M}_t = \sum_m g_t^{(m)} M_t^m \in \mathbb{R}^{d \times d},$$

trong đó $M_m$ là một bộ nhớ được kích hoạt và $g_t^{(m)}$ là điểm số quan trọng của $M_m$.

Sau đó chúng tôi thu được đầu ra của MoM bằng cách áp dụng vector truy vấn $q_t$ vào bộ nhớ đã trộn $\widetilde{M}_t$:

$$o_t = q_t \widetilde{M}_t \in \mathbb{R}^d,$$

Cuối cùng, đầu ra của lớp MoM được tính toán bằng cách áp dụng hàm kích hoạt, chuẩn hóa và phép biến đổi tuyến tính:

$$o_t = \text{RMSNorm}(\text{Swish}(o_t))W_o \in \mathbb{R}^d,$$

Trong suốt quá trình hồi quy, chỉ một tập con các trạng thái bộ nhớ được kích hoạt và cập nhật tại mỗi bước thời gian, trong khi các trạng thái bộ nhớ không được định tuyến vẫn không hoạt động và không thay đổi. Khi đầu vào đi qua lớp chiếu khóa-giá trị, nó tạo ra nhiều bộ khóa và giá trị được đưa vào các mô-đun bộ nhớ khác nhau. Thiết kế này cho phép mô hình duy trì nhiều trạng thái bộ nhớ, mỗi trạng thái bảo tồn các phần thông tin riêng biệt. Bằng cách tổng hợp các bộ nhớ được kích hoạt thành một bộ nhớ trộn toàn diện thông qua tổng có trọng số, truy vấn có thể truy xuất thông tin hiệu quả từ bộ nhớ trộn này, tạo ra "đầu ra attention" được theo sau bởi các lớp khác.

---

## Đánh giá Độ Hiệu quả của MoM dựa trên 2 Bảng Kết quả

### **1. Hiệu quả Vượt trội trên Tác vụ Đòi hỏi Khả năng Nhớ (Bảng 2)**

**Cải thiện đáng kể so với mô hình tuyến tính tốt nhất:**
- Ở 340M: MoM cải thiện **11.3%** (27.59 vs 24.78 của Gated DeltaNet)
- Ở 1.3B: MoM cải thiện **11.5%** (36.04 vs 32.30 của Gated DeltaNet)

**Thu hẹp khoảng cách với Transformer:**
- Ở 340M: MoM đạt **87.0%** hiệu suất của Transformer (27.59/31.70)
- Ở 1.3B: MoM đạt **96.5%** hiệu suất của Transformer (36.04/37.31)

→ **Kết luận**: MoM gần như loại bỏ khoảng cách giữa mô hình tuyến tính và Transformer trên tác vụ nhớ.

### **2. Hiệu quả Ổn định trên Tác vụ Suy luận (Bảng 3)**

**Cải thiện nhẹ nhưng nhất quán:**
- Ở 340M: MoM tốt hơn ~1.5% so với mô hình tuyến tính tốt nhất
- Ở 1.3B: MoM tốt hơn ~0.2% so với mô hình tuyến tính tốt nhất

**Vượt trội hơn Transformer ở một số chỉ số:**
- Perplexity thấp hơn đáng kể (14.83 vs 19.29 ở Lambada 1.3B)
- Hiệu suất tổng thể tương đương hoặc tốt hơn

### **3. Đánh giá Tổng thể**

**Điểm mạnh của MoM:**
- ✅ **Đột phá trên tác vụ nhớ**: Cải thiện 11-12% so với SOTA linear models
- ✅ **Hiệu quả quy mô**: Càng lớn càng hiệu quả (đặc biệt ở 1.3B)
- ✅ **Đa năng**: Hoạt động tốt trên cả tác vụ nhớ và suy luận
- ✅ **Chi phí thấp**: Duy trì độ phức tạp O(n) của mô hình tuyến tính

**Ý nghĩa thực tiễn:**
- MoM là giải pháp **cân bằng lý tưởng** giữa hiệu suất và chi phí tính toán
- Đặc biệt phù hợp cho các ứng dụng cần xử lý chuỗi dài với yêu cầu nhớ cao
- Mở ra khả năng thay thế Transformer trong nhiều ứng dụng thực tế

**Kết luận**: MoM đạt được mục tiêu thiết kế - **giữ chi phí tính toán thấp của mô hình tuyến tính nhưng đạt hiệu suất gần bằng Transformer**, đặc biệt ấn tượng trên các tác vụ đòi hỏi khả năng nhớ.
