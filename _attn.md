https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton

https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/
playground/68520e2b021167d8d85da9b7 Khi nhiều luồng xử lý cùng truy cập vào bộ nhớ chia sẻ, chúng có thể va chạm nếu cùng đọc từ một ngăn nhớ. Điều này khiến các luồng phải xếp hàng chờ, làm chậm quá trình tính toán. Để giải quyết vấn đề này, swizzle **sắp xếp lại cách lưu trữ dữ liệu**. Thay vì để các luồng liên tiếp truy cập cùng một ngăn, nó phân tán chúng ra các ngăn khác nhau. Ví dụ như luồng thứ nhất đọc từ ngăn 0, luồng thứ hai đọc từ ngăn 8, luồng thứ ba đọc từ ngăn 16. Cách làm này giúp các luồng có thể đọc dữ liệu đồng thời mà không phải chờ nhau.
```
Block_M = 128, Block_N = 128 (cho head_dim ≤  64)
Block_M =  64, Block_N =  64 (cho head_dim = 128)
```
- mặc định cho SM80 (và cả SM89/RTX 4090) là 128x128.
- Varlen forward đã có nhưng backward chưa hoàn thành
- ... Tri Dao còn đang hoàn thiện cutefa ...

---

# Cách đơn giản nhất để FA hỗ trợ đa dạng masking hơn là dùng block masking, với block là processing unit của FA

=> Efficiently Dispatching Flash Attention For Partially Filled Attention Masks
https://www.alphaxiv.org/abs/2409.15097 tạo ra một ma trận nhị phân "Binary Block Matrix" có kích thước `N//BLOCKSIZE_I × N//BLOCKSIZE_J`, trong đó mỗi phần tử được đặt bằng 1 nếu khối mask tương ứng có ít nhất một giá trị khác không. Như tác giả nhấn mạnh: "Our method focuses on optimizing Flash Attention by processing only the blocks of the attention matrix that have non-zero entries in their corresponding mask blocks." Điều này giúp giảm đáng kể tính toán không cần thiết. Việc áp dụng thuật toán Reverse Cuthill-McKee để tối ưu hóa các mask cực kỳ thưa thớt. Bằng cách sắp xếp lại cấu trúc ma trận, RCM giúp tăng "fill-in" của ma trận và giảm số lượng khối cần xử lý. Tác giả đã chứng minh: "Figure 2a illustrates a case where RCM reduces the number of blocks by 50%. Figure 2b shows results on a synthetic mask where RCM preprocessing reduces the number of blocks by 90%."
![](https://pbs.twimg.com/media/GtsK0gwaoAACItr?format=jpg&name=medium)

---

# MoBA - learnable sparse attn based on flash-attn (400 loc of python)
![](https://github.com/MoonshotAI/MoBA/raw/master/figures/running_example.png)

# Semantic focus
https://www.alphaxiv.org/abs/2506.14095

Bài báo này đưa ra một góc nhìn mới về cơ chế sparse attention (chú ý thưa) trong mô hình Transformer, không chỉ tập trung vào hiệu quả tính toán mà còn vào khả năng học và tổng quát hóa của mô hình.

Đầu tiên, nghiên cứu cho thấy "sparse attention với một dạng mẫu thưa phụ thuộc vào đầu vào giới hạn sự chú ý vào các điểm chú ý cao nhất - heavy-hitters (như `top-k attention`) - trên thực nghiệm có khả năng biểu diễn tương đương với full attention tiêu chuẩn, và có thể hội tụ nhanh hơn đáng kể trong quá trình huấn luyện, trong khi tổng quát hóa tốt bằng, và đôi khi tốt hơn, mô hình full attention" (trang 2).

Thứ hai, công trình này cung cấp một nền tảng lý thuyết vững chắc để giải thích cho những quan sát thực nghiệm. Các tác giả đã thiết lập mối liên hệ giữa sự ổn định của hàm softmax trong cơ chế attention với hằng số Lipschitz của hàm mất mát. Họ chỉ ra rằng việc giới hạn sự **chú ý vào các token "heavy-hitter" giúp giảm "độ phân tán ngữ nghĩa"** (semantic dispersion), từ đó `cải thiện sự ổn định của softmax`, dẫn đến hằng số Lipschitz tốt hơn và cuối cùng là đảm bảo sự hội tụ nhanh và khả năng tổng quát hóa tốt hơn. Bài báo nêu rõ: "Chúng tôi chỉ ra rằng hằng số Lipschitz của một mô hình dựa trên transformer gắn liền với sự ổn định đầu vào của softmax trong cơ chế attention... sự thưa phụ thuộc vào đầu vào chỉ tập trung vào các heavy-hitters có thể cải thiện đáng kể độ phân tán này, do đó ngụ ý sự ổn định đầu vào được cải thiện. Điều này chuyển thành một hằng số Lipschitz được cải thiện, do đó đảm bảo sự hội tụ và tổng quát hóa tốt hơn" (trang 2).

Cuối cùng, bài báo tạo ra một sự phân biệt rõ ràng và quan trọng giữa sparse attention phụ thuộc vào đầu vào (input-dependent) và không phụ thuộc vào đầu vào (input-agnostic). Trong khi các phương pháp input-agnostic như banded hay block-local attention thường gặp khó khăn về khả năng biểu diễn và không mang lại lợi ích rõ rệt về mặt học tập, các phương pháp input-dependent lại cho thấy hiệu quả vượt trội. Điều này cung cấp một định hướng quan trọng cho các nghiên cứu trong tương lai về việc lựa chọn và phát triển các dạng sparse attention hiệu quả. Bài báo nhận thấy rằng "sparse attention với các mẫu thưa không phụ thuộc vào đầu vào (input-agnostic) trên thực nghiệm gặp khó khăn với khả năng biểu diễn... và không cho thấy lợi ích về mặt hội tụ học tập và tổng quát hóa" (trang 2), trái ngược với kết quả tích cực của các phương pháp phụ thuộc vào đầu vào.

