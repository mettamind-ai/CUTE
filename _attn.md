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

