TODOs
- [x] Thử Lion optimizer (không tốt)
- [x] Tại sao bản viết lại chậm 1.5x so với bản gốc? (cần set trainer về "bf16")
- [x] Sử dụng gradient checkpoint để tiết kiệm vram (rất tốt cho mô hình lớn)
- [ ] Fused kernel cho L2Wrap
- [ ] Fused kernels cho channel-mixing (có thể dùng Triton)
  - `sigmoid(receptance @ xr)` dùng trong cả att và fnn
  - `square(relu(key @ xk))` dùng trong ffn
  - Note: nên chờ tính năng tự động fused của torch.compile 2.0 => không cần mất time để viết
- [ ] 8-bit training https://github.com/TimDettmers/bitsandbytes

# Thu gọn [rwkv-lm](https://github.com/BlinkDL/RWKV-LM) còn ~600 dòng code
- Code đơn giản và dễ hiểu nhất có thể
- Comments đủ rõ ràng để người mới có thể hiểu
- Tối đa hóa năng lực tính toán (xem TODOs trên)

### Mã nguồn
- `model.py`: mô hình ngôn ngữ RWKV, chỉ hỗ trợ kiểu dữ liệu bf16 để tiết kiệm bộ nhớ và sử dụng được nhân cuda tăng tốc phép nhân ma trận. Sử dụng lightning trainer support nhiều GPUs, và deepspeed fused adam, dùng deepspeed gradient checkpoint để tiêt kiệm vram (nhằm tăng batch_size) khi cần

- `dataset.py`: nạp dữ liệu văn bản đã được tokenized và lưu dưới dạng nhị phân (binidx) để có thể nạp hàng trăm GB dữ liệu vào mô hình mà không tốn thời gian tiền xử lý và không tràn bộ nhớ, nhờ sử dụng memory mapped.

- `train.py` toàn bộ kịch bản huấn luyện.

- `wkv_cuda.cu`: nhân cuda để tăng tốc tính TimeMix (quan trọng)
