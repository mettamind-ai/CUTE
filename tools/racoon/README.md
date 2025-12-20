# Racoon is RWKV


## racoon4, rwkv4, train4
- [x] Tích hợp `packed_dataset` và quản lý data hiệu quả hơn
    - Chỉ feed dữ liệu tuần tự, không shuffle, không random access (=> dữ liệu cần được trộn đều từ trước)
    - Có thể lựa chọn liệu từ nhiều nguồn với tỉ lệ khác nhau `data_a.jsonl:1.0,data_b.jsonl:0.9,hf-dataset-name:0.5`
    - Số samples phải được estimate thủ công bằng công thức `samples = data's bytes / agv bytes per token`
- [x] Sử dụng gradient checkpoint và cpu offload
- [x] Test racoon is rwkv forward


## racoon7, rwkv7, train7
`model7.py` là implement gốc của rwkv7, giữ lại để làm tiêu chuẩn.

- [x] Thêm rwkv7
- [x] racoon7
- [ ] train7
    - `./train7prepare.sh` => cần `data/minipile.idx`
