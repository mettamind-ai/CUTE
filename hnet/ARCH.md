`mixer_seq.py` định nghĩa lớp HNetForCausalLM. Lớp này làm 3 việc:
    1. Tạo một lớp `nn.Embedding` để chuyển input_ids (các con số) thành vector.
    2. Sử dụng HNet làm "backbone" để xử lý các vector này.
    3. Tạo một lớp lm_head để chuyển các vector đầu ra của HNet thành logits (xác suất cho từ tiếp theo)

