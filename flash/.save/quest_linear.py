# https://github.com/IST-DASLab/QuEST/blob/main/src/models/quantization/base_linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform

GAUSSIAN_SCALES = { 1:0.7978845587140913, 1.585:1.2240089519030855, 2:1.4935346200015913, 3:2.051068354131873,
    4:2.513930578568423, 5:2.9160938834961225, 6:3.276597282593217, 7:3.6010497188221655, 8:3.884938678807525,}

class HalfHadamardTrustQuantizer(nn.Module):
    aux_matrix = hadamard_transform(torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2))

    def __init__(self, bits=4, trust=None):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits
        self.matrix = None
        self.trust = GAUSSIAN_SCALES[self.bits]/(self.n_levels-1) if trust is None else trust

    def forward(self, x):
        if self.matrix is None: # khởi tạo lần đầu khi có dữ liệu x
            self.matrix = torch.block_diag(*[self.aux_matrix.to(x.device).to(x.dtype)]*(x.shape[-1]//128),)

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.weight_quantizer = HalfHadamardTrustQuantizer()
        self.activation_quantizer = HalfHadamardTrustQuantizer()

    def forward(self, x):
        x = self.activation_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias).to(x.dtype)



if __name__ == "__main__":
    import time
    import numpy as np
    from torch.nn import Linear
    torch.set_default_dtype(torch.float)
    
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tham số test
    batch_size = 4
    seq_len = 32
    in_features = 256  # Phải là bội số của 128
    out_features = 128
    input_shape = (batch_size, seq_len, in_features)
    
    # Tạo dữ liệu đầu vào
    x = torch.randn(*input_shape, device=device)
    
    # Tạo model
    model_quant = QuantizedLinear(in_features, out_features).to(device)
    model_linear = Linear(in_features, out_features).to(device)
    
    # Đồng bộ trọng số và bias giữa 2 model để so sánh công bằng
    with torch.no_grad():
        model_linear.weight.copy_(model_quant.weight)
        if model_quant.bias is not None:
            model_linear.bias.copy_(model_quant.bias)
    
    # In thông tin model
    print(f"Quantized Model: {model_quant}")
    print(f"Standard Linear Model: {model_linear}")
    print(f"Input shape: {x.shape}")
    
    # Warm-up
    for _ in range(3):
        model_quant(x)
        model_linear(x)
    
    # Đo thời gian thực thi
    def measure_time(model, x, n_runs=100):
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x)
        return (time.time() - start) * 1000 / n_runs  # ms per run
    
    # Đo thời gian
    time_quant = measure_time(model_quant, x)
    time_linear = measure_time(model_linear, x)
    
    # Tính toán output
    with torch.no_grad():
        out_quant = model_quant(x)
        out_linear = model_linear(x)
    
    # Tính toán sai số
    mse = torch.mean((out_quant - out_linear) ** 2).item()
    mae = torch.mean(torch.abs(out_quant - out_linear)).item()
    max_error = torch.max(torch.abs(out_quant - out_linear)).item()
    
    # In kết quả
    print("\n=== Kết quả so sánh ===")
    print(f"Quantized Linear time: {time_quant:.4f} ms/run")
    print(f"Standard Linear time: {time_linear:.4f} ms/run")
    print(f"Tốc độ chậm hơn: {time_quant/time_linear:.2f}x")
    print(f"\nSai số bình phương trung bình (MSE): {mse:.6f}")
    print(f"Sai số tuyệt đối trung bình (MAE): {mae:.6f}")
    print(f"Sai số lớn nhất: {max_error:.6f}")
    
    # In thống kê output
    print("\n=== Thống kê output ===")
    print("Quantized Output - Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
        out_quant.min().item(), out_quant.max().item(), 
        out_quant.mean().item(), out_quant.std().item()))
    print("Linear Output - Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}, Std: {:.4f}".format(
        out_linear.min().item(), out_linear.max().item(), 
        out_linear.mean().item(), out_linear.std().item()))
    
    print("\nKết thúc kiểm tra!")