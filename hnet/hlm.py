import torch, torch.nn as nn

from hnet import HNet, HNetState
from config import HNetConfig
from dc import RoutingModuleOutput

class HNetForCausalLM(nn.Module):
    def __init__(self, config: HNetConfig, device=None, dtype=None,) -> None:
        self.config = config
        super().__init__()

        vocab_size, d_embed = self.config.vocab_size, self.config.d_model[0]
        self.embeddings = nn.Embedding(vocab_size, d_embed, device=device, dtype=dtype)
        self.backbone = HNet(config=config, stage_idx=0, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, device=device, dtype=dtype)
        if self.config.tie_embeddings: self.lm_head.weight = self.embeddings.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, mask=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """ num_last_tokens: if > 0, only return the logits for the last n tokens """
        x = self.embeddings(input_ids)
        B, L, D = x.shape
        cu_seqlens = max_seqlen = None

        if mask is None:
            # Nếu người dùng gọi hàm forward mà không cung cấp một mask (tức là mask is None), thì chúng ta sẽ mặc định rằng
            # dữ liệu đầu vào đã được chuẩn bị để chạy ở chế độ Nén (Packed Mode). Do đó, code sẽ tiến hành nối các chuỗi lại
            # (flatten) và tự tạo ra cu_seqlens để các lớp bên trong (như Mamba) có thể xử lý hiệu quả.
            assert inference_params is None, "Inference params are not supported in packed mode"
            x = x.flatten(0, 1)  # Merge batch và sequence dimensions
            cu_seqlens = torch.arange(B + 1, device=x.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=x.device)

        x, bpred_output = self.backbone(x, cu_seqlens, max_seqlen, mask, inference_params, **mixer_kwargs)
        x = x.view(B, L, D)
        if num_last_tokens > 0: x = x[:, -num_last_tokens:]
        logits = self.lm_head(x)
        return (logits, bpred_output, inference_params)

    def step(self, input_ids, inference_params):
        bs = input_ids.shape[0]
        assert bs == 1, "HNetForCausalLM step currently only supports batch size 1"
        x = self.embeddings(input_ids)
        x, bpred_output = self.backbone.step(x, inference_params)
        logits = self.lm_head(x)
        return (logits, bpred_output, inference_params)
