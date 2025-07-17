#!/usr/bin/env python3
# Tham khảo thêm https://github.com/main-horse/hnet/blob/main/generate.py
import json, torch, numpy as np, os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import torch, torch.nn as nn
from hnet import HNet
from utils import HNetConfig, AttnConfig, SSMConfig

class HNetForCausalLM(nn.Module):
    def __init__(self, config: HNetConfig, device=None, dtype=None) -> None:
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


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255

    def text2bytes(self, text, add_bos, add_eos):
        text_byte = text.encode("utf-8")
        if add_bos: text_byte = bytes([self.bos_idx]) + text_byte
        if add_eos: text_byte = text_byte + bytes([self.eos_idx])
        return np.array(bytearray(text_byte), dtype=np.uint8)

    def encode(self, seqs, add_bos=False, add_eos=False, **kwargs):
        return [ { "input_ids": self.text2bytes(text, add_bos, add_eos) } for text in seqs ]

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, np.ndarray): tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)


def load_from_pretrained(model_path: str, model_config_path: str):
    with open(model_config_path, "r") as f: config = json.load(f)
    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg  = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    model = HNetForCausalLM(hnet_cfg, device="cuda", dtype=torch.bfloat16)
    state_dict = torch.load(model_path, map_location="cuda", weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def generate(model, prompt:str, max_tokens=512, temperature=.6):
    device = next(model.parameters()).device
    tokenizer = ByteTokenizer()

    encoded = tokenizer.encode([prompt], add_bos=True)[0]
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    inference_cache = model.allocate_inference_cache(1, input_ids.shape[1] + max_tokens, dtype=torch.bfloat16)

    with torch.inference_mode():  # khởi tạo và chạy forward lần đầu
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        logits, _, _ = model.forward(input_ids, mask=mask, inference_params=inference_cache)

    for _ in range(max_tokens):
        # Get logits and apply temperature
        logits = logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        current_token = next_token.unsqueeze(0)
        yield current_token

        if next_token.item() == tokenizer.eos_idx: break  # dừng sinh khi gặp EOS token
        with torch.inference_mode(): logits, _, _ = model.step(current_token, inference_cache)


if __name__ == "__main__":

    model = load_from_pretrained("hnet/2stage_L.pt", "hnet/2stage_L.json")
    tokenizer = ByteTokenizer()

    while True:
        prompt = input("\nPrompt: ").strip()
        print(f"\033[92m{prompt}\033[0m", end="")
        buf = []

        for token in generate(model, prompt, max_tokens=512, temperature=0.6):
            buf.append(token)
            for j in range(1, min(len(buf), 4)):
                try:
                    print(tokenizer.decode(buf[:j]), end="", flush=True)
                    buf = buf[j:]
                    break
                except: pass
