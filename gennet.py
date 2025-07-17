#!/usr/bin/env python3
# Tham khảo thêm https://github.com/main-horse/hnet/blob/main/generate.py
import json, torch, numpy as np

from hnet.config import AttnConfig, SSMConfig, HNetConfig
from hnet.hlm import HNetForCausalLM

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
    config = json.load(open(model_config_path).read())
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
        output = model.forward(input_ids, mask=mask, inference_params=inference_cache)

    for _ in range(max_tokens):
        # Get logits and apply temperature
        logits = output.logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        current_token = next_token.unsqueeze(0)
        yield current_token

        if next_token.item() == tokenizer.eos_idx: break  # dừng sinh khi gặp EOS token
        with torch.inference_mode(): output = model.step(current_token, inference_cache)


if __name__ == "__main__":

    model = load_from_pretrained("hnet/2stage_L.pt", "hnet/2stage_L.json")
    tokenizer = ByteTokenizer()

    while True:
        prompt = input("\nPrompt: ").strip()
        print(f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature})")
        print(f"\033[92m{prompt}\033[0m", end="")
        buf = []

        for token in generate(model, prompt, max_tokens=args.max_tokens, temperature=args.temperature):
            buf.append(token)
            for j in range(1, min(len(buf), 4)):
                try:
                    print(tokenizer.decode(buf[:j]), end="", flush=True)
                    buf = buf[j:]
                    break
                except: pass
