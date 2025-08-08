#!/usr/bin/env python3
# Tham khảo thêm https://github.com/main-horse/hnet/blob/main/generate.py
import json, torch, os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

import torch, torch.nn as nn
from hnet import HNetForCausalLM
from utils import HNetConfig, AttnConfig, SSMConfig, ByteTokenizer

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

    model = load_from_pretrained(f"{current_dir}/2stage_L.pt", f"{current_dir}/2stage_L.json")
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
