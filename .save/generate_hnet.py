# Tham khảo thêm https://github.com/main-horse/hnet/blob/main/generate.py
import torch, numpy as np
import json, argparse, sys
from omegaconf import ListConfig
from hnet import HNetForCausalLM, AttnConfig, SSMConfig, HNetConfig

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


def generate(model, prompt:str, max_tokens=512, temperature=.6, top_p=.9):
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
        '''
        if top_p < 1.0: # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("inf")
        # '''
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        current_token = next_token.unsqueeze(0)
        yield current_token

        if next_token.item() == tokenizer.eos_idx: break  # dừng sinh khi gặp EOS token
        with torch.inference_mode(): output = model.step(current_token, inference_cache)


def main():
    parser = argparse.ArgumentParser(description="Generate text from an H-Net model")
    parser.add_argument("--model-path",  type=str, required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the model configuration (.json file)")
    parser.add_argument("--max-tokens",  type=int,  default=1024, help="Maximum number of tokens to generate (default: 1024)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p",       type=float, default=1.0, help="Top-p sampling parameter (default: 1.0)")
    args = parser.parse_args()

    print("Loading model...")
    model = load_from_pretrained(args.model_path, args.config_path)
    tokenizer = ByteTokenizer()

    while True:
        prompt = input("\nPrompt: ").strip()
        print(f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})")
        print(f"\033[92m{prompt}\033[0m", end="")
        token_count = 0
        buf = []

        for token in generate(model, prompt, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p):
            buf.append(token)
            token_count += 1
            decoded = res = None
            for j in range(1, min(len(buf), 4)):
                try:
                    res = tokenizer.decode(buf[:j])
                    decoded = j
                except:
                    pass

            if res is not None:
                print(res, end="", flush=True)
                buf = buf[decoded:]


if __name__ == "__main__":
    main()
