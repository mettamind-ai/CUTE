## BPEasy
> You can think of bpeasy as the tiktoken training code that never was.

Treat text data at the **byte-level first** --- convert to bytes before training rather than using characters (Huggingface).

Always use a **regex-based split pre-tokenizer**. This is a customisable regex that is applied to the text before training.
```python
# should be an iterator over str
iterator = jsonl_content_iterator(args)
# example regex from GPT-4
regex_pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
# returns the vocab (dict[bytes, int])
vocab = bpeasy.train_bpe(
    iterator,
    regex_pattern,
    args.max_sentencepiece_length, # max length of tokens
    args.vocab_size, # max size of vocab
)
```

## Byte Latent Transformer

![](https://arxiv.org/html/2412.09871v1/extracted/6066458/assets/patching_types.png)

![](https://arxiv.org/html/2412.09871v1/x4.png)

![](https://arxiv.org/html/2412.09871v1/x5.png)

![](https://arxiv.org/html/2412.09871v1/x3.png)

![](https://arxiv.org/html/2412.09871v1/extracted/6066458/assets/patching.png)