import tokenmonster # pip install tokenmonster

tokenmonster.set_local_directory(".")

# https://huggingface.co/datasets/alexjc/fineweb-tokmon-10B
# wget https://huggingface.co/datasets/alexjc/fineweb-tokmon-10B/resolve/main/english-28416-balanced-v1/fineweb-tokmon_train_000001.bin
vocab = tokenmonster.load("english-28416-balanced-v1.vocab")

text = "Some text to turn into token IDs."
tokens = vocab.tokenize(text)
print(tokens)
