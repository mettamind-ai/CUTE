import tokenmonster # pip install tokenmonster
import lzma, json
import numpy as np

tokenmonster.set_local_directory(".")

# https://huggingface.co/datasets/alexjc/fineweb-tokmon-10B
# wget https://huggingface.co/datasets/alexjc/fineweb-tokmon-10B/resolve/main/english-28416-balanced-v1/fineweb-tokmon_train_000001.bin
vocab = tokenmonster.load("english-28416-balanced-v1.vocab")

text = "Some text to turn into token IDs."
tokens = vocab.tokenize(text)
print(tokens)

et = 32000-1
tids = []

for i in range(2):
	filename = f"tinystories.en_{i}.jsonl.xz"
	for line in lzma.open(filename, "rt"):
		data = json.loads(line)
		text = data["text"]
		tids += vocab.tokenize(text) + [et]

# Convert tids to numpy array and save to data.bin
tids_array = np.array(tids, dtype=np.int16)
print(f"Array shape: {tids_array.shape}, dtype: {tids_array.dtype}")

# Save to binary file
filename = f"data{et+1}.bin"
tids_array.tofile(filename)
print(f"Saved {len(tids)} tokens to {filename}")
