import tokenmonster, sys, time, json, lzma

filename = sys.argv[1]
vocab = sys.argv[2]
try:
    ctx_len = int(sys.argv[3])
    window = (ctx_len // 3) + 1
    print(f"Padding for ctx_len {ctx_len} ...")
except:
    ctx_len = False

monster = tokenmonster.load(vocab)
print("tokenmonster ...")

chars_count = 0; tids_count = 0; tic = time.time(); bytes_count = 0
if "xz" in filename or "lzma" in filename:
    f = lzma.open(filename, "rt")
else:
    f = open(filename, "rt")

for idx, line in enumerate(f):
    text = json.loads(line)["text"]
    if len(text) < 3: continue 
    tids = monster.tokenize(text)
    this_chars_count = len(text)
    this_tids_count = len(tids) + 1 # sep tokens

    chars_count += this_chars_count
    bytes_count += len(text.encode('utf-8'))

    if ctx_len: # need padding and window slicing
        if this_tids_count > 2 * window:
            this_tids_count += window

        # Luôn lấy sample đầu tiên có độ dài ctx_len
        tids_count += ctx_len
        curr_sample_end = ctx_len # end of sample đầu tiên

        # Nếu end of curr sample vẫn nằm trong số tokens đang có thì tiếp tục nhập kho mẫu tiếp theo
        while curr_sample_end <= this_tids_count:
            tids_count += ctx_len # nhập kho
            curr_sample_end += window # di chuyển mẫu đi độ dài window


    else:
        tids_count += this_tids_count

    idx += 1
    if idx % 3000 == 0:
        r = chars_count / tids_count
        f = tids_count / idx
        t = bytes_count / tids_count
        seconds = time.time() - tic
        print(idx, tids_count, f"avg. chars / token {r}, avg. bytes / token {t}, avg. tokens / doc {f}")

f.close()