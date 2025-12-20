import re, sys, json
from collections import defaultdict
import sentencepiece as spm

notalphabet = '[^a-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ]'
separator = re.compile(f'({notalphabet}+)', re.IGNORECASE)
sep_token = "<|sep|>"

prefix = "tuoc"; INPUT_DATA = "vi-en-code_50k.jsonl.xz"; MAX_LINE = 50_000
config = { "uncase": False, "max_ngram": 4, "vocab_size": 48_000, "sp_vocab_size": 8000 }

prefix = "tuoc_vi-uncase-24k"; INPUT_DATA = "vi_80k.jsonl.xz"; MAX_LINE = 100_000
config = { "uncase": True, "max_ngram": 4, "vocab_size": 24_000, "sp_vocab_size": 3000 }

prefix = "tuoc_vi-uncase-24k-6gram"; INPUT_DATA = "vi_80k.jsonl.xz"; MAX_LINE = 80_000
config = { "uncase": True, "max_ngram": 6, "vocab_size": 24_000, "sp_vocab_size": 4000 }

prefix = "tuoc_vi-24k-6gram"; INPUT_DATA = "vi_80k.jsonl.xz"; MAX_LINE = 80_000
config = { "uncase": False, "max_ngram": 6, "vocab_size": 24_000, "sp_vocab_size": 4000 }

ngram_vocab_size = config["vocab_size"] - config["sp_vocab_size"]
sp_input_file = "nonwords_remain_ngrams.txt"

with open(f"{prefix}.json", "wt") as f: f.write(json.dumps(config))

def parse(text):
    """Bóc tách text thành 2 tập words và nonwords. Sau đó sẽ:
    - nối words của các text với nhau để build vocab theo n-gram
    - nối các nonwords của các text vơi nhau để build vocab theo sentencepiece
    - một text sẽ được biểu diễn lại bởi 1 cấu trúc dữ liệu gồm 2 thành phần:
      * `words`: list các word-tokens của text
      * `nonwwords`: list các nonword-tokens của text
      * Trong `words` có 1 token đặc biệt là "<|sep|>", nó sẽ đc thay thế lần lượt bởi
        `nonwords` để khôi phục lại nội dung gốc của text
    """
    # Việc đầu tiên là cắt nhỏ text thành các chunks đan xen giữa word và nonword tokens
    chunks = re.split(separator, text)

    # Loại bỏ các chunks đầu chỉ chứa toàn spaces
    for start, x in enumerate(chunks):
        if len(x.strip()) > 0: break
    chunks = chunks[start:]

    # Xác định not word index, chỉ có thể là lẻ (1), hoặc chẵn (0)
    # - là 1 tức là token đầu tiên là word => word nonword word nonword ...
    # - và 0 thì token đầu tiên là nonword => nonword word nonword word ...
    nwi = 0 if re.match(separator, chunks[0]) else 1#print(chunks[0], nwi)

    # Chia chunks thành 2 tập words và nonwords
    words = []; nonwords = []
    for i, x in enumerate(chunks):
        if i % 2 == nwi:
            x = x.strip()
            if len(x) == 0: continue
            nonwords.append(x)
            if len(words) == 0 or words[-1] != sep_token:
                words.append(sep_token)
        else:
            words.append(x)

    # Loại bỏ từ ở cuối nếu nó là rỗng
    if len(words[-1].strip()) == 0:
        words = words[:-1]

    # Thêm sep_token ở cuối để không bị lẫn với text khác khi merge chung với nhau
    if len(words) == 0 or words[-1] != sep_token:
        words.append(sep_token)

    return words, nonwords


def restore_text(words, nonwords):
    txt = ""
    idx = 0; n = len(nonwords)
    for w in words:
        if idx < n and w == sep_token:
            w = nonwords[idx]
            idx += 1
        txt += w + " "
    return txt


def ngram(words, n):
    ans = []; temp = zip(*[words[i:] for i in range(0, n)])
    for ngram in temp:
        if sep_token in ngram: continue
        ans.append(" ".join(ngram))
    return ans


import json, lzma
import multiprocessing as mp
def extract(jsonl_file=INPUT_DATA, max_line=MAX_LINE, workers=16):
    docs = []
    with lzma.open(jsonl_file) as fin:
        for i, line in enumerate(fin):
            if i == max_line: break
            t = json.loads(line)["text"]
            if config["uncase"]: t = t.lower()
            docs.append(t)
    print(len(docs))

    all_words = []; all_nonwords = []
    with mp.Pool(workers) as p:
        for words, notwords in p.imap_unordered(parse, docs):
            all_words += words
            all_nonwords += notwords
    return all_words, all_nonwords


if __name__ == '__main__':
    words, nonwords = extract()
    print("(( THỐNG KÊ N-GRAM VỚI WORDS ))\n")
    # Thống kê số lượng 1,2,3,4-grams
    count = defaultdict(int)
    for i in range(1, 6):
        for x in ngram(words, i): count[x] += 1
    words = None # giải phóng bộ nhớ

    # Sort count theo thứ tự giảm dần
    desc_count = sorted(count.items(), key=lambda x:-x[1])
    print(desc_count[:100], "...\n\n")

    print(f"(( CHỌN {ngram_vocab_size} TỪ VỰNG TỪ N-GRAM ))")
    selected_ngrams = desc_count[:ngram_vocab_size]
    remain_ngrams = desc_count[ngram_vocab_size:]
    print(selected_ngrams[0:300], "\n\n")
    with open(f"{prefix}.ngram", "wt") as f:
        for ngram in selected_ngrams:
            f.write(f"{ngram[1]} {ngram[0]}\n")


    print(f"((NONWORDS VÀ {len(remain_ngrams)} NGRAMS CÒN LẠI ...\n")
    print(f"... MANG ĐI HUẤN LUYỆN SENTENCEPIECE ))\n")

    unigrams = []
    for x in selected_ngrams:
        if " " not in x[0]:
            unigrams.append(x[0])
    unigrams = set(unigrams)

    remain_text = ""; i = 0
    for ngram in remain_ngrams:
        for x in ngram[0].split():
            if x in unigrams: continue
            i += 1; remain_text += " " + x
            if i % 20 == 0: remain_text += "\n"

    with open(sp_input_file, "wt") as f:
        f.write(" ".join(nonwords) + "\n")
        f.write(remain_text)

    spm.SentencePieceTrainer.train(input=sp_input_file, model_prefix=prefix, 
        user_defined_symbols=[], model_type="bpe", vocab_size=config["sp_vocab_size"])

    sp = spm.SentencePieceProcessor(model_file=f'{prefix}.model')
    vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
    print(vocabs)
