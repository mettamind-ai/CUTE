import tknz, sentencepiece, json, sys, lzma, time, os

prefix = "tuoc_vi-24k-6gram"
config = json.loads(open(f"{prefix}.json").read())

# Load sentencepiece model, dùng để tknz những text ko tknz đc bằng n-grams
sp = sentencepiece.SentencePieceProcessor(model_file=f'{prefix}.model')

# Trie là một cấu trúc dữ liệu hiệu quả để lưu trữ và match text vào n-gram
# Node cuối của một trie luôn là tknz.sep_token và lưu trữ vocab_id của nó
def make_trie(ngrams):
    root = dict()
    for ngram in ngrams:
        idx = (int(ngram[0]), len(ngram) - 1)
        current_dict = root
        for word in ngram[1:]:
            current_dict = current_dict.setdefault(word, {})
        current_dict[tknz.sep_token] = idx
    return root

# Load n-grams từ file text và chuyển hóa thành trie data structure
ngrams = open(f"{prefix}.ngram").read().strip().split("\n")
ngrams = [tuple([idx + config["sp_vocab_size"]] + x.split()[1:]) for idx, x in enumerate(ngrams)]
trie = make_trie(ngrams)
# print(ngrams, trie) #DEBUG

# Tìm kiếm cách match words vào n-grams với số ids là nhỏ nhất
# Sẽ dừng tìm kiếm khi gặp tknz.sep_token hoặc một từ không có trong ngram trie
next_token = "-->"
unknown_token = "<|unk|>"
MAX_TRIALS = 9999
sys.setrecursionlimit(MAX_TRIALS + 100)
def greedy_search(words, tids, final, _trie = trie, n = 0, tokens = 0):
    if final["trials"] > MAX_TRIALS: return
    if final["best"] is not None: # đã có phương án
        x = final["last"] - n
        if x <= 4: min_tokens = 1
        else: min_tokens = int(x / 2.5)
        if tokens + min_tokens >= final["best"]:
            final["trials"] = MAX_TRIALS # thoát luôn :D

    w = words[n]
    if w != tknz.sep_token and w in _trie:
        tids[n] = next_token # đánh dấu ô này
        greedy_search(words, tids, final, _trie[w], n + 1, tokens)
        if tknz.sep_token in _trie: # có thể kết thúc token
            tids[n-1], _ = _trie[tknz.sep_token] # lấy token id, và token len
            greedy_search(words, tids, final, trie, n, tokens + 1)

    else:
        if tknz.sep_token in _trie: # cuối của trie
            tids[n-1], _ = _trie[tknz.sep_token] # lấy token id, và token len
            greedy_search(words, tids, final, trie, n, tokens + 1)

        else: # không có trong trie => kết thúc
            final["trials"] += 1
            if tids[n] != tknz.sep_token:
                tids[n] = unknown_token
            if final["best"] is None or final["best"] > tokens:
                final["best"] = tokens
                if final["last"] == 0: final["last"] = n
                else: assert final["last"] == n
                # Sao chép phương án tốt nhất cho tới hiện tại
                while n >= final["start"]:
                    final["tids"][n] = tids[n]
                    n -= 1


def encode(text):
    if config["uncase"]:
        text = text.lower()

    # Để map đoạn text trên thành các token ids ta làm 2 phần việc sau:
    # 1/ cố gắng dùng n-grams để match đc nhiều nhất có thể phần words của text
    # 2/ Các remain_words (ko matched đc với n-gram) và nonwords thì tách
    words, nonwords = tknz.parse(text)#; print(words)
    final = {
        "best": None, # chưa tìm đc phương án tốt nhất
        "trials": 0, # số phương án tìm được
        "start": 0, # vị trí bắt đầu
        "last": 0, # vị trí cuối của phương án, last cần đạt tới vị trí cuối cùng của words
        "tids": [None] * len(words), # lưu token ids của phương án tốt nhất
    }
    tids = [None] * len(words) # lưu phương án đang trial
    for i, w in enumerate(words):
        if w == tknz.sep_token:
            tids[i] = tknz.sep_token

    start = 0
    while final["last"] < len(words):
        # print(start, words[start], " => ", " ".join(words[start - 1 : start + 3]))
        # reset lại best trước khi bắt đầu tìm kiếm mới
        final["best"] = None; final["trials"] = 0
        final["start"] = start; final["last"] = 0
        greedy_search(words, tids, final, n=start)

        start = final["last"] + 1
        while start < len(words) and \
            (words[start] == tknz.sep_token or \
             words[start] == unknown_token):
            if words[start] == tknz.sep_token:
                final["tids"][start] = tknz.sep_token
            start += 1
        if start == len(words): break
    # print("\nFinal", final, "<==", words, nonwords, "\n")

    # Verify lần cuối
    for i, x in enumerate(final["tids"]):
        assert x is not None # đảm bảo mọi slot đều được map vào 1 token id

    # Lấy token_ids từ n-grams và sentencepiece
    token_ids = []; nw_idx = 0
    for i, w in enumerate(final["tids"]):
        # lấy content của sep token
        if w == tknz.sep_token:
            if nw_idx < len(nonwords):
                w = nonwords[nw_idx]
                nw_idx += 1
            else: continue # hết câu

        # lấy content của unknown token
        if w == unknown_token:
            w = words[i]

        if isinstance(w, int): # là token id
            token_ids.append(w)

        elif w != next_token: # không phải next_token thì dùng sentencepiece để encode
            token_ids += sp.encode(w)

    return token_ids


# Khôi phục lại text
def ngram2text(tid):
    x = ngrams[tid - config["sp_vocab_size"]]
    return " ".join(x[1:])


def decode(token_ids):
    text = ""
    sentencepiece_ids = []
    for tid in token_ids:
        if tid < config["sp_vocab_size"]: # là sentencepiece id
            sentencepiece_ids.append(tid)
        else: # là n-gram id
            if len(sentencepiece_ids) > 0:
                text += sp.decode(sentencepiece_ids) + " "
            sentencepiece_ids = [] # reset
            text += ngram2text(tid) + " "
    return text.strip()


if __name__ == '__main__':
    try: filename = sys.argv[1]
    except: filename = None
    if filename:
        try: # perf tokenmonster
            vocab = sys.argv[2]
            import tokenmonster
            monster = tokenmonster.load(f"{vocab}")
            print("tokenmonster ...")
        except:
            monster = None
            print("tuockenizer ...")
        chars_count = 0; tids_count = 0; tic = time.time()
        with lzma.open(filename, "rt") as f:
            for idx, line in enumerate(f):
                text = json.loads(line)["text"]
                if monster: tids = monster.tokenize(text)
                else: tids = encode(text)
                this_chars_count = len(text)
                this_tids_count = len(tids)
                this_r = this_chars_count / this_tids_count
                # if this_r < 4: print(text) # lọc text kém 
                chars_count += this_chars_count
                tids_count += this_tids_count
                idx += 1
                if idx % 1000 == 0:
                    seconds = time.time() - tic
                    print(idx, f"avg. chars / token {chars_count / tids_count}, \
                        avg. speed {seconds / idx}s / 1k doc")


    # Test với các text đơn lẻ
    text = """ dcác nỗ lực phản công của Ukraine đều thất bạie
Ông' Putin nói rằng Ukraine đã mở chiến dịch phản công được chờ đợi, nhưng không đạt mục tiêu và đang hứng chịu tổn thất lớn.
"Có thể khẳng định rằng chiến dịch phản công của Ukraine đã bắt đầu, khi họ huy động lực lượng dự bị chiến lược. Tuy nhiên, quân đội Ukraine không đạt được mục tiêu nào ở tất cả khu vực tác chiến", Tổng thống Nga Vladimir Putin nói với các phóng viên tại thành phố Sochi hôm 9/6.
Đề cập tới khả năng cuộc phản công của Ukraine đang sa lầy, ông chủ Điện Kremlin cho rằng "các nỗ lực tiến công của đối phương đến nay đều thất bại". Tổng thống Putin nhấn mạnh rằng quân đội Ukraine vẫn còn tiềm năng tấn công, dù đã chịu tổn thất nặng trong những ngày qua.
""" # avg. chars / tokens 6.478260869565218

    text = """Rosie Huntington-Whiteley tiết lộ phải lòng bạn trai hơn 20 tuổi - Jason Statham vì mê tính cách hài hước của anh ấy.
Theo BrightSide, Rosie từng khẳng định rằng bản thân cô biết đến Jason Statham với một hình ảnh nam tính và mạnh mẽ trong các bộ phim anh đóng, tuy nhiên, khiếu hài hước của Jason mới là điều khiến coi phải lòng người vạn trai này.
“Tôi nhớ sau lần đầu tôi gặp Jason, tôi đã điện cho nhỏ bạn thân vào ngay hôm sau và nói rằng [Ê không ngờ ổng khác với tưởng tượng của ta ghê á mày ơi. Jason khiến mình có cảm giác như mình có thể tin cậy được. Ổng còn rất hài hước và tràn đầy năng lượng nữa]. Đó hoàn toàn là những điều tôi không thể ngờ về anh ấy và là điều khiến tôi bị chàng trai đó thu hút.” - Rosie chia sẻ cùng tạp chí Elle.
Đến thời điểm hiện tại, cặp đôi đã quen nhau được hơn 13 năm.
#CuồngPhim #JasonStatham #rosiehuntingtonwhiteley
""" # avg. chars / token 4.560846560846561

    text = """Một phụ huynh vừa liên hệ với tôi nhờ tìm hướng giải quyết cho con của chị. Sau vài tháng sang nước ngoài học đại học, con chị cảm thấy không hòa nhập được với môi trường mới, nằng nặc đòi về nước.
Tám năm trước, tôi cũng từng chứng kiến con trai một người bạn bị nhà trường báo động vì kết quả học tập sa sút. Sau năm rưỡi du học, cậu về nhà làm sinh viên một đại học quốc tế trong nước. Dù vậy, cậu cũng mất nhiều năm chật vật mới có thể tốt nghiệp do lạc nhịp và mất phương hướng sau thời gian ở nước ngoài.
Những chuyện như thế vẫn thường xảy ra trong cộng đồng du học sinh, dù các gia đình đã đầu tư tài chính, quan tâm sát sao; phần lớn những đứa trẻ đều có ý thức và trình độ ngoại ngữ tốt. Tại sao bấy nhiêu sự đầu tư vẫn là chưa đủ?
Trước hết, trong quá trình chuẩn bị du học, du học sinh và gia đình ít khi được nghe nói về triết lý sư phạm của trường mà mình muốn đến. Triết lý sư phạm của nhà trường rất quan trọng, cho phép mỗi sinh viên nhận thấy đây có phải là cơ hội phù hợp để phát triển bản thân hay không.
Tuần trước, tôi phỏng vấn một sinh viên vừa hoàn tất chương trình đại cương tại nơi khác và muốn dự tuyển vào chương trình đào tạo kỹ sư tại trường chúng tôi. Ứng viên này có bảng điểm rất ấn tượng và mục tiêu rõ ràng về công việc muốn phát triển sau khi tốt nghiệp. Đồng nghiệp - phụ trách bộ phận quan hệ giữa nhà trường, doanh nghiệp và sinh viên - tham gia buổi phỏng vấn cùng tôi, rất thích ứng viên này. Lúc đầu tôi có cùng ấn tượng tốt như vậy. Tuy nhiên, sau khi tìm hiểu kỹ, tôi quyết định không chọn. Triết lý sư phạm của chúng tôi không phù hợp với bạn ấy. Nếu trở thành sinh viên của trường, trong ba năm học kế tiếp, bạn sẽ rất khổ sở khi bắt buộc phải hoàn tất 100% chương trình học bằng các dự án nhóm, dưới sự giám sát của giảng viên và hệ thống tài liệu hỗ trợ. Chúng tôi không đưa ra lời giải mà chỉ giúp sinh viên được tự do sáng tạo với những giải pháp họ tự đưa ra, và họ phải chịu trách nhiệm cho lời giải của chính mình. Trong khi đó, sinh viên này thừa nhận em khó hòa hợp trong một nhóm học tập, làm việc.
Kế đến, trong xã hội Việt Nam nói riêng và Á Đông nói chung, gia đình có sự đầu tư chu đáo về giáo dục thường đồng nghĩa với gia đình ít nhiều bao bọc con trẻ. Hệ quả của sự bao bọc là trẻ khi đến tuổi gần trưởng thành hoặc vẫn luôn mang tư tưởng dựa dẫm, hoặc mang mầm mống nổi loạn. Trong cả hai trường hợp, khi một người vừa bước vào tuổi trưởng thành bắt đầu ngay cuộc sống du học xa nhà sẽ dễ gặp những rắc rối nhất định: hoặc không tự giải quyết được các vấn đề cá nhân, hoặc có sự bùng nổ tự do dẫn đến đánh mất kỷ luật bản thân. Đứa trẻ ở nhà cùng cha mẹ là một đứa trẻ ngoan, nhưng khi bước ra thế giới tự chủ, sự "ngoan" có thể không duy trì được nữa.
""" # avg. chars / token 6.723456790123457 (uncase), 6.467933491686461

    text = """Một phụ huynh vừa liên hệ với tôi nhờ tìm hướng giải quyết cho con của chị . Sau vài tháng sang nước ngoài học đại học , con chị cảm thấy không hòa nhập được với môi trường mới , nằng nặc đòi về nước . Tám năm trước , tôi cũng từng chứng kiến con trai một người bạn bị nhà trường báo động vì kết quả học tập sa sút . Sau năm rưỡi du học , cậu về nhà làm sinh viên một đại học quốc tế trong nước . Dù vậy , cậu cũng mất nhiều năm chật vật mới có thể tốt nghiệp do lạc nhịp và mất phương hướng sau thời gian ở nước ngoài . Những chuyện như thế vẫn thường xảy ra trong cộng đồng du học sinh , dù các gia đình đã đầu tư tài chính , quan tâm sát sao ; phần lớn những đứa trẻ đều có ý thức và trình độ ngoại ngữ tốt . Tại sao bấy nhiêu sự đầu tư vẫn là chưa đủ ? Trước hết , trong quá trình chuẩn bị du học , du học sinh và gia đình ít khi được nghe nói về triết lý sư phạm của trường mà mình muốn đến . Triết lý sư phạm của nhà trường rất quan trọng , cho phép mỗi sinh viên nhận thấy đây có phải là cơ hội phù hợp để phát triển bản thân hay không . Tuần trước , tôi phỏng vấn một sinh viên vừa hoàn tất chương trình đại cương tại nơi khác và muốn dự tuyển vào chương trình đào tạo kỹ sư tại trường chúng tôi . Ứng viên này có bảng điểm rất ấn tượng và mục tiêu rõ ràng về công việc muốn phát triển sau khi tốt nghiệp . Đồng nghiệp - phụ trách bộ phận quan hệ giữa nhà trường , doanh nghiệp và sinh viên - tham gia buổi phỏng vấn cùng tôi , rất thích ứng viên này . Lúc đầu tôi có cùng ấn tượng tốt như vậy . Tuy nhiên , sau khi tìm hiểu kỹ , tôi quyết định không chọn . Triết lý sư phạm của chúng tôi không phù hợp với bạn ấy . Nếu trở thành sinh viên của trường , trong ba năm học kế tiếp , bạn sẽ rất khổ sở khi bắt buộc phải hoàn tất 100% chương trình học bằng các dự án nhóm , dưới sự giám sát của giảng viên và hệ thống tài liệu hỗ trợ . Chúng tôi không đưa ra lời giải mà chỉ giúp sinh viên được tự do sáng tạo với những giải pháp họ tự đưa ra , và họ phải chịu trách nhiệm cho lời giải của chính mình . Trong khi đó , sinh viên này thừa nhận em khó hòa hợp trong một nhóm học tập , làm việc . Kế đến , trong xã hội Việt Nam nói riêng và Á Đông nói chung , gia đình có sự đầu tư chu đáo về giáo dục thường đồng nghĩa với gia đình ít nhiều bao bọc con trẻ . Hệ quả của sự bao bọc là trẻ khi đến tuổi gần trưởng thành hoặc vẫn luôn mang tư tưởng dựa dẫm , hoặc mang mầm mống nổi loạn . Trong cả hai trường hợp , khi một người vừa bước vào tuổi trưởng thành bắt đầu ngay cuộc sống du học xa nhà sẽ dễ gặp những rắc rối nhất định : hoặc không tự giải quyết được các vấn đề cá nhân , hoặc có sự bùng nổ tự do dẫn đến đánh mất kỷ luật bản thân . Đứa trẻ ở nhà cùng cha mẹ là một đứa trẻ ngoan , nhưng khi bước ra thế giới tự chủ , sự " ngoan " có thể không duy trì được nữa"""

    # text = "một"

    token_ids = encode(text)
    print(token_ids, len(token_ids), end="\n\n")
    print(decode(token_ids), end="\n\n")
    print(f"Total token {len(token_ids)}, avg. chars / token {len(text) / len(token_ids)}")
