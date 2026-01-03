#!/usr/bin/env python3
"""
UTF-8 to SYMATO Converter
=========================

Chuyển đổi văn bản tiếng Việt UTF-8 sang định dạng SYMATO.

SYMATO là gì?
-------------
SYMATO = SYM (âm tiết không dấu) + MARKTONE (nét phụ + thanh điệu)

Tiếng Việt có ~2535 âm tiết khác nhau (không tính dấu). Mỗi âm tiết có thể
kết hợp với 18 loại marktone (3 nét phụ × 6 thanh điệu).

Ví dụ chuyển đổi:
    "người"     -> "nguoi|wf"    (nguoi + horn + huyền)
    "Việt"      -> "^viet|zj"    (viết hoa + viet + hat + nặng)
    "đường"     -> "dduong|wf"   (đ->dd + uong + horn + huyền)

Logic chuyển đổi:
-----------------
Chỉ convert sang SYMATO nếu thỏa CẢ 3 điều kiện:

1. SYM hợp lệ: Từ phải nằm trong danh sách 2535 SYM tiếng Việt
   - "viet", "nam", "nguoi" là SYM hợp lệ
   - "hello", "world", "python" KHÔNG phải SYM

2. Pure VN chars: Từ chỉ chứa ký tự tiếng Việt (a-z + dấu VN)
   - "việt", "nam" là pure VN
   - "café", "naïve" KHÔNG phải (có ký tự lạ)

3. Word boundary hợp lệ: Đằng trước phải là:
   - Đầu văn bản
   - Khoảng trắng (space, tab, newline)
   - Dấu câu MỞ: ( [ { " ' ! ? - ...
   - KHÔNG phải dấu "dính": . @ / (thường trong URL/email/path)

Tại sao cần word boundary?
--------------------------
Tiếng Việt mỗi âm tiết cách nhau bằng space. Ví dụ:
    "com"           -> "com|"       (cơm, còm... là SYM)
    "gmail.com"     -> "gmail.com"  (giữ nguyên vì . dính trước "com")
    "vi"            -> "vi|"        (vi là SYM)
    "vi.wikipedia"  -> giữ nguyên   (trong URL)

Cách sử dụng:
-------------
    # Python
    from utf8_to_symato import text_to_symato
    text_to_symato("Việt Nam")  # -> "^viet|zj ^nam|"
    
    # Command line
    python3 utf8_to_symato.py "Việt Nam"
    python3 utf8_to_symato.py --test  # chạy tests
"""

import sys
import re
import os

# =============================================================================
# LOAD DANH SÁCH SYM
# =============================================================================
# Đọc danh sách 2535 SYM từ file symato_2816.py
# Chỉ lấy constant SYMATO_SYMS, không import cả module (tránh dependency bogo)
_symato_path = os.path.join(os.path.dirname(__file__), 'symato', 'symato_2816.py')
with open(_symato_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('SYMATO_SYMS = '):
            _syms_str = line.split('=', 1)[1].strip().strip('"')
            VALID_SYMS = set(_syms_str.split())  # Set để lookup O(1)
            break

# =============================================================================
# BẢNG NGUYÊN ÂM VÀ DẤU THANH
# =============================================================================
# Bảng 72 nguyên âm tiếng Việt (12 nguyên âm × 6 thanh điệu)
# Thứ tự trong mỗi nhóm 6: huyền, sắc, hỏi, ngã, nặng, ngang
# Ví dụ: "àáảãạa" = à(huyền), á(sắc), ả(hỏi), ã(ngã), ạ(nặng), a(ngang)
VOWELS = ("àáảãạa"    # a với 6 thanh
          "ằắẳẵặă"    # ă với 6 thanh
          "ầấẩẫậâ"    # â với 6 thanh
          "èéẻẽẹe"    # e với 6 thanh
          "ềếểễệê"    # ê với 6 thanh
          "ìíỉĩịi"    # i với 6 thanh
          "òóỏõọo"    # o với 6 thanh
          "ồốổỗộô"    # ô với 6 thanh
          "ờớởỡợơ"    # ơ với 6 thanh
          "ùúủũụu"    # u với 6 thanh
          "ừứửữựư"    # ư với 6 thanh
          "ỳýỷỹỵy")   # y với 6 thanh

# Map index thanh điệu -> mã marktone
# Index được tính từ vị trí trong VOWELS: tone = 5 - (idx % 6)
TONE_MAP = {
    0: "",   # ngang (không dấu)
    1: "j",  # nặng (dấu chấm dưới)
    2: "x",  # ngã (dấu ngã)
    3: "r",  # hỏi (dấu hỏi)
    4: "s",  # sắc (dấu sắc)
    5: "f",  # huyền (dấu huyền)
}

# Map nguyên âm có nét phụ -> (nguyên âm gốc, mã nét phụ)
# Nét phụ: "w" = horn/breve, "z" = hat (mũ)
# Lưu ý: BREVE (ă) và HORN (ơ,ư) dùng chung mã "w" vì không nhập nhằng
MARK_MAP = {
    'ă': ('a', 'w'),  # a + breve (dấu trăng)
    'â': ('a', 'z'),  # a + hat (mũ)
    'ê': ('e', 'z'),  # e + hat
    'ô': ('o', 'z'),  # o + hat
    'ơ': ('o', 'w'),  # o + horn (râu)
    'ư': ('u', 'w'),  # u + horn
}

# =============================================================================
# HÀM XỬ LÝ THANH ĐIỆU VÀ NÉT PHỤ
# =============================================================================


def get_tone(char):
    """
    Lấy thanh điệu từ một ký tự nguyên âm.
    
    Cách hoạt động:
        - Tìm vị trí ký tự trong bảng VOWELS (72 ký tự)
        - Mỗi nguyên âm có 6 biến thể thanh theo thứ tự cố định
        - Dùng phép chia lấy dư để xác định thanh
    
    Args:
        char: ký tự nguyên âm (vd: 'à', 'á', 'ả', 'ã', 'ạ', 'a')
    
    Returns:
        int: 0=ngang, 1=nặng, 2=ngã, 3=hỏi, 4=sắc, 5=huyền
    
    Ví dụ:
        get_tone('á') -> 4 (sắc)
        get_tone('à') -> 5 (huyền)
        get_tone('a') -> 0 (ngang)
    """
    idx = VOWELS.find(char.lower())
    if idx == -1:
        return 0  # không phải nguyên âm VN -> coi như ngang
    return 5 - (idx % 6)


def remove_tone(char):
    """
    Bỏ dấu thanh khỏi một ký tự, giữ nguyên nét phụ.
    
    Ví dụ:
        remove_tone('á') -> 'a'   (bỏ sắc)
        remove_tone('ắ') -> 'ă'   (bỏ sắc, giữ breve)
        remove_tone('ế') -> 'ê'   (bỏ sắc, giữ hat)
    """
    idx = VOWELS.find(char.lower())
    if idx == -1:
        return char
    # Vị trí nguyên âm không dấu thanh = cuối nhóm 6 (index 5 trong nhóm)
    base_idx = idx - (idx % 6) + 5
    base_char = VOWELS[base_idx]
    return base_char.upper() if char.isupper() else base_char


def get_mark_and_base(char):
    """
    Tách nét phụ và nguyên âm gốc từ một ký tự.
    
    Ví dụ:
        get_mark_and_base('ă') -> ('a', 'w')  # breve
        get_mark_and_base('â') -> ('a', 'z')  # hat
        get_mark_and_base('ơ') -> ('o', 'w')  # horn
        get_mark_and_base('a') -> ('a', '')   # không có nét phụ
    
    Returns:
        tuple: (nguyên_âm_gốc, mã_nét_phụ)
    """
    char_no_tone = remove_tone(char)
    lower = char_no_tone.lower()
    
    if lower in MARK_MAP:
        base, mark = MARK_MAP[lower]
        return base, mark
    return lower, ""


# =============================================================================
# HÀM CHUYỂN ĐỔI ÂM TIẾT
# =============================================================================

def syllable_to_symato(syllable):
    """
    Chuyển đổi một âm tiết tiếng Việt sang định dạng SYMATO.
    
    Quá trình xử lý:
        1. Phát hiện viết hoa (Title case, ALL CAPS)
        2. Chuyển đ -> dd
        3. Tách thanh điệu và nét phụ từ các nguyên âm
        4. Ghép thành SYM (âm tiết không dấu) + MARKTONE
    
    Args:
        syllable: âm tiết tiếng Việt (vd: "người", "Việt", "đường")
    
    Returns:
        tuple: (sym, marktone, is_upper_first, is_upper_all)
        
    Ví dụ:
        "người" -> ("nguoi", "|wf", False, False)
            - nguoi: sym (ư->u, ờ->o)
            - |wf: horn(w) + huyền(f)
            
        "Việt"  -> ("viet", "|zj", True, False)
            - viet: sym (ệ->e)
            - |zj: hat(z) + nặng(j)
            - True: viết hoa chữ đầu
            
        "VIỆT"  -> ("viet", "|zj", False, True)
            - True: viết hoa toàn bộ
    """
    if not syllable:
        return "", "|", False, False
    
    # Phát hiện viết hoa
    is_upper_all = syllable.isupper() and len(syllable) > 1
    is_upper_first = syllable[0].isupper() and not is_upper_all
    
    syllable = syllable.lower()
    
    # Xây dựng SYM
    sym = ""
    tone = 0
    mark = ""
    
    for char in syllable:
        if char == 'đ':
            # đ được encode thành dd trong SYM
            sym += 'dd'
        elif char in VOWELS or char in 'aeiouy':
            # Lấy thanh điệu (chỉ giữ thanh cuối cùng nếu có nhiều nguyên âm)
            char_tone = get_tone(char)
            if char_tone != 0:
                tone = char_tone
            
            # Lấy nét phụ và nguyên âm gốc
            base, char_mark = get_mark_and_base(char)
            if char_mark:
                mark = char_mark
            sym += base
        else:
            # Phụ âm - giữ nguyên
            sym += char
    
    # Ghép marktone: | + mark + tone
    tone_suffix = TONE_MAP.get(tone, "")
    marktone = "|" + mark + tone_suffix
    
    return sym, marktone, is_upper_first, is_upper_all


# =============================================================================
# PATTERNS VÀ CONSTANTS CHO VIỆC TÁCH TỪ
# =============================================================================

# Regex pattern cho ký tự tiếng Việt (dùng trong VN_WORD_PATTERN)
VN_CHARS = (
    r'a-zA-Z'
    r'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
    r'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ'
)
VN_WORD_PATTERN = re.compile(f'[{VN_CHARS}]+|[^{VN_CHARS}]+')

# Pattern tách theo Unicode word boundaries
# \w+ = letters, digits, underscore (bao gồm Unicode letters như é, ü, ...)
# \W+ = non-word characters
WORD_PATTERN = re.compile(r'\w+|\W+', re.UNICODE)

# Set chứa tất cả ký tự tiếng Việt hợp lệ (dùng để check pure VN word)
# Bao gồm: a-z, A-Z, và tất cả nguyên âm có dấu, đ/Đ
VN_CHARS_SET = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                   'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
                   'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ')


def is_pure_vn_word(word):
    """
    Kiểm tra xem từ chỉ chứa ký tự tiếng Việt hợp lệ.
    
    Dùng để lọc các từ có ký tự lạ như café (é không phải VN), naïve (ï).
    Các từ này sẽ được giữ nguyên, không convert sang SYMATO.
    
    Ví dụ:
        is_pure_vn_word("việt")  -> True
        is_pure_vn_word("café")  -> False (é không phải VN, là tiếng Pháp)
        is_pure_vn_word("naïve") -> False (ï không phải VN)
    """
    return all(c in VN_CHARS_SET for c in word)


def is_vietnamese_syllable(word):
    """
    Kiểm tra xem word có phải âm tiết tiếng Việt không.
    
    Hàm này KHÔNG được dùng trong logic chính của text_to_symato.
    Chỉ để tham khảo/debug. Logic chính dùng VALID_SYMS + is_pure_vn_word.
    
    Cách hoạt động:
        - Nếu có nguyên âm VN hoặc đ -> True
        - Nếu chỉ có phụ âm ASCII -> True (word.isalpha())
        - Nếu có ký tự không phải chữ -> False
    """
    word_lower = word.lower()
    for char in word_lower:
        if char in VOWELS or char == 'đ':
            return True
        if char.isalpha() and char in 'abcdefghijklmnopqrstuvwxyz':
            continue
        if not char.isalpha():
            return False
    return word.isalpha()


# =============================================================================
# HÀM CHUYỂN ĐỔI VĂN BẢN
# =============================================================================

def text_to_symato(text, keep_non_vn=True):
    """
    Chuyển đổi văn bản tiếng Việt UTF-8 sang định dạng SYMATO.
    
    Đây là hàm chính của module. Nó xử lý toàn bộ văn bản, tách thành từng
    token và quyết định convert hay giữ nguyên dựa trên 3 điều kiện.
    
    3 ĐIỀU KIỆN ĐỂ CONVERT:
    -----------------------
    1. SYM hợp lệ: Từ (sau khi bỏ dấu) phải nằm trong danh sách 2535 SYM
       - "viet", "nam" -> có trong danh sách -> convert
       - "hello", "python" -> không có -> giữ nguyên
    
    2. Pure VN chars: Từ gốc chỉ chứa ký tự tiếng Việt
       - "việt" -> OK
       - "café" -> có é (tiếng Pháp) -> giữ nguyên
    
    3. Word boundary: Đằng trước phải là ranh giới từ hợp lệ
       - Đầu văn bản, space, hoặc dấu câu mở -> OK
       - Dấu "dính" như . @ / \\ -> KHÔNG convert
       - Lý do: "gmail.com" -> "com" không nên convert dù "com" là SYM
    
    Args:
        text: văn bản UTF-8 đầu vào
        keep_non_vn: giữ nguyên các ký tự không phải tiếng Việt (mặc định True)
    
    Returns:
        str: văn bản đã chuyển đổi sang SYMATO
        
    Ví dụ:
        # Tiếng Việt thuần túy
        text_to_symato("Việt Nam")        -> "^viet|zj ^nam|"
        
        # Hỗn hợp Anh-Việt
        text_to_symato("I love Việt Nam") -> "^i| love ^viet|zj ^nam|"
        # "i" là SYM, "love" không phải
        
        # URL/email - giữ nguyên
        text_to_symato("email@gmail.com") -> "email@gmail.com"
        # Vì @ và . là dấu "dính"
        
        # Có dấu câu - convert bình thường
        text_to_symato("(Việt)")          -> "(^viet|zj)"
        # ( là word boundary hợp lệ
    """
    # Tách văn bản thành tokens theo Unicode word boundaries
    # Ví dụ: "Việt Nam!" -> ["Việt", " ", "Nam", "!"]
    tokens = WORD_PATTERN.findall(text)
    
    result = []
    prev_is_word_boundary = True  # Đầu text coi như word boundary
    
    for token in tokens:
        if not token:
            continue
            
        # -----------------------------------------------------------------
        # XỬ LÝ TOKEN KHÔNG PHẢI CHỮ (dấu câu, số, khoảng trắng, ...)
        # -----------------------------------------------------------------
        if not token[0].isalpha() and token[0] != 'đ' and token[0] != 'Đ':
            if keep_non_vn:
                # Normalize nhiều khoảng trắng thành 1 space
                if token.isspace():
                    if not result or not result[-1].isspace():
                        result.append(" ")
                else:
                    result.append(token)
                
                # Xác định đây có phải word boundary hợp lệ không
                # Dấu "dính" (. @ / \) KHÔNG tạo word boundary
                # -> Token sau dấu này sẽ không được convert
                prev_is_word_boundary = not any(c in '.@/\\' for c in token)
            continue
        
        # -----------------------------------------------------------------
        # XỬ LÝ TOKEN LÀ CHỮ (có thể là âm tiết tiếng Việt)
        # -----------------------------------------------------------------
        sym, marktone, is_upper_first, is_upper_all = syllable_to_symato(token)
        
        if sym:
            # Kiểm tra 3 điều kiện
            is_valid_sym = (sym in VALID_SYMS and           # 1. SYM hợp lệ
                           is_pure_vn_word(token) and       # 2. Pure VN chars
                           prev_is_word_boundary)           # 3. Word boundary OK
            
            if is_valid_sym:
                # Convert sang SYMATO
                prefix = ""
                if is_upper_all:
                    prefix = "^^"  # VIỆT -> ^^viet|zj
                elif is_upper_first:
                    prefix = "^"   # Việt -> ^viet|zj
                result.append(f"{prefix}{sym}{marktone}")
            else:
                # Không thỏa điều kiện -> giữ nguyên token gốc
                result.append(token)
        
        # Token chữ cái không tạo word boundary cho token tiếp theo
        prev_is_word_boundary = False
    
    return "".join(result)


def text_to_symato_tokens(text):
    """
    Convert văn bản sang danh sách các SYMATO tokens.
    
    Khác với text_to_symato():
        - Trả về list thay vì string
        - Mỗi token là tuple chứa thông tin chi tiết
        - Không check word boundary (convert tất cả)
        - Dùng để xử lý từng token riêng biệt
    
    Args:
        text: văn bản UTF-8 đầu vào
    
    Returns:
        list: mỗi phần tử là:
            - (sym, marktone, cap_type) cho âm tiết VN
            - (non_vn_text,) cho text không phải VN (tuple 1 phần tử)
        
    cap_type:
        - 0: lowercase (việt)
        - 1: capitalize first (Việt)
        - 2: all caps (VIỆT)
    
    Ví dụ:
        text_to_symato_tokens("Việt Nam!")
        -> [('viet', '|zj', 1), (' ',), ('nam', '|', 1), ('!',)]
        
        text_to_symato_tokens("VIỆT")
        -> [('viet', '|zj', 2)]  # cap_type=2 = all caps
    """
    tokens = VN_WORD_PATTERN.findall(text)
    
    result = []
    for token in tokens:
        if not token:
            continue
            
        if not token[0].isalpha() and token[0].lower() != 'đ':
            result.append((token,))  # non-Vietnamese
            continue
        
        sym, marktone, is_upper_first, is_upper_all = syllable_to_symato(token)
        
        if sym:
            cap_type = 2 if is_upper_all else (1 if is_upper_first else 0)
            result.append((sym, marktone, cap_type))
    
    return result


def symato_to_telex(sym, marktone):
    """
    Convert SYMATO (sym + marktone) thành chuỗi Telex.
    
    Telex là gì?
    ------------
    Telex là phương pháp gõ tiếng Việt phổ biến nhất, dùng các ký tự ASCII
    để biểu diễn dấu thanh và nét phụ. Ví dụ:
        - "vietj" -> "việt" (j = nặng)
        - "nguoiwf" -> "người" (w = horn, f = huyền)
    
    Bogo là thư viện Python convert Telex -> UTF-8.
    Hàm này giúp chuyển SYMATO -> Telex để có thể dùng Bogo convert về UTF-8.
    
    Cách hoạt động:
        - Bỏ dấu | ở đầu marktone
        - Ghép sym + mark_code + tone_code
    
    Args:
        sym: âm tiết không dấu (vd: "nguoi", "viet")
        marktone: mã marktone (vd: "|wf", "|zj")
    
    Returns:
        str: chuỗi Telex (vd: "nguoiwf", "vietzj")
    
    Ví dụ:
        symato_to_telex("nguoi", "|wf") -> "nguoiwf"
        symato_to_telex("viet", "|zj")  -> "vietzj"
        symato_to_telex("ma", "|")      -> "ma"
        symato_to_telex("ma", "|s")     -> "mas"
    
    Lưu ý:
        Để convert về UTF-8, cần thêm bước dùng Bogo:
            from bogo import process_sequence
            utf8 = process_sequence(symato_to_telex(sym, marktone))
    """
    # Parse marktone: |[w|z][s|f|r|x|j]
    if not marktone.startswith("|"):
        return sym
    
    mt = marktone[1:]  # bỏ dấu |
    
    # Tách mark và tone
    mark_code = ""
    tone_code = ""
    
    for c in mt:
        if c in "wz":
            mark_code = c
        elif c in "sfrxj":
            tone_code = c
    
    # Build telex string
    return sym + mark_code + tone_code


def _test():
    """
    Inline tests cho module utf8_to_symato.
    
    Chạy bằng: python3 utf8_to_symato.py --test
    
    Test coverage:
        - get_tone(): 6 thanh điệu + edge cases
        - remove_tone(): bỏ dấu thanh, giữ nét phụ
        - get_mark_and_base(): tách nét phụ (breve, hat, horn)
        - syllable_to_symato(): tones, marks, đ->dd, capitalization
        - text_to_symato(): 3 điều kiện convert, URL/email, dấu câu
        - text_to_symato_tokens(): token list format
        - symato_to_telex(): SYMATO -> Telex conversion
        - SYM vs non-SYM: từ Anh vs từ Việt
        - Edge cases: empty, whitespace, Unicode lạ
    """
    
    # === get_tone ===
    assert get_tone('a') == 0   # ngang
    assert get_tone('á') == 4   # sắc
    assert get_tone('à') == 5   # huyền
    assert get_tone('ả') == 3   # hỏi
    assert get_tone('ã') == 2   # ngã
    assert get_tone('ạ') == 1   # nặng
    assert get_tone('ứ') == 4   # sắc với ư
    assert get_tone('b') == 0   # consonant
    
    # === remove_tone ===
    assert remove_tone('á') == 'a'
    assert remove_tone('ế') == 'ê'
    assert remove_tone('ứ') == 'ư'
    assert remove_tone('Á') == 'A'
    assert remove_tone('a') == 'a'
    
    # === get_mark_and_base ===
    assert get_mark_and_base('ă') == ('a', 'w')  # breve
    assert get_mark_and_base('â') == ('a', 'z')  # hat
    assert get_mark_and_base('ê') == ('e', 'z')  # hat
    assert get_mark_and_base('ô') == ('o', 'z')  # hat
    assert get_mark_and_base('ơ') == ('o', 'w')  # horn
    assert get_mark_and_base('ư') == ('u', 'w')  # horn
    assert get_mark_and_base('a') == ('a', '')   # no mark
    
    # === syllable_to_symato: tones ===
    assert syllable_to_symato('ma')[:2] == ('ma', '|')
    assert syllable_to_symato('má')[:2] == ('ma', '|s')
    assert syllable_to_symato('mà')[:2] == ('ma', '|f')
    assert syllable_to_symato('mả')[:2] == ('ma', '|r')
    assert syllable_to_symato('mã')[:2] == ('ma', '|x')
    assert syllable_to_symato('mạ')[:2] == ('ma', '|j')
    
    # === syllable_to_symato: marks ===
    assert syllable_to_symato('ăn')[:2] == ('an', '|w')   # breve
    assert syllable_to_symato('ân')[:2] == ('an', '|z')   # hat
    assert syllable_to_symato('ơn')[:2] == ('on', '|w')   # horn
    assert syllable_to_symato('ưng')[:2] == ('ung', '|w') # horn
    
    # === syllable_to_symato: mark + tone ===
    assert syllable_to_symato('ắn')[:2] == ('an', '|ws')  # breve + sắc
    assert syllable_to_symato('ầm')[:2] == ('am', '|zf')  # hat + huyền
    assert syllable_to_symato('ợt')[:2] == ('ot', '|wj')  # horn + nặng
    
    # === đ handling ===
    assert syllable_to_symato('di')[:2] == ('di', '|')
    assert syllable_to_symato('đi')[:2] == ('ddi', '|')
    assert syllable_to_symato('đường')[:2] == ('dduong', '|wf')
    
    # === capitalization ===
    sym, mt, up1, upall = syllable_to_symato('Việt')
    assert (sym, mt, up1, upall) == ('viet', '|zj', True, False)
    sym, mt, up1, upall = syllable_to_symato('VIỆT')
    assert (sym, mt, up1, upall) == ('viet', '|zj', False, True)
    sym, mt, up1, upall = syllable_to_symato('việt')
    assert (sym, mt, up1, upall) == ('viet', '|zj', False, False)
    
    # === complex syllables ===
    assert syllable_to_symato('người')[:2] == ('nguoi', '|wf')
    assert syllable_to_symato('nước')[:2] == ('nuoc', '|ws')
    assert syllable_to_symato('nghiêng')[:2] == ('nghieng', '|z')
    assert syllable_to_symato('trường')[:2] == ('truong', '|wf')
    assert syllable_to_symato('')[:2] == ('', '|')
    
    # === text_to_symato ===
    assert text_to_symato('Việt Nam') == '^viet|zj ^nam|'
    assert text_to_symato('Đường đi') == '^dduong|wf ddi|'
    assert text_to_symato('VIỆT NAM vô địch') == '^^viet|zj ^^nam| vo|z ddich|j'
    assert text_to_symato('Xin chào!') == '^xin| chao|f!'
    assert text_to_symato('năm 2024') == 'nam|w 2024'
    
    # === horn vs hat ===
    assert text_to_symato('nước') == 'nuoc|ws'  # ươ = horn
    assert text_to_symato('buổi') == 'buoi|zr'  # uô = hat
    assert text_to_symato('đường') == 'dduong|wf'
    assert text_to_symato('đuôi') == 'dduoi|z'
    
    # === edge cases ===
    assert text_to_symato('') == ''
    assert text_to_symato('   ') == ' '
    assert text_to_symato('12345') == '12345'
    assert text_to_symato('!@#$%') == '!@#$%'
    assert text_to_symato('Việt-Nam') == '^viet|zj-^nam|'
    assert text_to_symato('(Việt)') == '(^viet|zj)'
    assert text_to_symato('hello world') == 'hello world'  # không phải SYM VN
    
    # === common words ===
    assert text_to_symato('và') == 'va|f'
    assert text_to_symato('của') == 'cua|r'  # ủ = u + hỏi, no mark
    assert text_to_symato('được') == 'dduoc|wj'  # ượ = horn + nặng
    assert text_to_symato('những') == 'nhung|wx'
    assert text_to_symato('không') == 'khong|z'
    assert text_to_symato('để') == 'dde|zr'
    
    # === all 18 marktones ===
    # no mark: 6 tones
    for syl, mt in [('ma','|'),('má','|s'),('mà','|f'),('mả','|r'),('mã','|x'),('mạ','|j')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    # breve: 6 tones
    for syl, mt in [('ăn','|w'),('ắn','|ws'),('ằn','|wf'),('ẳn','|wr'),('ẵn','|wx'),('ặn','|wj')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    # hat: 6 tones
    for syl, mt in [('ân','|z'),('ấn','|zs'),('ần','|zf'),('ẩn','|zr'),('ẫn','|zx'),('ận','|zj')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    
    # === symato_to_telex ===
    assert symato_to_telex('nguoi', '|wf') == 'nguoiwf'
    assert symato_to_telex('viet', '|zj') == 'vietzj'
    assert symato_to_telex('ma', '|') == 'ma'
    assert symato_to_telex('ma', '|s') == 'mas'
    
    # === uppercase đ ===
    assert text_to_symato('Đi') == '^ddi|'
    assert text_to_symato('ĐI') == '^^ddi|'
    
    # === y vowel ===
    assert syllable_to_symato('ý')[:2] == ('y', '|s')
    assert syllable_to_symato('kỳ')[:2] == ('ky', '|f')
    
    # === more edge cases ===
    # single characters
    assert syllable_to_symato('a')[:2] == ('a', '|')
    assert syllable_to_symato('b')[:2] == ('b', '|')
    assert syllable_to_symato('đ')[:2] == ('dd', '|')
    
    # all y tones
    assert syllable_to_symato('y')[:2] == ('y', '|')
    assert syllable_to_symato('ỳ')[:2] == ('y', '|f')
    assert syllable_to_symato('ỷ')[:2] == ('y', '|r')
    assert syllable_to_symato('ỹ')[:2] == ('y', '|x')
    assert syllable_to_symato('ỵ')[:2] == ('y', '|j')
    
    # horn vowels with all tones (ơ)
    for syl, mt in [('ơn','|w'),('ớn','|ws'),('ờn','|wf'),('ởn','|wr'),('ỡn','|wx'),('ợn','|wj')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    # horn vowels with all tones (ư)
    for syl, mt in [('ưng','|w'),('ứng','|ws'),('ừng','|wf'),('ửng','|wr'),('ững','|wx'),('ựng','|wj')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    
    # ê hat with all tones
    for syl, mt in [('ên','|z'),('ến','|zs'),('ền','|zf'),('ển','|zr'),('ễn','|zx'),('ện','|zj')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    # ô hat with all tones
    for syl, mt in [('ôn','|z'),('ốn','|zs'),('ồn','|zf'),('ổn','|zr'),('ỗn','|zx'),('ộn','|zj')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    
    # gi, qu special cases
    assert syllable_to_symato('gi')[:2] == ('gi', '|')
    assert syllable_to_symato('gì')[:2] == ('gi', '|f')
    assert syllable_to_symato('già')[:2] == ('gia', '|f')
    assert syllable_to_symato('giá')[:2] == ('gia', '|s')
    assert syllable_to_symato('giấy')[:2] == ('giay', '|zs')
    assert syllable_to_symato('qua')[:2] == ('qua', '|')
    assert syllable_to_symato('quà')[:2] == ('qua', '|f')
    assert syllable_to_symato('quê')[:2] == ('que', '|z')
    assert syllable_to_symato('quế')[:2] == ('que', '|zs')
    assert syllable_to_symato('quyết')[:2] == ('quyet', '|zs')  # ế = hat + sắc
    
    # multiple diacritics in one syllable (ươ, uô, iê)
    assert syllable_to_symato('ươi')[:2] == ('uoi', '|w')
    assert syllable_to_symato('ười')[:2] == ('uoi', '|wf')
    assert syllable_to_symato('uôi')[:2] == ('uoi', '|z')
    assert syllable_to_symato('uổi')[:2] == ('uoi', '|zr')
    assert syllable_to_symato('iêu')[:2] == ('ieu', '|z')
    assert syllable_to_symato('iếu')[:2] == ('ieu', '|zs')
    
    # long consonant clusters
    assert syllable_to_symato('ngh')[:2] == ('ngh', '|')
    assert syllable_to_symato('nghĩ')[:2] == ('nghi', '|x')
    assert syllable_to_symato('nghiêng')[:2] == ('nghieng', '|z')
    assert syllable_to_symato('nghiêm')[:2] == ('nghiem', '|z')
    assert syllable_to_symato('khuya')[:2] == ('khuya', '|')
    assert syllable_to_symato('khuyến')[:2] == ('khuyen', '|zs')
    assert syllable_to_symato('trường')[:2] == ('truong', '|wf')
    assert syllable_to_symato('nguyễn')[:2] == ('nguyen', '|zx')
    
    # text with tabs and newlines (all whitespace normalized to single space)
    # "a" là SYM VN, "b" không phải
    assert text_to_symato('a\tb') == 'a| b'
    assert text_to_symato('a\nb') == 'a| b'
    assert text_to_symato('a\r\nb') == 'a| b'
    
    # mixed Vietnamese and English (chỉ SYM VN có |)
    assert text_to_symato('I love Việt Nam!') == '^i| love ^viet|zj ^nam|!'  # i là SYM VN
    assert text_to_symato('Python 3.12') == 'Python 3.12'  # không phải SYM VN
    assert text_to_symato('email@gmail.com') == 'email@gmail.com'  # không convert vì @ dính
    
    # URL-like text (không convert vì . / dính)
    assert text_to_symato('https://vi.wikipedia.org') == 'https://vi.wikipedia.org'
    
    # quotes and brackets
    assert text_to_symato('"Việt Nam"') == '"^viet|zj ^nam|"'
    assert text_to_symato("'Việt'") == "'^viet|zj'"
    assert text_to_symato('[Việt]') == '[^viet|zj]'
    assert text_to_symato('{Việt}') == '{^viet|zj}'
    
    # consecutive punctuation
    assert text_to_symato('Việt!!!') == '^viet|zj!!!'
    assert text_to_symato('Việt...') == '^viet|zj...'
    assert text_to_symato('Việt?!') == '^viet|zj?!'
    
    # capitalization edge cases
    sym, mt, up1, upall = syllable_to_symato('A')
    assert (up1, upall) == (True, False)  # single uppercase = capitalize first
    sym, mt, up1, upall = syllable_to_symato('AB')
    assert (up1, upall) == (False, True)  # two uppercase = all caps
    sym, mt, up1, upall = syllable_to_symato('Ab')
    assert (up1, upall) == (True, False)  # title case
    sym, mt, up1, upall = syllable_to_symato('aB')
    assert (up1, upall) == (False, False)  # weird case = lowercase
    
    # text_to_symato_tokens
    tokens = text_to_symato_tokens('Việt Nam!')
    assert tokens[0] == ('viet', '|zj', 1)  # Việt, cap_type=1
    assert tokens[1] == (' ',)  # space
    assert tokens[2] == ('nam', '|', 1)  # Nam, cap_type=1
    assert tokens[3] == ('!',)  # punctuation
    
    tokens = text_to_symato_tokens('VIỆT')
    assert tokens[0] == ('viet', '|zj', 2)  # cap_type=2 = all caps
    
    # symato_to_telex edge cases
    assert symato_to_telex('nguoi', '|w') == 'nguoiw'  # mark only
    assert symato_to_telex('ma', '|j') == 'maj'  # tone only
    assert symato_to_telex('an', '|wj') == 'anwj'  # mark + tone
    assert symato_to_telex('test', 'invalid') == 'test'  # no | prefix
    
    # all i tones
    for syl, mt in [('in','|'),('ín','|s'),('ìn','|f'),('ỉn','|r'),('ĩn','|x'),('ịn','|j')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    
    # all o tones (plain o, no mark)
    for syl, mt in [('on','|'),('ón','|s'),('òn','|f'),('ỏn','|r'),('õn','|x'),('ọn','|j')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    
    # all e tones (plain e, no mark)
    for syl, mt in [('en','|'),('én','|s'),('èn','|f'),('ẻn','|r'),('ẽn','|x'),('ẹn','|j')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    
    # all u tones (plain u, no mark)
    for syl, mt in [('un','|'),('ún','|s'),('ùn','|f'),('ủn','|r'),('ũn','|x'),('ụn','|j')]:
        assert syllable_to_symato(syl)[1] == mt, f"Failed: {syl}"
    
    # === SYM vs non-SYM detection ===
    # Các từ tiếng Anh phổ biến KHÔNG phải SYM
    assert text_to_symato('the') == 'the|'  # "the" là SYM VN (thể, thè, ...)
    assert text_to_symato('and') == 'and'
    assert text_to_symato('for') == 'for'
    assert text_to_symato('with') == 'with'
    assert text_to_symato('from') == 'from'
    assert text_to_symato('this') == 'this'
    assert text_to_symato('that') == 'that|'  # "that" là SYM VN
    assert text_to_symato('have') == 'have'
    assert text_to_symato('will') == 'will'
    assert text_to_symato('your') == 'your'
    assert text_to_symato('what') == 'what'
    assert text_to_symato('when') == 'when'
    assert text_to_symato('where') == 'where'
    assert text_to_symato('which') == 'which'
    assert text_to_symato('their') == 'their'
    assert text_to_symato('about') == 'about'
    
    # Các SYM tiếng Việt phổ biến CÓ |
    assert text_to_symato('a') == 'a|'
    assert text_to_symato('an') == 'an|'
    assert text_to_symato('ba') == 'ba|'
    assert text_to_symato('ca') == 'ca|'
    assert text_to_symato('da') == 'da|'
    assert text_to_symato('di') == 'di|'
    assert text_to_symato('em') == 'em|'
    assert text_to_symato('ga') == 'ga|'
    assert text_to_symato('ha') == 'ha|'
    assert text_to_symato('la') == 'la|'
    assert text_to_symato('ma') == 'ma|'
    assert text_to_symato('me') == 'me|'
    assert text_to_symato('na') == 'na|'
    assert text_to_symato('no') == 'no|'  # "no" là SYM VN
    assert text_to_symato('ta') == 'ta|'
    assert text_to_symato('to') == 'to|'  # "to" là SYM VN
    
    # Từ có thể là cả tiếng Anh và tiếng Việt (SYM quyết định)
    assert text_to_symato('be') == 'be|'  # be là SYM VN
    assert text_to_symato('can') == 'can|'  # can là SYM VN (cạn, cán, ...)
    assert text_to_symato('chi') == 'chi|'  # chi là SYM VN
    assert text_to_symato('co') == 'co|'  # co là SYM VN
    assert text_to_symato('con') == 'con|'  # con là SYM VN
    assert text_to_symato('go') == 'go|'  # go là SYM VN (gò, gọ, ...)
    assert text_to_symato('hay') == 'hay|'  # hay là SYM VN
    assert text_to_symato('hi') == 'hi|'  # hi là SYM VN
    assert text_to_symato('ho') == 'ho|'  # ho là SYM VN
    assert text_to_symato('in') == 'in|'  # in là SYM VN
    assert text_to_symato('it') == 'it|'  # it là SYM VN (ít, ịt...)
    assert text_to_symato('my') == 'my|'  # my là SYM VN
    assert text_to_symato('on') == 'on|'  # on là SYM VN
    assert text_to_symato('so') == 'so|'  # so là SYM VN
    
    # Một số từ tiếng Anh TRÙNG là SYM VN
    # (đây là hành vi mong đợi - nếu trong danh sách SYM thì có |)
    assert 'the' in VALID_SYMS  # "the" là SYM VN!
    assert 'and' not in VALID_SYMS
    assert 'be' in VALID_SYMS
    assert 'can' in VALID_SYMS
    assert 'go' in VALID_SYMS
    
    # === Câu hỗn hợp Anh-Việt ===
    assert text_to_symato('Hello Việt Nam') == 'Hello ^viet|zj ^nam|'
    assert text_to_symato('Welcome to Hà Nội') == 'Welcome to| ^ha|f ^noi|zj'
    assert text_to_symato('This is Sài Gòn') == 'This is ^sai|f ^gon|f'  # "is" không phải SYM
    
    # === Tên riêng ===
    assert text_to_symato('Nguyễn Văn An') == '^nguyen|zx ^van|w ^an|'
    assert text_to_symato('Trần Thị Bình') == '^tran|zf ^thi|j ^binh|f'  # Trần = |zf
    assert text_to_symato('Lê Hoàng Long') == '^le|z ^hoang|f ^long|'
    
    # === Số và ký tự đặc biệt ===
    assert text_to_symato('Năm 2024') == '^nam|w 2024'
    assert text_to_symato('100% Việt') == '100% ^viet|zj'
    assert text_to_symato('$100 USD') == '$100 USD'
    assert text_to_symato('#hashtag') == '#hashtag'
    assert text_to_symato('@username') == '@username'
    
    # === Unicode edge cases ===
    assert text_to_symato('café') == 'café'  # tiếng Pháp, không phải SYM
    assert text_to_symato('naïve') == 'naïve'  # giữ nguyên vì có ký tự lạ
    assert text_to_symato('über') == 'über'  # tiếng Đức
    
    # === Empty và whitespace ===
    assert text_to_symato('') == ''
    assert text_to_symato('   ') == ' '
    assert text_to_symato('\t\n\r') == ' '
    
    # === Dấu câu phức tạp ===
    assert text_to_symato('Việt Nam!!! Tuyệt vời!!!') == '^viet|zj ^nam|!!! ^tuyet|zj voi|wf!!!'
    assert text_to_symato('Hỏi: "Bạn khỏe không?"') == '^hoi|r: "^ban|j khoe|r khong|z?"'
    assert text_to_symato('(Ghi chú: xem thêm)') == '(^ghi| chu|s: xem| them|z)'
    
    # === Đường dẫn và code ===
    assert text_to_symato('/home/user/file.txt') == '/home/user/file.txt'
    assert text_to_symato('function_name()') == 'function_name()'
    assert text_to_symato('var_1 = 10') == 'var_1 = 10'
    
    # === Tất cả các mark với text_to_symato ===
    # Breve (ă)
    assert text_to_symato('ăn') == 'an|w'
    assert text_to_symato('lăn') == 'lan|w'
    assert text_to_symato('tắt') == 'tat|ws'
    
    # Hat (â, ê, ô)
    assert text_to_symato('ân') == 'an|z'
    assert text_to_symato('tân') == 'tan|z'
    assert text_to_symato('lên') == 'len|z'  # ê = hat, không có tone
    assert text_to_symato('tôn') == 'ton|z'
    
    # Horn (ơ, ư)
    assert text_to_symato('ơn') == 'on|w'
    assert text_to_symato('sơn') == 'son|w'
    assert text_to_symato('tưng') == 'tung|w'
    assert text_to_symato('hưng') == 'hung|w'
    
    # === dd handling in text ===
    assert text_to_symato('đi đâu đó') == 'ddi| ddau|z ddo|s'
    assert text_to_symato('Đông Đô') == '^ddong|z ^ddo|z'
    assert text_to_symato('ĐẠI VIỆT') == '^^ddai|j ^^viet|zj'
    
    print("All tests passed!")


# =============================================================================
# ENTRY POINT
# =============================================================================
# Cách sử dụng command line:
#   python3 utf8_to_symato.py --test           # Chạy tất cả tests
#   python3 utf8_to_symato.py "Việt Nam"       # Convert từ argument
#   echo "Việt Nam" | python3 utf8_to_symato.py  # Convert từ stdin

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _test()
    elif len(sys.argv) > 1:
        # Convert văn bản từ command line arguments
        text = " ".join(sys.argv[1:])
        print(text_to_symato(text))
    else:
        # Convert văn bản từ stdin (pipe hoặc redirect)
        text = sys.stdin.read()
        print(text_to_symato(text))
