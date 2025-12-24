#!/usr/bin/env python3
import sys
import unicodedata
import re

# Tone mapping (Vietnamese)
TONE_MAP = {
    "\u0301": "s",  # acute
    "\u0300": "f",  # grave
    "\u0309": "r",  # hook above
    "\u0303": "x",  # tilde
    "\u0323": "j",  # dot below
    "\u0341": "s",  # acute (combining)
    "\u0340": "f",  # grave (combining)
}

CIRCUMFLEX = {"\u0302"}  # â ê ô
BREVE = {"\u0306"}      # ă
HORN = {"\u031B"}       # ơ ư
STROKE = {"\u0335", "\u0338"}  # for đ in NFD (rare)

VOWELS = set("aeiouy")

AM_DAU_LIST = [
    "ngh", "ng", "gh", "ch", "nh", "th", "tr", "ph", "kh", "gi", "qu",
    "dd",
    "b", "c", "d", "g", "h", "l", "m", "n", "p", "r", "s", "t", "v", "x",
    "k", "q", "z",
]

AM_CUOI_LIST = ["nh", "ng", "ch", "c", "t", "p", "m", "n", "u", "o", "i", "y"]


def _push_telex(buf, chars):
    # Mimic telex_char_stream swap rules
    if not chars:
        return
    if len(chars) == 2:
        if buf and buf[-1] == "w" and chars[0] in ("o", "a"):
            buf[-1] = chars[0]
            buf.append("w")
            return
        buf.extend(chars)
        return

    ch = chars[0]
    if buf and buf[-1] == "w" and ch == "o":
        buf[-1] = "o"
        buf.append("w")
    else:
        buf.append(ch)


def utf8_to_telex_buffer(s: str):
    s = s.strip().lower()
    decomp = unicodedata.normalize("NFD", s)
    buf = []
    tone = None
    has_mark = False

    i = 0
    while i < len(decomp):
        ch = decomp[i]
        if unicodedata.combining(ch):
            # combining mark without base
            t = TONE_MAP.get(ch)
            if t and tone is None:
                tone = t
            i += 1
            continue

        # gather combining marks for this base
        j = i + 1
        marks = []
        while j < len(decomp) and unicodedata.combining(decomp[j]):
            marks.append(decomp[j])
            j += 1
        i = j

        base = ch

        # handle đ (U+0111) directly
        if base == "đ":
            _push_telex(buf, "dd")
            has_mark = True
            continue

        # If base is d with stroke mark -> đ
        if base == "d" and any(m in STROKE for m in marks):
            _push_telex(buf, "dd")
            has_mark = True
            continue

        # tone
        for m in marks:
            t = TONE_MAP.get(m)
            if t and tone is None:
                tone = t

        # vowel marks
        mark_char = None
        if any(m in CIRCUMFLEX for m in marks):
            mark_char = "z"
        elif any(m in BREVE for m in marks):
            mark_char = "w"
        elif any(m in HORN for m in marks):
            mark_char = "w"

        if mark_char is not None:
            has_mark = True
            if base in VOWELS:
                _push_telex(buf, base + mark_char)
            else:
                _push_telex(buf, base)
        else:
            _push_telex(buf, base)

    return "".join(buf), tone, has_mark


# AmGiua mapping

def _am_giua(s: str):
    if not s:
        return "_none"
    c0 = s[0]
    c1 = s[1] if len(s) > 1 else ""
    c2 = s[2] if len(s) > 2 else ""

    if c0 == "u":
        if c1 == "a":
            if c2 in ("a", "z"):
                return "uaz"
            if c2 == "w":
                return "uaw"
            return "ua"
        if c1 == "e":
            return "uez"
        if c1 == "w":
            if c2 == "a":
                return "uaw"
            if c2 == "o":
                return "uow"
            return "uw"
        if c1 == "o":
            if c2 == "o":
                return "uoz"
            if c2 == "z":
                return "uoz"
            if c2 == "w":
                return "uow"
            return "u"
        if c1 == "y":
            if c2 == "a":
                return "uya"
            if c2 == "e":
                return "uyez"
            return "uy"
        return "u"

    if c0 == "o":
        if c1 == "o":
            return "ooo"
        if c1 == "z":
            return "oz"
        if c1 == "w":
            return "ow"
        if c1 == "e":
            return "oe"
        if c1 == "a":
            if c2 == "z":
                return "uaz"
            if c2 == "w":
                return "oaw"
            return "oa"
        return "o"

    if c0 == "i":
        if c1 == "a":
            return "ia"
        if c1 == "e":
            return "iez"
        return "i"

    if c0 == "y":
        if c1 == "e":
            return "iez"
        return "y"

    if c0 == "e":
        if c1 in ("e", "z"):
            return "ez"
        return "e"

    if c0 == "a":
        if c1 in ("a", "z"):
            return "az"
        if c1 == "w":
            return "aw"
        return "a"

    return "_none"


def _am_cuoi(s: str):
    if not s:
        return "_none"
    c0 = s[0]
    c1 = s[1] if len(s) > 1 else ""
    if c0 == "n":
        if c1 == "h":
            return "nh"
        if c1 == "g":
            return "ng"
        return "n"
    if c0 == "c":
        if c1 == "h":
            return "ch"
        return "c"
    if c0 == "t":
        return "t"
    if c0 == "p":
        return "p"
    if c0 == "m":
        return "m"
    if c0 == "i":
        return "i"
    if c0 == "y":
        return "y"
    if c0 == "u":
        return "u"
    if c0 == "o":
        return "o"
    return "_none"


def _am_dau(s: str):
    # emulate Zig logic on first 1-3 chars
    if not s:
        return "_none"
    c0 = s[0]
    c1 = s[1] if len(s) > 1 else ""
    c2 = s[2] if len(s) > 2 else ""

    if c0 == "b":
        return "b"
    if c0 == "h":
        return "h"
    if c0 == "l":
        return "l"
    if c0 == "m":
        return "m"
    if c0 == "r":
        return "r"
    if c0 == "s":
        return "s"
    if c0 == "v":
        return "v"
    if c0 == "x":
        return "x"

    if c0 == "q":
        return "qu" if c1 == "u" else "_none"

    if c0 == "c":
        return "ch" if c1 == "h" else "c"

    if c0 == "d":
        return "zd" if c1 == "d" else "d"

    if c0 == "z":
        return "zd" if c1 == "d" else "_none"

    if c0 == "g":
        if c1 == "h":
            return "gh"
        if c1 == "i":
            if c2 in ("e", "y", "u", "i", "o", "a"):
                return "gi"
            return "g"
        return "g"

    if c0 == "k":
        return "kh" if c1 == "h" else "c"

    if c0 == "n":
        if c1 == "h":
            return "nh"
        if c1 == "g":
            return "ngh" if c2 == "h" else "ng"
        return "n"

    if c0 == "p":
        return "ph" if c1 == "h" else "p"

    if c0 == "t":
        if c1 == "r":
            return "tr"
        if c1 == "h":
            return "th"
        return "t"

    return "_none"


def _am_dau_len(am_dau: str) -> int:
    if am_dau == "_none":
        return 0
    if am_dau == "ngh":
        return 3
    # length in telex buffer
    if am_dau == "zd":
        return 2
    return len(am_dau)


def _am_giua_len(am_giua: str) -> int:
    if am_giua in ("a", "e", "i", "o", "u", "y"):
        return 1
    if am_giua in ("az", "aw", "ez", "oz", "ow", "uw", "oa", "oe", "ooo", "uy", "ua", "ia"):
        return 2
    if am_giua in ("iez", "oaw", "uaz", "uez", "uoz", "uaw", "uya"):
        return 3
    if am_giua in ("uow", "uyez"):
        return 4
    return 0


def _am_cuoi_len(am_cuoi: str) -> int:
    if am_cuoi == "_none":
        return 0
    if am_cuoi in ("ng", "nh", "ch"):
        return 2
    return 1


def syllable_has_mark(am_dau: str, am_giua: str) -> bool:
    if am_dau == "zd":
        return True
    return am_giua in {
        "az", "aw", "ez", "uw", "oz", "ow", "oaw", "uaz", "uez",
        "uow", "uoz", "uaw", "iez", "uyez",
    }


def token_has_mark(token: str) -> bool:
    if "đ" in token.lower():
        return True
    nfd = unicodedata.normalize("NFD", token)
    for ch in nfd:
        if unicodedata.combining(ch):
            return True
    return False


def normalize(am_dau: str, am_giua: str, am_cuoi: str):
    # normalize am_giua
    if am_giua == "ua":
        am_giua = "uoz"
    elif am_giua == "ia":
        am_giua = "iez"
    elif am_giua == "uaw":
        am_giua = "uow"
    elif am_giua == "uya":
        am_giua = "uyez"

    # normalize am_dau
    if am_dau == "gi":
        if am_giua == "ez" and am_cuoi != "_none":
            am_giua = "iez"
    elif am_dau == "ngh":
        am_dau = "ng"
    elif am_dau == "gh":
        am_dau = "g"
    else:
        if am_dau == "g" and am_giua == "i":
            am_dau = "gi"

    return am_dau, am_giua, am_cuoi


def validate_syllable(am_dau: str, am_giua: str, am_cuoi: str, tone: str | None) -> bool:
    # Mirrors validateSyllable + validation rules in syllable_parsers.zig (strict subset).
    if am_giua in ("ah", "oah"):
        if am_cuoi not in ("c", "ng"):
            return False

    # validateAmDau
    if am_dau == "gi":
        if am_giua in ("i", "y", "ia", "iez"):
            return False
    if am_dau == "c" and am_giua in ("oa", "oaw", "oe"):
        return False
    if am_dau == "qu":
        if am_giua in ("oe", "oa", "oaw"):
            return False
        if am_giua.startswith("u"):
            return False

    # validateBanAmCuoiVan
    if am_cuoi in ("o", "u", "i", "y"):
        if am_giua in ("e", "oe") and am_cuoi != "o":
            return False
        if am_giua in ("i", "ez", "iez", "uy") and am_cuoi != "u":
            return False
        if am_giua in ("o", "ow", "oz") and am_cuoi != "i":
            return False
        if am_giua in ("y", "aw", "ia", "ooo", "ua", "uez", "uaw", "uya", "uyez"):
            if am_dau == "qu" and am_giua == "y" and am_cuoi == "u":
                return True
            if am_dau == "kh" and am_giua == "uez" and am_cuoi == "u":
                return True
            return False
        if am_giua in ("u", "uoz") and am_cuoi != "i":
            return False
        if am_giua == "oa" and am_cuoi == "u":
            return False
        if am_giua in ("uw", "uow") and am_cuoi not in ("i", "u"):
            return False
        if am_giua == "uaz" and am_cuoi != "y":
            return False

    # validateNguyenAm (subset)
    if am_giua == "y" and am_dau != "qu" and am_cuoi != "_none":
        return False
    if am_giua == "ia" and am_cuoi != "_none":
        return False
    if am_giua == "uyez" or (am_dau == "qu" and am_giua == "iez"):
        if am_dau == "c":
            return False
        if am_cuoi == "_none":
            return False
        if am_cuoi in ("ng", "nh", "c", "p"):
            return False
    if am_dau == "c" and am_giua in ("uyez", "uaz", "uez", "uy"):
        return False
    if am_giua == "iez" and am_cuoi == "_none":
        return False
    if am_giua == "uya" and am_cuoi != "_none":
        return False
    if am_giua == "uyez" and am_cuoi not in ("n", "t"):
        return False
    if am_giua == "oa" and am_cuoi == "u":
        return False
    if am_giua in ("ua", "uaw") and am_cuoi != "_none":
        return False
    if am_giua == "ooo" and am_cuoi not in ("ng", "c"):
        return False
    if am_giua == "oaw":
        if am_cuoi in ("nh", "ch", "o", "u", "i", "y", "_none"):
            return False
    if am_giua == "uez":
        if am_cuoi in ("ng", "c", "o", "i", "y"):
            return False
    if am_giua == "uaz":
        if am_cuoi in ("nh", "ch", "o", "u", "i") or am_cuoi == "_none":
            return False
    if am_giua == "oe":
        if am_cuoi in ("nh", "ch", "u", "i", "y"):
            return False
    if am_giua == "uy":
        if am_cuoi in ("m", "ng", "c", "o", "i", "y"):
            return False
    if am_giua == "aw":
        if am_cuoi in ("nh", "ch", "u", "o", "i", "y") or am_cuoi == "_none":
            return False
    if am_giua == "az":
        if am_cuoi in ("nh", "ch", "o", "i") or am_cuoi == "_none":
            return False
    if am_giua == "u":
        if am_cuoi in ("nh", "ch", "o", "u"):
            return False
    if am_giua == "uw":
        if am_cuoi in ("nh", "ch", "y"):
            return False
    if am_giua == "o":
        if am_cuoi in ("nh", "ch", "u", "o", "y"):
            return False
    if am_giua == "oz":
        if am_cuoi in ("nh", "ch", "u", "o", "y"):
            return False
    if am_giua == "ow":
        if am_cuoi in ("nh", "ch", "u", "o", "y"):
            return False
    if am_giua == "iez" and am_dau == "_none" and am_cuoi == "_none":
        return False
    if am_giua == "uow" and am_cuoi == "_none":
        return False
    if am_giua == "uow":
        if am_cuoi in ("ch", "nh", "o", "y", "_none"):
            return False
    if am_giua == "uaw" and am_cuoi != "_none":
        return False
    if am_giua == "uoz":
        if am_cuoi in ("ch", "nh", "u", "o", "y", "_none"):
            return False
    if am_giua == "ua" and am_cuoi != "_none":
        return False
    if am_cuoi == "nh" and am_giua in ("e", "iez", "uoz", "uow"):
        return False
    if am_cuoi == "ng" and am_giua in ("y", "i", "ez") and am_dau != "gi":
        return False
    if am_cuoi == "ch" and am_giua in ("e", "iez"):
        return False
    if am_cuoi == "c" and am_giua in ("y", "i", "ez") and am_dau != "gi":
        return False

    # tone stop rule (c, ch, t, p)
    if am_cuoi in ("c", "ch", "t", "p") and tone not in ("s", "j"):
        return False

    return True

def parse_syllable_utf8(word: str):
    telex, tone, has_mark_input = utf8_to_telex_buffer(word)
    if not telex:
        return None

    am_dau = _am_dau(telex[:3])
    dau_len = _am_dau_len(am_dau)

    part1 = telex[dau_len:dau_len + 4]
    am_giua = _am_giua(part1)
    if am_giua == "_none":
        return None

    n = dau_len + _am_giua_len(am_giua)

    if am_giua == "uyez":
        cc = telex[dau_len + 3] if len(telex) > dau_len + 3 else ""
        if n > len(telex) or cc not in ("e", "z"):
            n -= 1
    elif am_giua in ("iez", "uez"):
        cc = telex[dau_len + 2] if len(telex) > dau_len + 2 else ""
        if n > len(telex) or cc not in ("e", "z"):
            n -= 1
    elif am_giua == "uow":
        cc = telex[dau_len + 3] if len(telex) > dau_len + 3 else ""
        if n > len(telex) or cc != "w":
            n -= 1

    part3 = telex[n:]
    if not part3:
        am_cuoi = "_none"
        if am_giua == "uow":
            am_giua = "ua"
    else:
        am_cuoi = _am_cuoi(part3)

    # Strict checks from parseTokenToGetSyllable (for UTF-8 input)
    syll_len = _am_dau_len(am_dau) + _am_giua_len(am_giua) + _am_cuoi_len(am_cuoi) + (1 if tone else 0)
    if len(telex) > syll_len:
        if not (am_giua == "ua" and telex.endswith("uow")):
            return None

    if syllable_has_mark(am_dau, am_giua) and not has_mark_input:
        if am_giua == "uyez":
            has_mark_input = True
        elif am_giua in ("iez", "uez"):
            score = (2 if tone else 0) + _am_dau_len(am_dau) + _am_cuoi_len(am_cuoi)
            if score >= 4:
                has_mark_input = True
            else:
                return None
        else:
            return None

    # Validate pre-normalized syllable
    if not validate_syllable(am_dau, am_giua, am_cuoi, tone):
        return None

    am_dau, am_giua, am_cuoi = normalize(am_dau, am_giua, am_cuoi)
    return am_dau, am_giua, am_cuoi, tone


def syllable_to_parts(am_dau, am_giua, am_cuoi, tone):
    # map dau
    if am_dau == "_none":
        dau = ""
    elif am_dau == "zd":
        dau = "dd"
    elif am_dau == "gi":
        dau = "d"
    elif am_dau == "qu":
        dau = "cu"
    else:
        dau = am_dau

    # map giua
    if am_giua == "ooo":
        giua = "oo"
    elif am_giua == "iez":
        giua = "yez"
    elif am_giua == "i":
        giua = "y"
    elif am_giua == "a":
        if am_cuoi in ("y", "o"):
            giua = "aw"
        else:
            giua = "a"
    elif am_giua == "oa":
        if am_cuoi == "y":
            giua = "oaw"
        else:
            giua = "oa"
    else:
        giua = am_giua

    # map cuoi
    if am_cuoi == "o":
        cuoi = "u"
    elif am_cuoi == "y":
        cuoi = "i"
    elif am_cuoi == "_none":
        cuoi = ""
    else:
        cuoi = am_cuoi

    parts = []

    if am_dau == "qu":
        # special handling for qu => _c oa/ue/uy...
        parts.append("_c")
        if giua:
            if len(giua) == 1 and giua in ("a", "e"):
                giua = "o" + giua
            else:
                giua = "u" + giua
            parts.append(giua)
    else:
        if dau:
            parts.append("_" + dau)
        if giua:
            parts.append(giua)

    if cuoi:
        parts.append(cuoi)
    if tone:
        parts.append(tone)

    return " ".join(parts)


def convert_line(line: str):
    # preserve hyphen as a separate token, mirror tokenizer's nonalpha tokens
    line = line.replace("-", " - ")
    clean = "".join(ch if (ch.isalpha() or ch.isspace() or ch == "-") else " " for ch in line)
    words = clean.strip().split()
    if not words:
        return ""

    tokens = []
    for w in words:
        parsed = parse_syllable_utf8(w)
        if parsed is None:
            token_str = w.lower()
            is_vi = False
        else:
            am_dau, am_giua, am_cuoi, tone = parsed
            token_str = syllable_to_parts(am_dau, am_giua, am_cuoi, tone)
            is_vi = True

        tokens.append((token_str, is_vi, token_has_mark(w)))

    line_str = " ".join(t[0] for t in tokens if t[0])
    line_bytes_len = len(line_str.encode("utf-8"))

    line_vi_tokens_len = 0
    for token_str, is_vi, has_mark in tokens:
        if not token_str:
            continue
        if is_vi or (has_mark and len(token_str) <= 20):
            line_vi_tokens_len += len(token_str.encode("utf-8")) + 1

    if line_bytes_len * 20 > line_vi_tokens_len * 100:
        return None
    if line_bytes_len * 50 >= line_vi_tokens_len * 100:
        return None

    return line_str


def _self_test():
    failures = 0

    def check_eq(name: str, got, exp):
        nonlocal failures
        ok = got == exp
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}")
        print(f"  expected: {exp!r}")
        print(f"  got     : {got!r}")
        if not ok:
            failures += 1

    # Basic conversion tests (from dict.parts.xyz and Zig tests)
    cases = [
        ("hiền nhân", "_h yez n f _nh az n"),
        ("ngó", "_ng o s"),
        ("nghiến", "_ng yez n s"),
        ("ngôn", "_ng oz n"),
        ("quân", "_c uaz n"),
        ("mưa", "_m uow"),
        ("chủ nghĩa mác-lênin", "_ch u r _ng yez x _m a c s - lênin"),
        ("rađiô catxet", "rađiô catxet"),
    ]
    for src, exp in cases:
        check_eq(f"convert_line({src!r})", convert_line(src), exp)

    # Low-Vietnamese ratio should be filtered
    check_eq("filter_low_vi('abc xyz')", convert_line("abc xyz"), None)

    # Keep hyphen as its own token (when line passes filtering)
    check_eq(
        "hyphen_token",
        convert_line("chủ nghĩa mác-lênin"),
        "_ch u r _ng yez x _m a c s - lênin",
    )

    if failures:
        raise AssertionError(f"{failures} test(s) failed")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--test":
        _self_test()
        print("tests OK")
        return

    args = sys.argv[1:]
    if not args:
        print("Usage: telexify_parts.py <input.txt> <output.xyz>")
        print("   or: telexify_parts.py \"một con vịt\"")
        print("   or: telexify_parts.py --test")
        sys.exit(1)

    if len(args) == 1:
        converted = convert_line(args[0])
        print("" if converted is None else converted)
        return

    if len(args) != 2:
        print("Usage: telexify_parts.py <input.txt> <output.xyz>")
        print("   or: telexify_parts.py \"một con vịt\"")
        print("   or: telexify_parts.py --test")
        sys.exit(1)

    inp = args[0]
    out = args[1]

    with open(inp, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                f_out.write("\n")
                continue
            converted = convert_line(line)
            if converted is None:
                continue
            f_out.write(converted + "\n")


if __name__ == "__main__":
    main()
