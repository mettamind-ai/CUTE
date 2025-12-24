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
            continue

        # If base is d with stroke mark -> đ
        if base == "d" and any(m in STROKE for m in marks):
            _push_telex(buf, "dd")
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
            if base in VOWELS:
                _push_telex(buf, base + mark_char)
            else:
                _push_telex(buf, base)
        else:
            _push_telex(buf, base)

    return "".join(buf), tone


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


def parse_syllable_utf8(word: str):
    telex, tone = utf8_to_telex_buffer(word)
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
    # replace non-letters with space
    clean = "".join(ch if ch.isalpha() or ch.isspace() else " " for ch in line)
    words = clean.strip().split()
    out_parts = []
    for w in words:
        parsed = parse_syllable_utf8(w)
        if parsed is None:
            # fallback: keep raw lowercase word
            out_parts.append(w.lower())
            continue
        am_dau, am_giua, am_cuoi, tone = parsed
        out_parts.append(syllable_to_parts(am_dau, am_giua, am_cuoi, tone))
    return " ".join(p for p in out_parts if p)


def main():
    if len(sys.argv) < 3:
        print("Usage: py_telexify_parts.py <input.txt> <output.xyz>")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2]

    with open(inp, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                f_out.write("\n")
                continue
            f_out.write(convert_line(line) + "\n")


if __name__ == "__main__":
    main()
