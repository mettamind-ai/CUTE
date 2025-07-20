import array, functools, logging, multiprocessing, os, sys, time
from typing import Iterable

SCRIPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unicode_scripts.txt")

@functools.cache
def supercategory(category):
    sc = category[0]
    if sc in {"P", "S"}: return "PS"  # Punctuation/Symbol
    if sc in {"L", "M"}: return "LM"  # Letter/Non-spacing Mark (like accept modifiers)
    return sc

@functools.cache
def unicode_script_map(filename=SCRIPTS_PATH) -> dict[str, dict[str, str]]:
    """ Load Unicode script and category data from a file.
    Returns: A dictionary mapping codepoint (int) to a dict with 'script' and 'category' keys
    """
    char_info: dict[str, dict[str, str]] = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse 0000..001F    ; Common # Cc  [32] <control-0000>..<control-001F>
            range_str, semicol, script, hash, category, *_ = line.split()
            assert semicol == ";" and hash == "#", f"Unexpected format in line: {line}"
            # Handle single codepoint or range
            if ".." in range_str:
                    start_str, end_str = range_str.split("..")
                    start, end = int(start_str, 16), int(end_str, 16)
            else:   start = end = int(range_str, 16)

            # Add each codepoint in the range to the result dictionary
            for cp in range(start, end + 1): char_info[chr(cp)] = dict(script=script, category=category)
    for entry in char_info.values(): entry["supercategory"] = supercategory(entry["category"])
    return char_info


# one dir lower than this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- multiprocessing context ----
mp_ctx = multiprocessing.get_context("forkserver")

# ---- typing ----
# Internal/output types
TokenSeq = array.array  # [int]
PretokenizedT = list[TokenSeq]

# inputs more flexible
InputTokenSeq = array.array | list[int]

def token_array(values: Iterable[int]) -> TokenSeq:
    return array.array("i", values)

# ---- logging ----
def create_logger(tag: str, verbose: bool = True):
    default_fields = logging.getLogRecordFactory()
    t0 = time.perf_counter()

    # https://stackoverflow.com/questions/63056270/python-logging-time-since-start-in-seconds
    def record_factory(*args, **kwargs):
        record = default_fields(*args, **kwargs)
        record.uptime = time.perf_counter() - t0
        record.level_nocaps = record.levelname.lower()
        return record

    logging.setLogRecordFactory(record_factory)
    logger = logging.getLogger(tag)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter(f"[%(uptime)6.1fs][{tag}] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# --- string/utf8 utils ---
UNASSIGNED_CATEGORIES = {"Cn", "Co", "Cs"}  # we ignore Cn=Not Assigned, Co=Private Use, Cs=Surrogate

def remove_unassigned_private_surrogate(s):
    return "".join(c for c in s if not is_unassigned_private_surrogate(c))

@functools.cache
def is_unassigned_private_surrogate(char):
    return char not in unicode_script_map()

@functools.cache
def utf_byte_type(b: int) -> int:
    start_byte = f"{b:08b}"  # cached so we can be really explicit
    if start_byte.startswith("0"): return 1
    if start_byte.startswith("10"): return 0  # continuation byte
    if start_byte.startswith("110"): return 2
    if start_byte.startswith("1110"): return 3
    if start_byte.startswith("11110"): return 4
    return 5  # not part of utf8
