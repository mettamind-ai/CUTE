import copy, unicodedata, functools, hashlib, itertools, regex as re
from typing import Any
from abc import ABC, abstractmethod
from script_utils import InputTokenSeq, PretokenizedT, token_array, TokenSeq, remove_unassigned_private_surrogate

class BasePretokenizer(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the BasePretokenizer with a configuration.
        :param config: A dictionary containing 'starting_token_id' and 'normalization' (default 'NFC').
        """
        self.config = copy.deepcopy(config)
        self.starting_token_id: int = config.get("starting_token_id", 1)
        self.remove_unassigned = True
        self.normalization: str = config.get("normalization", "NFC")
        if self.normalization not in ["NFC", "NFD", "NFKC", "NFKD", None]:
            raise ValueError(f"Invalid normalization form: {self.normalization}")
        self.base_tokens: dict[int, str] = {}  # Dictionary mapping token IDs to strings

    def hash(self) -> str:
        hash = hashlib.sha1()
        hash.update(str(self.starting_token_id).encode("utf-8"))
        hash.update(str(self.normalization).encode("utf-8"))
        hash.update(str(self.remove_unassigned).encode("utf-8"))
        return "Base" + hash.hexdigest()[:8]

    def encode_and_chunk(self, text: str) -> PretokenizedT:
        """normalize -> chunk -> encode"""
        return self._encode_and_chunk(self.normalize(text))

    @abstractmethod
    def _encode_and_chunk(self, text: str) -> PretokenizedT:
        """abstract method for encoding and chunking, to be implemented by subclasses"""
        pass

    @abstractmethod
    def decode(self, token_ids: InputTokenSeq, errors: str = "replace") -> str:
        """decodes a list of base tokens back to string"""
        pass

    def chunk(self, text: str) -> list[str]:
        """Returns chunks as strings, mainly for debugging"""
        return [self.decode(c) for c in self.encode_and_chunk(text)]

    def tokens_to_readable_string(self, base_token_ids: InputTokenSeq) -> str:
        """Representation that is able to handle partial/broken sequences"""
        return self.decode(base_token_ids, errors="backslashreplace")

    def normalize(self, text: str) -> str:
        """
        Normalize the input text using Unicode normalization.
        :param text: The input string to normalize.
        :return: The normalized string.
        """
        if self.remove_unassigned:
            text = remove_unassigned_private_surrogate(text)
        if self.normalization is not None:
            text = unicodedata.normalize(self.normalization, text)
        return text

    def bpe_merge_allowed(self, token_seq1: InputTokenSeq, token_seq2: InputTokenSeq) -> bool:
        """In BPE training, can token_seq1 and token_seq2 be merged?"""
        return True


TokenPairT = tuple[int, int]
ScriptEncT = tuple[int, TokenPairT]
ScriptEncNoneT = tuple[int | None, TokenPairT]


class ScriptEncodingPretokenizer(BasePretokenizer):
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the ScriptEncodingPretokenizer with a configuration.
        :param config: A dictionary containing 'starting_token_id' and 'normalization'.
        """
        super().__init__(config)

        self.script_cat_with_space = {tuple(l) for l in config["script_cat_with_space"]}
        self.enforce_char_boundaries: bool = config["enforce_char_boundaries"]
        self.blocks = config["blocks"]
        self.num_index_tokens: int = config["num_index_tokens"]
        assert len(self.blocks) == config["num_blocks"], "Number of blocks does not match the configuration"
        self.build_tokenization_maps()
        self.space_group = self.script_encode(" ")

    def hash(self) -> str:
        hash = hashlib.sha1()
        hash.update(super().hash().encode("utf-8"))
        hash.update(str(self.blocks).encode("utf-8"))
        hash.update(str(sorted(self.script_cat_with_space)).encode("utf-8"))
        return "SE-" + hash.hexdigest()[:8]

    # loading
    def build_tokenization_maps(self):
        self.script_encoding = {}
        self.base_tokens.update(
            {i + self.starting_token_id: f"<|SCRIPT_INDEX_{i}|>" for i in range(self.num_index_tokens)}
        )
        self.detokenize_map = {}
        self.script_combines_with_spaces = {
            sid: ((script, supercat) in self.script_cat_with_space) for sid, script, supercat, *_ in self.blocks
        }
        # self.inherited_script = {sid: script == "Inherited" for sid, script, *_ in self.blocks}
        self.scripts_can_merge = {
            (sid_a, sid_b)
            for sid_a, script_a, sc_a, *_ in self.blocks
            for sid_b, script_b, sc_b, *_ in self.blocks
            if script_b == "Inherited"
            or (sc_a == sc_b and (script_a == script_b or {script_a, script_b} == {"Hiragana", "Han"}))
        }
        for block_id, (sid, script, supercat, sub_block_id, cs) in enumerate(self.blocks):
            block_token_id = self.starting_token_id + self.num_index_tokens + block_id
            self.base_tokens[block_token_id] = f"<|BLOCK_{script}_{supercat}_{sub_block_id}|>"
            for ix, c in enumerate(cs):
                token_pair = (block_token_id, self.starting_token_id + ix)
                self.script_encoding[c] = (sid, token_pair)
                self.detokenize_map[token_pair] = c

    # pretokenization

    def _script_encode_char(self, c: str) -> ScriptEncNoneT:
        if c not in self.script_encoding:
            print(f"Warning: character '{c}' ({ord(c)}) not found in script encoding map.")
            return None, (-1, -1)
        else:
            return self.script_encoding[c]

    def script_encode(self, text: str) -> list[ScriptEncT]:
        """
        Encode the input text into a list of tuples containing script ID and token pair.
        :param text: The input string to encode.
        :return: A list of tuples (script_id, (token_id, sub_token_id)).
        """
        encoded = [self._script_encode_char(c) for c in text]
        return [t for t in encoded if t[0] is not None]  # type: ignore   -- filter out None values

    def chunk_script_encoding(self, encoding: list[ScriptEncT]) -> list[list[tuple[int, TokenPairT]]]:
        """
        Pretokenize the encoding by grouping adjacent tokens with the same script.
        Special cases:
        - If a script combines with spaces, it will be merged with the space group.
        - Inherited scripts are merged with the previous group, and do not split the group.
        """
        space_group = self.space_group
        script_groups = [list(g) for _, g in itertools.groupby(encoding, key=lambda x: x[0])]

        merged_groups = []
        i = 0
        while i < len(script_groups) - 1:
            current_group, next_group = script_groups[i], script_groups[i + 1]
            current_script, next_script = current_group[0][0], next_group[0][0]
            merged_group = current_group
            if self.script_combines_with_spaces[next_script] and current_group == space_group:
                merged_group = current_group + next_group
                current_script = next_script
                i += 2
            else:
                i += 1
            while i < len(script_groups) and (current_script, script_groups[i][0][0]) in self.scripts_can_merge:
                merged_group += script_groups[i]
                i += 1
            merged_groups.append(merged_group)
        if i < len(script_groups):
            merged_groups.append(script_groups[i])  # add last group
        return merged_groups

    def _encode_and_chunk(self, text: str) -> PretokenizedT:
        encoded_and_grouped = self.chunk_script_encoding(self.script_encode(text))
        # Strip script ID from final groups and concat pairs
        return [token_array([t for _, ts in group for t in ts]) for group in encoded_and_grouped]

    def decode(self, tokenization: InputTokenSeq, errors="replace") -> str:
        decoded = ""
        i = 0
        while i < len(tokenization):
            script_tok = tokenization[i]
            ix_tok = tokenization[i + 1] if i + 1 < len(tokenization) else None
            if (script_tok, ix_tok) in self.detokenize_map:
                decoded += self.detokenize_map[(script_tok, ix_tok)]
                i += 2
            else:
                if errors == "backslashreplace":  # not backslash, but compatible with bytes version
                    decoded += self.base_tokens[script_tok]
                elif errors == "replace":
                    decoded += "�"
                elif errors == "strict":
                    raise ValueError(f"Invalid tokenization: ({script_tok}, {ix_tok}) is not a valid token pair!")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        return decoded

    def is_index_token(self, i: int) -> bool:
        return self.starting_token_id <= i < self.num_index_tokens + self.starting_token_id

    def bpe_merge_allowed(self, token_seq1: InputTokenSeq, token_seq2: InputTokenSeq) -> bool:
        if not self.enforce_char_boundaries:
            return True
        if len(token_seq1) >= 2 and len(token_seq2) >= 2:
            return True  # both are full sequences
        if len(token_seq1) >= 2 or len(token_seq2) >= 2:
            return False  # one is full, the other is partial
        return not self.is_index_token(token_seq1[0])

    # utility
    @functools.cache
    def token_script_name(self, block_token_id: int) -> str:  # maps token to block name without sub-block
        return re.sub(r"<\|BLOCK_(.*)_\d+\|>", r"\1", self.base_tokens[block_token_id])

    def sequence_script_name(self, seq: list[int]) -> str | None:
        si = 0
        if self.is_index_token(seq[si]) and len(seq) > 1:
            si += 1
        if tuple(seq[si : si + 2]) == self.space_group[0][1] and len(seq) > si + 2:
            si += 2
        if self.is_index_token(seq[si]):
            return None
        return self.token_script_name(seq[si])


class ScriptEncodingPretokenizerRegexSplitting(ScriptEncodingPretokenizer):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.regex = re.compile(config["regex"], re.DOTALL)  # No default, must be provided in config

    def hash(self) -> str:
        hash = hashlib.sha1()
        hash.update(super().hash().encode("utf-8"))
        hash.update(str(self.config["regex"]).encode("utf-8"))
        return "SE+R-" + hash.hexdigest()[:8]

    def _encode_and_chunk(self, text: str) -> PretokenizedT:
        """split with regex, and then encode with script encoding"""
        chunks = re.findall(self.regex, text)
        encoded = [self.script_encode(t) for t in chunks]
        # filter empty groups - like private use area only
        return [token_array([t for _, ts in group for t in ts]) for group in encoded if group]
