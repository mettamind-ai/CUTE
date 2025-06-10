__version__ = "0.1.0"

from .infllmv2_sparse_attn_interface  import (
    blockmask_to_uint64,
    topk_to_uint64,
    uint64_to_bool,
    infllmv2_attn_varlen_func,
    infllmv2_attn_stage1,
    infllmv2_attn_with_kvcache,
)
from .utils import generate_topk_indices
