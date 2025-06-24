from typing import Optional

import cutlass
import cutlass.cute as cute


class SeqlenInfo:

    def __init__(
        self,
        batch_idx: cutlass.Int32,                   # Index của sample trong batch
        seqlen_q_static: cutlass.Int32,             # Độ dài query cố định (nếu có)
        seqlen_k_static: cutlass.Int32,             # Độ dài key cố định (nếu có)
        mCuSeqlensQ: Optional[cute.Tensor] = None,  # Cumulative sum của query lengths
        mCuSeqlensK: Optional[cute.Tensor] = None,  # Cumulative sum của key lengths  
        mSeqUsedQ:   Optional[cute.Tensor] = None,  # Độ dài thực tế được sử dụng của query
        mSeqUsedK:   Optional[cute.Tensor] = None,  # Độ dài thực tế được sử dụng của key
    ):
    def __init__(
        self,
        batch_idx: cutlass.Int32,
        seqlen_q_static: cutlass.Int32,
        seqlen_k_static: cutlass.Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
    ):
        self.offset_q = 0 if cutlass.const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
        self.offset_k = 0 if cutlass.const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx]
        if cutlass.const_expr(mSeqUsedQ is not None):
            self.seqlen_q = mSeqUsedQ[batch_idx]
        else:
            self.seqlen_q = seqlen_q_static if cutlass.const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx + 1] - self.offset_q
        if cutlass.const_expr(mSeqUsedK is not None):
            self.seqlen_k = mSeqUsedK[batch_idx]
        else:
            self.seqlen_k = seqlen_k_static if cutlass.const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx + 1] - self.offset_k
