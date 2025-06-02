# Code from https://github.com/linkedin/Liger-Kernel
import functools
import torch, triton
import triton.language as tl
from torch import nn

try: from triton.language.extra.libdevice import tanh
except ModuleNotFoundError: from triton.language.extra.cuda.libdevice import tanh

amp_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
amp_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")
def is_hip() -> bool: return torch.version.hip is not None


@triton.jit
def element_mul_kernel(X_ptr, X_stride, value_ptr, n_cols, BLOCK_SIZE: tl.constexpr,):
    """ Nhân giá trị với ma trận theo từng hàng """
    row_i = tl.program_id(0).to(tl.int64)   # Convert program ID to int64 to avoid overflow
    X_ptr += row_i * X_stride               # Locate the start index
    value = tl.load(value_ptr)              # Load the gradient output value
    col_offsets = tl.arange(0, BLOCK_SIZE)  # Nhân theo từng BLOCK_SIZE một (tối ưu IO)

    for _ in range(tl.cdiv(n_cols, BLOCK_SIZE)):
        pointer, mask = X_ptr+col_offsets, col_offsets < n_cols
        tl.store(pointer, tl.load(pointer, mask=mask)*value, mask=mask)
        col_offsets += BLOCK_SIZE

def test_element_mul_kernel():
    # Tạo dữ liệu test
    n_rows, n_cols = 3, 5
    X = torch.ones((n_rows, n_cols), device='cuda')
    v = torch.tensor([2.0], device='cuda')  # Giá trị nhân
    element_mul_kernel[(n_rows,)](X, X.stride(0), v, n_cols, BLOCK_SIZE=2)
    expected = torch.ones_like(X) * v.item()
    assert torch.allclose(X, expected), "test_element_mul_kernel failed: Results don't match expected output"
    print("test_element_mul_kernel passed! All elements are correctly multiplied by 2.0")

if __name__ == "__main__":
    test_element_mul_kernel()


# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py
@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    # reduction: tl.constexpr,  # set it as constexpr since reduction is always known at compile time
    softcap,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    weight_ptr: Pointer to weight tensor.
    loss_ptr: Pointer to tensor to store the loss.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (float): The number of non-ignored elements in the batch.
    sum_non_ignore_weight (float): The sum of non-ignored target's weights in the batch.
    weight_sum (float): The sum of weight tensor.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    reduction (str): The string for the reduction to apply, ONLY SUPPORT "mean" for now
    softcap (float): The upper threshold for scaling logits to the range (-softcap, +softcap).
    BLOCK_SIZE (int): The block size for Triton operations.
    HAS_WEIGHT (bool): The boolean value to determine whether assigning weight to each of the classes.
    HAS_SOFTCAPPING (bool): The boolean value to determine whether applying soft-capping or not.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    if HAS_WEIGHT: weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)  # we need to store the original value of X_y for the loss calculation

    if HAS_SOFTCAPPING: ori_X_y = softcap * tanh(ori_X_y / softcap)

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)

        if HAS_SOFTCAPPING: X_block = softcap * tanh(X_block / softcap)

        block_max = tl.max(X_block)
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0))
            else:
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))

        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"), # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)

        if HAS_SOFTCAPPING:
            intermediate = tanh(X_block / softcap)
            X_block = softcap * intermediate

        if not HAS_WEIGHT:
            # softmax(x_i)
            X_block = tl.exp(X_block - m) / d

            # derivative of z-loss: 2 * lse_square_scale * lse * softmax(x_i)
            X_block += 2 * lse_square_scale * lse * X_block

            # smoothing term
            X_block += -eps

            # special handle dx_y
            X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))

            # reduction == "mean":
            X_block = X_block / n_non_ignore

        else:
            weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
            softmax_X = tl.exp(X_block - m) / d

            # derivative of original_loss
            dloss_ori = (1 - label_smoothing) * softmax_X

            # specially handle dx_y
            dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
            dloss_ori = dloss_ori * weight_y

            # derivative of smooth_loss
            dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)

            # derivative of z-loss
            dz_loss = 2 * lse_square_scale * lse * softmax_X

            # reduction == "mean":
            dloss_ori = dloss_ori / sum_non_ignore_weight
            dloss_smooth = dloss_smooth / sum_non_ignore_weight

            # TODO: Currently, z_loss is not scaled by weight.
            dz_loss = dz_loss / n_non_ignore

            # derivative of total_loss
            X_block = dloss_ori + dloss_smooth + dz_loss

        # chain rule softcapping
        # d(softcap * tanh(x / softcap)) = (1 - tanh^2(x / softcap))
        if HAS_SOFTCAPPING: X_block = X_block * (1 - intermediate * intermediate)
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
    tl.debug_barrier()

    # 5. Calculate the loss
    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    #      =  X_y - m - log d = X_y - lse
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = lse - ori_X_y
    if HAS_WEIGHT: loss = weight_y * loss

    # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (sum(-eps * x_i) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
    # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if label_smoothing > 0:
        if HAS_WEIGHT:  smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:           smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    # An auxiliary loss, z_loss
    # Refer to Page14 Loss function section in the paper PaLM: https://www.jmlr.org/papers/v24/22-1144.html
    z_loss = lse_square_scale * lse * lse

    # Normalize the loss by the number of non-ignored elements if reduction is "mean"
    # reduction == "mean"
    if HAS_WEIGHT:  loss = loss / sum_non_ignore_weight
    else:           loss = loss / n_non_ignore

    # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
    z_loss = z_loss / n_non_ignore

    loss += z_loss
    tl.store(loss_ptr, loss)


MAX_FUSED_SIZE = 65536 // 2
from optimus import quantize_int8, scaled_mm # sử dụng phép nhân INT8 Mixed

def fused_linear_cross_entropy_forward(
    _input, weight, target, ce_weight=None,
    ignore_index=-100, lse_square_scale=0.0,
    label_smoothing=0.0, softcap=None,
):
    device = _input.device
    BT, H = _input.shape
    V = weight.shape[0]

    chunk_size = 1024*8
    num_chunks = triton.cdiv(BT, chunk_size)

    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    grad_input = torch.zeros_like(_input, device=device)

    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()

    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0

    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1: ce_weight = ce_weight.contiguous()


    X,   X_row_scale = quantize_int8(_input, dim=1, sr=False) 
    wT, wT_col_scale = quantize_int8(weight.t(), dim=0, sr=True)

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]
        target_chunk = target[start_idx:end_idx]

        # logits_chunk = _input_chunk @ weight.t()
        logits_chunk = scaled_mm(X[start_idx:end_idx], wT, X_row_scale[start_idx:end_idx], wT_col_scale,)

        n_rows = logits_chunk.shape[0]
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk, X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk, Y_stride=target_chunk.stride(-1),  # always 1
            weight_ptr=ce_weight, loss_ptr=loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V, n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
            weight_sum=ce_weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            softcap=softcap,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=min(MAX_FUSED_SIZE, triton.next_power_of_2(V)),
            num_warps=32 if not is_hip() else 16,
        )

        loss_1d[start_idx:end_idx] = loss_1d_slice
        grad_logits_chunk = logits_chunk  # chunk_size x V
        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            torch.addmm(grad_weight, mat1=logits_chunk.t().to(_input_chunk.dtype), mat2=_input_chunk)

    loss, z_loss = torch.sum(loss_1d), None
    return loss, z_loss, grad_input, grad_weight


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight):
    BT, H = grad_input.shape
    BLOCK_SIZE=min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

    element_mul_kernel[(BT,)](
        grad_input, grad_input.stride(-2), grad_output, H,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=32 if not is_hip() else 16,
    )
    # handle grad_weight
    if grad_weight is not None:
        V, H = grad_weight.shape
        element_mul_kernel[(V,)](
            grad_weight, grad_weight.stride(-2), grad_output, H,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=32 if not is_hip() else 16,
        )
    return grad_input, grad_weight


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx, _input, weight, target, ce_weight=None,
        ignore_index=-100, lse_square_scale=0.0,
        label_smoothing=0.0, softcap=None,
    ):
        """
Ref https://github.com/mgmalek/efficient_cross_entropy
Xử lý forward và backward pass của linear layer cuối cùng với cross-entropy loss bằng cách 
tránh tạo ra tensor logits lớn. Vì Cross Entropy Loss là layer cuối, TA CÓ THỂ TÍNH 
GRADIENT NGAY TRONG FORWARD PASS. Nhờ đó không cần lưu _input và target cho backward pass.
        """
        loss, z_loss, grad_input, grad_weight = fused_linear_cross_entropy_forward(
            _input=_input, weight=weight, target=target, ce_weight=ce_weight,
            ignore_index=ignore_index, lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing, softcap=softcap,
        )
        # downcast to dtype and store for backward
        if grad_weight is not None: grad_weight = grad_weight.detach()
        ctx.save_for_backward(grad_input.detach(), grad_weight)
        return loss, z_loss


    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_out, grad_out2):
        del grad_out2  # z_loss is only for logging
        grad_in, grad_w = ctx.saved_tensors
        # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
        if not torch.equal(grad_out, torch.tensor(1.0, device=grad_out.device)):
            grad_in, grad_w = fused_linear_cross_entropy_backward(grad_out, grad_in, grad_w,)
        return grad_in, grad_w, None, None, None, None, None, None, None, None, None
