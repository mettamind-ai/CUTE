@triton.autotune(configs=[ 
    triton.Config({'BLOCK_SIZE_M': m, 'BLOCK_SIZE_K': k, 'GROUP_SIZE_M': 8}, num_stages=s, num_warps=w)
    for m in [32, 64, 128, 256] for k in [32, 64, 128] for s in [2, 3, 4] for w in [4, 8]
], key=['M', 'K', 'stride_xk', 'second_step'])
@triton.jit
def mmt_kernel(
    x, y, M, K,  # input: x[M, K], output: y[M, M]
    stride_xm, stride_xk,
    stride_ym, stride_yn,
    second_step,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  # số blocks trong một group
):
    pid = tl.program_id(axis=0)

    # Tính số lượng và vị trí của blocks
    blks_per_row = tl.cdiv(M, BLOCK_SIZE_M)               
    total_blks_per_group = GROUP_SIZE_M * blks_per_row  
    current_group = pid // total_blks_per_group           

    # Tính vị trí block trong group
    first_blk_in_group = current_group * GROUP_SIZE_M                              
    blks_in_this_group = min(blks_per_row - first_blk_in_group, GROUP_SIZE_M)  

    # Tính tọa độ block (hàng, cột)
    blk_row = first_blk_in_group + ((pid % total_blks_per_group) % blks_in_this_group)
    blk_col = (pid % total_blks_per_group) // blks_in_this_group

    # Chỉ tính nửa trên của ma trận (vì đối xứng)
    if blk_row > blk_col: return

    # Tính offset cho truy cập ma trận
    row_indices = (blk_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    col_indices = (blk_col * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    k_indices = tl.arange(0, BLOCK_SIZE_K)

    # Tính địa chỉ cho các phần tử của ma trận
    row_ptrs = x + (row_indices[:, None] * stride_xm + k_indices[None, :] * stride_xk)
    col_ptrs = x + (col_indices[:, None] * stride_xm + k_indices[None, :] * stride_xk)

    # Khởi tạo ma trận kết quả
    result = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    # Thực hiện phép nhân ma trận theo từng block
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load dữ liệu từ ma trận x
        mask = k_indices[None, :] < K - k * BLOCK_SIZE_K
        blk_a = tl.load(row_ptrs, mask=mask, other=0.0)
        blk_b = tl.load(col_ptrs, mask=mask, other=0.0)
        
        # Nhân ma trận và cộng vào kết quả
        blk_bT = tl.permute(blk_b, (1, 0))
        result = tl.dot(blk_a, blk_bT, result)
        
        # Cập nhật con trỏ cho block tiếp theo
        row_ptrs += BLOCK_SIZE_K * stride_xk
        col_ptrs += BLOCK_SIZE_K * stride_xk

    # Tính vị trí lưu kết quả
    out_row_indices = blk_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_col_indices = blk_col * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Con trỏ đến vị trí lưu kết quả
    result_ptrs  = y + stride_ym * out_row_indices[:, None] + stride_yn * out_col_indices[None, :]
    result_mask  = (out_row_indices[:, None] < M) & (out_col_indices[None, :] < M)

    if second_step == 1 and M == K: # chỉ áp dụng step 2 cho ma trận vuông
        inp_ptrs = x + stride_xm * out_row_indices[:, None] + stride_xk * out_col_indices[None, :]
        inp      = tl.load(inp_ptrs, mask=result_mask, other=0.0)
        result   = -4.7750 * inp + 2.0315 * result  # b * Y + c * Z

    # Lưu kết quả
    tl.store(result_ptrs, result, mask=result_mask)

    # Lưu phần đối xứng (transpose)
    if blk_row < blk_col:
        trans_ptrs = y + stride_ym * out_col_indices[:, None] + stride_yn * out_row_indices[None, :]
        trans_mask = (out_col_indices[:, None] < M) & (out_row_indices[None, :] < M)
        resultT = tl.permute(result, (1,0))
        tl.store(trans_ptrs, resultT, mask=trans_mask)


def mmt(x: Tensor, y: Tensor, second_step=False):
    M, K = x.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    mmt_kernel[grid](
        x, y, M, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        1 if second_step else 0
    )

@torch.compile()
def zeropower_newtonschulz5(X:Tensor)->Tensor:  # zeropower_newtonschulz5 phiên bản need4speed
    need_invert = X.size(-2) > X.size(-1)       # Sẽ báo lỗi nếu X.dim < 2
    if need_invert: X = X.mT                    # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    X /= X.norm(dim=(-2,-1), keepdim=True)+1e-7 # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    M = X.shape[0]
    Y = torch.empty((M, M), device=X.device, dtype=X.dtype)
    Z = torch.empty((M, M), device=X.device, dtype=X.dtype)
    for _ in range(5):
        X = X.contiguous()
        mmt(X, Y, False)        # Y = X @ X.mT
        mmt(Y, Z, True)         # Z = b * Y + c * Y @ Y
        X = 3.4445 * X + Z @ X  # X = a * X + (b * Y + c * Y @ Y) @ X
    return X.mT if need_invert else X
