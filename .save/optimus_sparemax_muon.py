############################
##  Fused Sparsemax loss  ##
############################

@triton.jit
def per_label_sparsemax_loss(
        logits_ptr,             # [vocab]  — sẽ bị ghi đè = grad
        sorted_ptr,             # [vocab]  — logit đã sort giảm dần
        target_ptr, loss_ptr,   # 1 phần tử
        stride: tl.constexpr, sorted_stride: tl.constexpr,
        vocab:  tl.constexpr, ignore: tl.constexpr,
        BLOCK:  tl.constexpr, num_warps: tl.constexpr
    ):
    ### --- Mỗi programme xử lý 1 sample -------------------------------
    pid  = tl.program_id(0).to(tl.int64)
    row  = logits_ptr + pid*stride
    srow = sorted_ptr + pid*sorted_stride
    tgt  = tl.load(target_ptr + pid)

    offs = tl.arange(0, BLOCK)
    mask = offs < vocab
    if tgt == ignore: tl.store(row + offs, 0, mask=mask); return

    # 1) Lấy logits (đã sort) để tính tau
    z_sorted = tl.load(srow + offs, mask=mask, other=-float("inf")).to(tl.float32)
    z_valid  = tl.where(mask, z_sorted, 0.)
    cssv     = tl.cumsum(z_valid, axis=0)
    r        = (offs + 1).to(tl.float32)
    t_vec    = (cssv - 1) / tl.where(mask, r, 1.)
    support  = (z_sorted > t_vec) & mask
    k        = tl.maximum(tl.sum(support.to(tl.int32), 0), 1).to(tl.float32)
    s        = tl.sum(tl.where(support, z_sorted, 0.), 0)
    tau      = (s - 1) / k # threshold cần tính

    # 2) Tính y_i = max(z_i‑tau, 0)
    z = tl.load(row + offs, mask=mask, other=0.).to(tl.float32)
    y = tl.maximum(z - tau, 0)

    # 3) Tính loss
    z_tgt       = tl.load(row + tgt).to(tl.float32)
    square_sum  = tl.sum(y*y, axis=0)
    loss        = 0.5*square_sum + tau - z_tgt + 0.5
    tl.store(loss_ptr + pid, loss)

    # 4) Tính grad = y_i - 1_{i=tgt}
    grad = y - tl.where(offs == tgt, 1, 0)
    tl.store(row + offs, grad, mask=mask)  # ghi đè logits = grad


class FusedLinearSparsemaxLoss(torch.autograd.Function):
    """ Linear (x @ Wᵀ) + Sparsemax‑loss, tính gradient (∂L/∂logits) NGAY trong forward nhờ kernel Triton """
    @staticmethod
    @torch.no_grad()                            # tắt autograd bên trong
    @torch.amp.custom_fwd(device_type="cuda")   # hỗ trợ AMP
    def forward(ctx,
        _input: torch.Tensor,                   # [T, D]
        weight: torch.Tensor,                   # [V, D]
        target: torch.Tensor,                   # [T]
        n_ignores: int = 0,
        ignore:    int = -100,
    ) -> torch.Tensor:
        # bộ đệm kết quả / grad
        grad_weight = torch.zeros_like(weight, device=_input.device) if weight.requires_grad else None
        grad_input  = torch.empty_like(_input, device=_input.device)
        losses      = torch.zeros(_input.size(0), device=_input.device, dtype=torch.float32)

        n_labels, vocab = _input.shape[0], weight.shape[0]
        BLOCK = triton.next_power_of_2(vocab)

        num_warps = 8 if vocab <= 1024*8 else 16
        step = min(1024*4, n_labels // 2) # để luôn test được chunked CE

        for s in range(0, n_labels, step):
            e = min(s + step, n_labels)

            # (1) Tính logits = x @ Wᵀ
            logits = (_input[s:e] @ weight.t()).contiguous() # [chunk, V]

            # (2) Cần logits đã sort ↓ cho sparsemax
            sorted_logits = torch.sort(logits.float(), dim=-1, descending=True).values

            # (3) Kernel Triton: vừa trả loss, vừa ghi đè logits = ∂L/∂logit
            per_label_sparsemax_loss[(logits.size(0),)](
                logits_ptr      = logits,
                sorted_ptr      = sorted_logits,
                target_ptr      = target[s:e],
                loss_ptr        = losses[s:e],
                stride          = logits.stride(0),
                sorted_stride   = sorted_logits.stride(0),
                vocab           = vocab,
                ignore          = ignore,
                BLOCK           = BLOCK,
                num_warps       = num_warps,
            )

            # (4) Tính ∂L/∂input = grad_logits @ W
            grad_input[s:e] = logits @ weight  # logits lúc này chứa grad

            # (5) Tính ∂L/∂W   += grad_logitsᵀ @ x
            if grad_weight is not None: grad_weight += logits.t() @ _input[s:e]

        # Giảm trung bình (đã loại bỏ các nhãn ignore nếu caller cung cấp)
        mean_reduction = 1.0 / (n_labels - n_ignores)

        # Lưu gradients (đã scale) để dùng ở backward
        ctx.save_for_backward(
            grad_input.detach()  * mean_reduction,
            grad_weight.detach() * mean_reduction if grad_weight is not None else None
        )
        return losses.sum() * mean_reduction

    # ------------------------------------------------------------------ #
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        """ grad_output: ∂L_total/∂loss_scalar (thường = 1 nếu .backward() trực tiếp) """
        grad_input, grad_weight = ctx.saved_tensors
        grad_input = grad_input * grad_output  # chain‑rule
        if grad_weight is not None: grad_weight = grad_weight * grad_output
        # trả lần lượt cho (_input, weight, target, n_ignores, ignore)
        return grad_input, grad_weight, None, None, None


#################################################################
##  MUON optimizer - MomentUm Orthogonalized by Newton-schulz  ##
#################################################################
'''
Muon is "Matrix-structured steepest descent with spectral norm regularization"

Phương pháp này bắt đầu từ ý tưởng steepest descent cơ bản - đi theo hướng giảm nhanh nhất của hàm loss bằng cách di chuyển ngược hướng gradient. Khác với các optimizer truyền thống cập nhật từng tham số riêng lẻ, Muon áp dụng cách tiếp cận có cấu trúc ma trận (matrix-structured). Điều này có nghĩa là thay vì xem mỗi element của weight matrix như các đơn vị độc lập, Muon xử lý toàn bộ ma trận như một thể thống nhất và tìm cách biến đổi nó một cách có hệ thống.

Để đảm bảo quá trình cập nhật không bị "vượt quá giới hạn", Muon áp dụng spectral norm regularization (điều chuẩn chuẩn phổ) - một ràng buộc giới hạn sức mạnh của ma trận update thông qua điều kiện ||O_t||₂ ≤ 1. Ràng buộc này không chỉ đảm bảo tính ổn định mà còn tự nhiên dẫn đến nghiệm tối ưu có dạng O_t = UV^T từ phép phân tích SVD của gradient. Cách tiếp cận này cho phép Muon tự động cân bằng toàn bộ ma trận thông qua một ràng buộc toàn cục, thay vì phải điều chỉnh learning rate cho từng parameter như AdamW. Kết quả là một phương pháp optimization vừa đơn giản vừa hiệu quả, duy trì cấu trúc ma trận trong khi đảm bảo convergence ổn định.

- Chuẩn phổ (spectral norm): Là giá trị singular value lớn nhất của ma trận - ĐO "SỨC MẠNH KÉO DÀI" TỐI ĐA mà ma trận có thể gây ra cho vector.
- UV^T là ma trận trực giao (orthogonal) với spectral norm = 1. Newton-Schulz là cách tính gần đúng UV^T mà không cần SVD đắt đỏ.

TẠI SAO MUON LẠI SỬ DỤNG ĐIỀU CHUẨN CHUẨN PHỔ? (spectral norm regularization)
- Muon giới hạn chuẩn phổ <= 1 nghĩa là giới hạn Sức mạnh kéo dài mà ma trận gây ra cho vector (control maximum damage) tránh bùng nổ gradient
- Sử dụng spectral norm regularization tự nhiên dẫn tới SVD structure (nghiệm tối ưu TỰ ĐỘNG là O=UV^T từ SVD G = UΣV^T)
- Spectral norm liên kết với inverse Fisher matrix approximation là 1 phương pháp đạo hàm bậc 2 (điều này quan trọng)
  Shampoo dùng: E[GG^T]^{-1/4} G E[G^T G]^{-1/4} => Simplify → UV^T (spectral structure) => Muon là minimal version của class này

=> Muon thực sự nhìn optimization landscape từ góc nhìn geometric hoàn toàn khác với Adam !!! Và vì Muon xấp xỉ tối thiểu inverse Hessian, nó có thể nhìn thấy "valleys and ridges" của loss landscape, trong khi đó Adam chỉ nhìn thấy local slopes, đặc biệt khi BATCH SIZE lớn và global structure quan trọng hơn local adaptivity.
'''
@torch.compile()
def zeropower_via_newtonschulz5(X:Tensor)->Tensor:  # zero(excess)power có nghĩa là spectral norm = 1 => perfect balance
    need_invert = X.size(-2) > X.size(-1)           # Sẽ báo lỗi nếu X.dim < 2
    X = X.bfloat16()                                # Need for Speed
    if need_invert: X = X.mT                        # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    X /= X.norm(dim=(-2,-1), keepdim=True)+1e-7     # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    a, b, c = ( 3.4445, -4.7750, 2.0315 )           # Hằng số tối ưu hóa cho NS iteration, tối ưu sau 5 iters
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 1: error ≈ ε  (NS có sai số giảm theo lũy thừa)
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 2: error ≈ ε²
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 3: error ≈ ε⁴
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 4  ... có thể xem mỗi NS iter như 1 lần khử nhiễu ...
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 5: error ≈ ε¹⁶, flatten singular values to range (0.7, 1.3)
    A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X       # iter 6: thêm 1 lần khử nhiễu đẻ ổn định hơn với int8? (0.9, 1.1) ?!?
    return X.mT if need_invert else X
    '''Khi thực hiện phép tính A = X @ X.mT, chúng ta đang tạo ra một hiệu ứng trung bình hóa. Trong phép nhân ma trận này, mỗi phần tử của ma trận kết quả A được hình thành từ tổng của nhiều phép nhân giữa các phần tử khác nhau trong X. Noise ngẫu nhiên, do bản chất không có cấu trúc, có xu hướng triệt tiêu lẫn nhau trong quá trình cộng tổng này, trong khi tín hiệu thật có cấu trúc rõ ràng được củng cố và tăng cường.

    Phép update X = aX + (bA + c*A@A) @ X làm mịn dữ liệu: Sau khi ma trận A đã được "lọc" với X ban đầu, chúng ta áp dụng một phép biến đổi mà trong đó mỗi phần tử mới được tạo thành từ sự kết hợp của nhiều phần tử cũ. Quá trình này tương tự như việc áp dụng bộ lọc không gian, nơi các giá trị lân cận ảnh hưởng và cân bằng lẫn nhau, từ đó làm mịn những biến động đột ngột và bất thường.

    Cuối cùng, xu hướng convergence hướng về cấu trúc orthogonal tạo ra một cơ chế lọc tự nhiên. Sau nhiều iterations, X dần hội tụ về ma trận orthogonal, và quá trình này tự động "đẩy ra" những thành phần không phù hợp với cấu trúc orthogonal, bao gồm cả noise, trong khi bảo tồn những directions quan trọng và có ý nghĩa nhất.
    '''

## POLAR https://alphaxiv.org/abs/2505.16932 | https://x.com/gowerrobert/status/1930292178456039739
coeffs_list = [  (8.28721201814563   , -23.595886519098837  , 17.300387312530933   ),    # iter 1
                 (4.107059111542203  ,  -2.9478499167379106 ,  0.5448431082926601  ),    # iter 2
                 (3.9486908534822946 ,  -2.908902115962949  ,  0.5518191394370137  ),    # iter 3
                 (3.3184196573706015 ,  -2.488488024314874  ,  0.51004894012372    ),    # iter 4
                 (2.300652019954817  ,  -1.6689039845747493 ,  0.4188073119525673  ),    # iter 5
                 (1.891301407787398  ,  -1.2679958271945868 ,  0.37680408948524835 ),    # iter 6
                 (1.8750014808534479 ,  -1.2500016453999487 ,  0.3750001645474248  ),    # iter 7
              ]#  1.875                 -1.25                  0.375         => subsequent coeffs
coeffs_list = [(a/1.01,b/1.01**3,c/1.01**5) for (a,b,c) in coeffs_list] + [(1.875, -1.25 , 0.375)]*3
#    safety factor for numerical stability
@torch.compile()
def PolarExpress(X:Tensor, steps=6)->Tensor:        # coeffs_list cho 5 tới 10 lần lặp
    need_invert = X.size(-2) > X.size(-1)           # Sẽ báo lỗi nếu X.dim < 2
    X = X.bfloat16()                                # Need for Speed
    if need_invert: X = X.mT                        # Ensure số cột ≥ số hàng; giúp NS hoạt động tốt
    # X /= X.norm(dim=(-2,-1), keepdim=True)*1.01   # Ensure spectral norm ≤ 1, <= cách làm tròn này gây NaN
    X   /= X.norm(dim=(-2,-1), keepdim=True)+1e-7   # Ensure spectral norm ≤ 1, điều kiện bắt buộc để NS hội tụ
    for (a,b,c) in coeffs_list[:steps]:
        A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X   # X <- aX + bXˆ3 + cXˆ5
    return X.mT if need_invert else X


class Muon1GPU(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, **args):
        super().__init__(list(params), dict(lr=lr, wd=weight_decay, mm=momentum))

    @torch.no_grad()
    @torch.compiler.disable
    def step(self):
        for group in self.param_groups:
            for p in group['params']:                   # với mỗi tham số p trong model
                if p.grad is None: continue             # bỏ qua nếu không có gradient

                g, st = p.grad, self.state[p]           # lấy gradient và optim state và khởi tạo momentum nếu chưa có
                if 'mm' not in st: 
                    st['mm'] = torch.zeros_like(g, dtype=torch.bfloat16)

                st['mm'].lerp_(g, 1 - group['mm'])          # momentum = momentum * 0.95 + gradient * 0.05
                g = g.lerp_(st['mm'], group['mm'])          # gradient = gradient * 0.05 + momentum * 0.95

                if g.ndim != 2: g = g.view(len(g), -1)      # 2D hoá
                # go = zeropower_via_newtonschulz5(g)       # Trực giao Newton-Schulz gốc
                go = PolarExpress(g)                        # Thuật toán Polar Express tính orthogonal grad
                if go.shape != p.shape: go=go.view_as(p)    # Reshape back if needed

                # Cập nhật tham số p, theo gradient, learning rate và weight decay với 2 phép tính:
                p.mul_(1 - group['lr']*group['wd'])     # 1) p *= (1 - lr*wd) <= thu nhỏ p nếu wd > 0
                rows, cols = p.size(-2), p.size(-1)     # 2) p -= go * lr * sqrt(max(1, rows / cols))
                x = max(1, rows / cols)**0.5 
                p.add_(go, alpha=-group['lr']*x)
