// Rút gọn từ https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/cuda/wkv_cuda.cu
#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
#define MIN_VALUE (-1e38)

// By using the __restrict__ keyword, the programmer informs the compiler that a particular pointer argument or variable 
// does not alias with any other pointer in the same scope. This allows the compiler to perform certain optimizations
// and generate more efficient code.

// Note: Đây là kernel của công thức GPT nhưng triển khai theo RNN để tránh overflow !!!
__global__ void kernel_forward( const int B, const int T, const int C,
                                const float *__restrict__ const _w, 
                                const bf16 *__restrict__ const _u,
                                const bf16 *__restrict__ const _k,
                                const bf16 *__restrict__ const _v,
                                bf16 *__restrict__ const _y) {

    // Xác định index hiện tại trong mảng B * C threads
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // idx = _b * C + _c đại diện cho phần tử (_b, c) của ma trận (B, C)
    const int _b = idx / C; // hàng _b: batch đang xét
    const int _c = idx % C; // cột _c: channel đang xét
    // Mỗi batch chứa _b phần tử (T * C) kiểu F (scalar)
    const int _offset = _b * T * C + _c; // offset để trỏ tới các scalar values của channel đang xét

    float u = _u[_c]; // u của channel đang xét
    float w = _w[_c]; // w của channel đang xét

    const bf16 *__restrict__ const k = _k + _offset; // trỏ tới k của channel đang xét
    const bf16 *__restrict__ const v = _v + _offset; // trỏ tới v của channel đang xét
    bf16 *__restrict__ const y = _y + _offset; // trỏ tới giá trị đầu ra của channel đang xét

    // aa and bb are running sums divided by exp(pp) (to avoid overflow)
    float aa = 0, bb = 0, pp = MIN_VALUE;

    // Tính giá trị đầu ra bằng cách chạy dọc theo ctx_len T
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float( k[ii] );
        const float vv = float( v[ii] );

        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[ii] = bf16( (e1 * aa + e2 * vv) / (e1 * bb + e2) );

        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
}

__global__ void kernel_backward(const int B, const int T, const int C,
                                const float *__restrict__ const _w,
                                const bf16 *__restrict__ const _u,
                                const bf16 *__restrict__ const _k,
                                const bf16 *__restrict__ const _v,
                                const bf16 *__restrict__ const _y,
                                const bf16 *__restrict__ const _gy, // gradient đầu vào
                                bf16 *__restrict__ const _gw, // gradient đầu ra của _w
                                bf16 *__restrict__ const _gu, // gradient đầu ra của _u
                                bf16 *__restrict__ const _gk, // gradient đầu ra của _k
                                bf16 *__restrict__ const _gv) // gradient đầu ra của _v
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    float u = _u[_c];
    float w = _w[_c];

    const bf16 *__restrict__ const k = _k + _offset;
    const bf16 *__restrict__ const v = _v + _offset;
    const bf16 *__restrict__ const y = _y + _offset;
    const bf16 *__restrict__ const gy = _gy + _offset;

    bf16 *__restrict__ const gk = _gk + _offset;
    bf16 *__restrict__ const gv = _gv + _offset;

    float q[Tmax], r[Tmax];

    float gw = 0, gu = 0;
    float aa = 0, bb = 0;
    float ga = 0, gb = 0;
    float pp = MIN_VALUE;

    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = k[ii];
        const float vv = v[ii];
        const float yy = y[ii];

        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);

        const float qq = gy[ii] / (e1 * bb + e2);
        gw += (ga - gb * yy) * e1 * qq;
        gu += (vv - yy) * e2 * qq;
        q[i] = qq;
        r[i] = ww - p;

        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        ga = e1 * (aa + ga);
        gb = e1 * (bb + gb);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }

    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] = bf16( gw * _w[_c] ); // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = bf16( gu );

    aa = 0, bb = 0, pp = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        const float kk = k[ii];
        const float vv = v[ii];
        const float yy = y[ii];
        const float qq = q[i];
        const float rr = r[i];

        float e1 = qq * exp(rr);
        float e2 = exp(kk + pp);
        gk[ii] = bf16( e1 * (vv - yy) + e2 * (aa * vv + bb) );
        gv[ii] = bf16( e1 + e2 * aa );

        const float ww = w + pp;
        const float www = rr - u - kk;
        const float p = max(ww, www);
        e1 = exp(ww - p);
        e2 = qq * exp(www - p);
        aa = e1 * aa + e2;
        bb = e1 * bb - e2 * yy;
        pp = p;
    }
}


void forward(int64_t _B, int64_t _T, int64_t _C, 
        torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    const int B = _B, T = _T, C = _C; // convert i64 to i32

    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
 
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, 
        w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>());
}


void backward(int64_t _B, int64_t _T, int64_t _C, 
        torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
        torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {    
    const int B = _B, T = _T, C = _C; // convert i64 to i32

    dim3 threadsPerBlock( min(C, 32) ); // lấy min đề phòng khi C < 32 dẫn tới trường hợp ko có thread nào đc launch
    assert(B * C % threadsPerBlock.x == 0); 
    dim3 numBlocks(B * C / threadsPerBlock.x);

    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, 
        w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(), 
        gy.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "racoon_wkv forward");
    m.def("backward", &backward, "racoon_wkv backward");
}

TORCH_LIBRARY(racoon_bf16_wkv, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
