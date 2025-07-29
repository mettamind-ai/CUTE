- https://www.alphaxiv.org/abs/2507.19595
- https://x.com/spiraldalat/status/1950099550917493107

Retnet (γ độc lập với dữ liệu)
```js
S_t = γ * S_{t-1} + k_t^T @ v_t  // γ là data-independent decay
o_t = q_t @ S_t

// Parallel (training)
O = (Q@K^T ⊙ D) @ V  // D là decay matrix

// Chunkwise (hybrid)
// Chia sequence thành chunks, parallel trong chunk, recurrent giữa chunks
```

Mamba (A_t phụ thuộc dữ liệu)
```js
h_t = A_t ⊙ h_{t-1} + B_t @ x_t  // A_t là data-dependent gate
y_t = C_t @ h_t

// Từ input x_t, tạo ra các parameters động
B_t = Linear_B(x_t)  // [batch, d_state]
C_t = Linear_C(x_t)  // [batch, d_state] 
Δ_t = Linear_Δ(x_t)  // [batch, d_state] (time step)

// A_t được tính từ Δ_t
A_t = discretize(A_continuous, Δ_t)

// Associative scan (efficient parallel)
// Sử dụng parallel prefix sum
```

