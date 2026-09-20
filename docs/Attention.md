# Attention

## Attention forward pass

This example uses three input tokens, each with an embedding dimension of 2. It describes single-head scaled dot-product attention with an output projection.

### 1. Token embeddings and projection matrices

The input matrix is

$$
X =
\begin{bmatrix}
x_1 & x_2 \\
y_1 & y_2 \\
z_1 & z_2
\end{bmatrix}.
$$

The query, key, value, and output projection matrices are

$$
W_Q =
\begin{bmatrix}
Q_1 & Q_2 \\
Q_3 & Q_4
\end{bmatrix},
\qquad
W_K =
\begin{bmatrix}
K_1 & K_2 \\
K_3 & K_4
\end{bmatrix},
$$

$$
W_V =
\begin{bmatrix}
V_1 & V_2 \\
V_3 & V_4
\end{bmatrix},
\qquad
W_O =
\begin{bmatrix}
O_1 & O_2 \\
O_3 & O_4
\end{bmatrix}.
$$

### 2. Query, key, and value vectors

The projected query, key, and value matrices are

$$
Q = XW_Q =
\begin{bmatrix}
xq_1 & xq_2 \\
yq_1 & yq_2 \\
zq_1 & zq_2
\end{bmatrix},
$$

$$
K = XW_K =
\begin{bmatrix}
xk_1 & xk_2 \\
yk_1 & yk_2 \\
zk_1 & zk_2
\end{bmatrix},
$$

$$
V = XW_V =
\begin{bmatrix}
xv_1 & xv_2 \\
yv_1 & yv_2 \\
zv_1 & zv_2
\end{bmatrix}.
$$

### 3. Attention scores

Because the key dimension is $d_k = 2$, each query-key dot product is divided by $\sqrt{2}$:

$$
S = \frac{QK^T}{\sqrt{2}}.
$$

For the query belonging to token $x$:

$$
\left[
\frac{xq_1xk_1+xq_2xk_2}{\sqrt{2}},
\frac{xq_1yk_1+xq_2yk_2}{\sqrt{2}},
\frac{xq_1zk_1+xq_2zk_2}{\sqrt{2}}
\right]
= [xS_x, xS_y, xS_z].
$$

For the query belonging to token $y$:

$$
\left[
\frac{yq_1xk_1+yq_2xk_2}{\sqrt{2}},
\frac{yq_1yk_1+yq_2yk_2}{\sqrt{2}},
\frac{yq_1zk_1+yq_2zk_2}{\sqrt{2}}
\right]
= [yS_x, yS_y, yS_z].
$$

For the query belonging to token $z$:

$$
\left[
\frac{zq_1xk_1+zq_2xk_2}{\sqrt{2}},
\frac{zq_1yk_1+zq_2yk_2}{\sqrt{2}},
\frac{zq_1zk_1+zq_2zk_2}{\sqrt{2}}
\right]
= [zS_x, zS_y, zS_z].
$$

Therefore,

$$
S =
\begin{bmatrix}
xS_x & xS_y & xS_z \\
yS_x & yS_y & yS_z \\
zS_x & zS_y & zS_z
\end{bmatrix}.
$$

### 4. Attention weights

Softmax is applied independently to each row of the score matrix:

$$
\mathrm{softmax}([xS_x,xS_y,xS_z]) = [xW_x,xW_y,xW_z],
$$

$$
\mathrm{softmax}([yS_x,yS_y,yS_z]) = [yW_x,yW_y,yW_z],
$$

$$
\mathrm{softmax}([zS_x,zS_y,zS_z]) = [zW_x,zW_y,zW_z].
$$

Thus,

$$
W = \mathrm{softmax}(S) =
\begin{bmatrix}
xW_x & xW_y & xW_z \\
yW_x & yW_y & yW_z \\
zW_x & zW_y & zW_z
\end{bmatrix}.
$$

### 5. Weighted sums of the value vectors

The attention output before the output projection is

$$
H = WV.
$$

For token $x$:

$$
[xW_x xv_1 + xW_y yv_1 + xW_z zv_1,\;
  xW_x xv_2 + xW_y yv_2 + xW_z zv_2] = [m,n].
$$

For token $y$:

$$
[yW_x xv_1 + yW_y yv_1 + yW_z zv_1,\;
  yW_x xv_2 + yW_y yv_2 + yW_z zv_2] = [o,p].
$$

For token $z$:

$$
[zW_x xv_1 + zW_y yv_1 + zW_z zv_1,\;
  zW_x xv_2 + zW_y yv_2 + zW_z zv_2] = [q,r].
$$

Therefore,

$$
H =
\begin{bmatrix}
m & n \\
o & p \\
q & r
\end{bmatrix}.
$$

### 6. Final outputs

The final attention output is

$$
A = HW_O.
$$

For token $x$:

$$
[mO_1+nO_3,\;mO_2+nO_4] = [a,b].
$$

For token $y$:

$$
[oO_1+pO_3,\;oO_2+pO_4] = [c,d].
$$

For token $z$:

$$
[qO_1+rO_3,\;qO_2+rO_4] = [e,f].
$$

Thus,

$$
A =
\begin{bmatrix}
a & b \\
c & d \\
e & f
\end{bmatrix}.
$$

In compact form, the complete forward pass is

$$
A = \mathrm{softmax}\!\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{2}}\right)(XW_V)W_O.
$$

## Attention backward pass

Given the loss $L$, we need to calculate gradients with respect to the query, key, value, and output projection matrices, as well as the input activations. We shall derive the gradient of one representative element from each projection matrix and one representative input activation before presenting the corresponding matrix formulas..

### Example: gradient of $O_1$

The output-projection element $O_1$ contributes to $a$, $c$, and $e$:

$$
a = mO_1+nO_3,
\qquad
c = oO_1+pO_3,
\qquad
e = qO_1+rO_3.
$$

It does not contribute to $b$, $d$, or $f$. Therefore,

$$
\frac{\partial a}{\partial O_1}=m,
\qquad
\frac{\partial b}{\partial O_1}=0,
$$

$$
\frac{\partial c}{\partial O_1}=o,
\qquad
\frac{\partial d}{\partial O_1}=0,
$$

$$
\frac{\partial e}{\partial O_1}=q,
\qquad
\frac{\partial f}{\partial O_1}=0.
$$

Applying the chain rule gives

```math
\frac{\partial L}{\partial O_1}
=
\frac{\partial L}{\partial a}\frac{\partial a}{\partial O_1}
+
\frac{\partial L}{\partial c}\frac{\partial c}{\partial O_1}
+
\frac{\partial L}{\partial e}\frac{\partial e}{\partial O_1}.
```

Hence,

```math
\boxed{
\frac{\partial L}{\partial O_1}
=
\frac{\partial L}{\partial a}m
+
\frac{\partial L}{\partial c}o
+
\frac{\partial L}{\partial e}q
}
```

More generally, because $A=HW_O$,

```math
\frac{\partial L}{\partial W_O}=H^T\frac{\partial L}{\partial A},
\qquad
\frac{\partial L}{\partial H}=\frac{\partial L}{\partial A}W_O^T.
```

## Gradient of Q1

Q1 contributes directly to the first element of the query projection for all tokens in the sequence. Each of these query components affects the attention scores in that token’s row of the score matrix. Those scores affect the corresponding row of attention weights, weighted value sums, and final outputs.