# Attention

## Attention forward pass

This example uses an input sequence of 3 tokens, each with an embedding dimension of 2. It describes single-head scaled dot-product attention with an output projection.

### 1. Token embeddings and projection matrices

The input sequence consists of the three token embeddings

$$
[x_1,x_2],\qquad
[y_1,y_2],\qquad
[z_1,z_2].
$$

The query, key, value, and output projection matrices are

$$
W_Q =
\begin{bmatrix}
Q_1 & Q_2 \
Q_3 & Q_4
\end{bmatrix},
\qquad
W_K =
\begin{bmatrix}
K_1 & K_2 \
K_3 & K_4
\end{bmatrix},
$$

$$
W_V =
\begin{bmatrix}
V_1 & V_2 \
V_3 & V_4
\end{bmatrix},
\qquad
W_O =
\begin{bmatrix}
O_1 & O_2 \
O_3 & O_4
\end{bmatrix}.
$$

### 2. Query, key, and value vectors

Each token embedding is independently projected to produce a query vector, a key vector, and a value vector.

For token $x$:

$$
[xq_1,xq_2]=[x_1,x_2]W_Q,
$$

$$
[xk_1,xk_2]=[x_1,x_2]W_K,
$$

$$
[xv_1,xv_2]=[x_1,x_2]W_V.
$$

For token $y$:

$$
[yq_1,yq_2]=[y_1,y_2]W_Q,
$$

$$
[yk_1,yk_2]=[y_1,y_2]W_K,
$$

$$
[yv_1,yv_2]=[y_1,y_2]W_V.
$$

For token $z$:

$$
[zq_1,zq_2]=[z_1,z_2]W_Q,
$$

$$
[zk_1,zk_2]=[z_1,z_2]W_K,
$$

$$
[zv_1,zv_2]=[z_1,z_2]W_V.
$$

For example,

$$
xq_1=x_1Q_1+x_2Q_3,
\qquad
xq_2=x_1Q_2+x_2Q_4.
$$

The resulting sequence of query vectors is therefore

$$
[xq_1,xq_2],\qquad
[yq_1,yq_2],\qquad
[zq_1,zq_2],
$$

with corresponding sequences of key and value vectors.

### 3. Attention scores

Each token's query vector is compared with the key vector of every token in the sequence, including its own key vector.

Because the key dimension is $d_k=2$, each query-key dot product is divided by $\sqrt{2}$.

For the query belonging to token $x$:

$$
\frac{xq_1xk_1+xq_2xk_2}{\sqrt{2}}=xS_x,
$$

$$
\frac{xq_1yk_1+xq_2yk_2}{\sqrt{2}}=xS_y,
$$

$$
\frac{xq_1zk_1+xq_2zk_2}{\sqrt{2}}=xS_z.
$$

Thus token $x$ has the sequence of attention scores

$$
[xS_x,xS_y,xS_z].
$$

For the query belonging to token $y$:

$$
\frac{yq_1xk_1+yq_2xk_2}{\sqrt{2}}=yS_x,
$$

$$
\frac{yq_1yk_1+yq_2yk_2}{\sqrt{2}}=yS_y,
$$

$$
\frac{yq_1zk_1+yq_2zk_2}{\sqrt{2}}=yS_z.
$$

Thus token $y$ has the sequence

$$
[yS_x,yS_y,yS_z].
$$

For the query belonging to token $z$:

$$
\frac{zq_1xk_1+zq_2xk_2}{\sqrt{2}}=zS_x,
$$

$$
\frac{zq_1yk_1+zq_2yk_2}{\sqrt{2}}=zS_y,
$$

$$
\frac{zq_1zk_1+zq_2zk_2}{\sqrt{2}}=zS_z.
$$

Thus token $z$ has the sequence

$$
[zS_x,zS_y,zS_z].
$$

### 4. Attention weights

Softmax is applied independently to the three scores belonging to each token.

For token $x$:

$$
\operatorname{softmax}([xS_x,xS_y,xS_z])
========================================

[xW_x,xW_y,xW_z].
$$

For token $y$:

$$
\operatorname{softmax}([yS_x,yS_y,yS_z])
========================================

[yW_x,yW_y,yW_z].
$$

For token $z$:

$$
\operatorname{softmax}([zS_x,zS_y,zS_z])
========================================

[zW_x,zW_y,zW_z].
$$

For example, $xW_y$ is the attention weight given to token $y$ by token $x$.

For each token, its three attention weights sum to 1:

$$
xW_x+xW_y+xW_z=1,
$$

and likewise for $y$ and $z$.

### 5. Weighted sums of the value vectors

Each token's attention weights are used to calculate a weighted sum of the three value vectors.

For token $x$:

$$
m=xW_x xv_1+xW_y yv_1+xW_z zv_1,
$$

$$
n=xW_x xv_2+xW_y yv_2+xW_z zv_2.
$$

Its weighted value vector is therefore

$$
[m,n].
$$

For token $y$:

$$
o=yW_x xv_1+yW_y yv_1+yW_z zv_1,
$$

$$
p=yW_x xv_2+yW_y yv_2+yW_z zv_2.
$$

Its weighted value vector is

$$
[o,p].
$$

For token $z$:

$$
q=zW_x xv_1+zW_y yv_1+zW_z zv_1,
$$

$$
r=zW_x xv_2+zW_y yv_2+zW_z zv_2.
$$

Its weighted value vector is

$$
[q,r].
$$

Thus the sequence of weighted value vectors is

$$
[m,n],\qquad[o,p],\qquad[q,r].
$$

### 6. Final outputs

Each weighted value vector is projected through the output projection matrix $W_O$.

For token $x$:

$$
[m,n]W_O=[a,b],
$$

where

$$
a=mO_1+nO_3,
\qquad
b=mO_2+nO_4.
$$

For token $y$:

$$
[o,p]W_O=[c,d],
$$

where

$$
c=oO_1+pO_3,
\qquad
d=oO_2+pO_4.
$$

For token $z$:

$$
[q,r]W_O=[e,f],
$$

where

$$
e=qO_1+rO_3,
\qquad
f=qO_2+rO_4.
$$

The output sequence from the attention block is therefore

$$
[a,b],\qquad[c,d],\qquad[e,f].
$$

---

# Attention backward pass

Given the loss $L$, we need to calculate gradients with respect to the elements of the query, key, value, and output projection matrices, as well as with respect to the input activations.

For each parameter or activation, its gradient tells us how a small change in its value affects the value of $L$.

We shall derive the gradient of one representative element from each projection matrix and one representative input activation.

## Gradient of $O_1$

The output-projection element $O_1$ contributes to the first output element of every token:

$$
a=mO_1+nO_3,
$$

$$
c=oO_1+pO_3,
$$

$$
e=qO_1+rO_3.
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

$$
\frac{\partial L}{\partial O_1}
===============================

\frac{\partial L}{\partial a}
\frac{\partial a}{\partial O_1}
+
\frac{\partial L}{\partial c}
\frac{\partial c}{\partial O_1}
+
\frac{\partial L}{\partial e}
\frac{\partial e}{\partial O_1}.
$$

Hence,

$$
\boxed{
\frac{\partial L}{\partial O_1}
===============================

m\frac{\partial L}{\partial a}
+
o\frac{\partial L}{\partial c}
+
q\frac{\partial L}{\partial e}
}
$$

The gradient of a shared projection parameter is therefore the sum of its contributions through every token for which it was used.

## Gradient of $Q_1$

The element $Q_1$ of the query projection matrix contributes to the first element of the query vector of every token.

For token $x$,

$$
xq_1=x_1Q_1+x_2Q_3.
$$

For token $y$,

$$
yq_1=y_1Q_1+y_2Q_3.
$$

For token $z$,

$$
zq_1=z_1Q_1+z_2Q_3.
$$

Thus $Q_1$ has three initial paths through the attention calculation:

$$
Q_1
\rightarrow xq_1
\rightarrow [xS_x,xS_y,xS_z]
\rightarrow [xW_x,xW_y,xW_z]
\rightarrow [m,n]
\rightarrow [a,b],
$$

$$
Q_1
\rightarrow yq_1
\rightarrow [yS_x,yS_y,yS_z]
\rightarrow [yW_x,yW_y,yW_z]
\rightarrow [o,p]
\rightarrow [c,d],
$$

$$
Q_1
\rightarrow zq_1
\rightarrow [zS_x,zS_y,zS_z]
\rightarrow [zW_x,zW_y,zW_z]
\rightarrow [q,r]
\rightarrow [e,f].
$$

Applying the chain rule at the query projections gives

$$
\frac{\partial L}{\partial Q_1}
===============================

\frac{\partial L}{\partial xq_1}
\frac{\partial xq_1}{\partial Q_1}
+
\frac{\partial L}{\partial yq_1}
\frac{\partial yq_1}{\partial Q_1}
+
\frac{\partial L}{\partial zq_1}
\frac{\partial zq_1}{\partial Q_1}.
$$

Since

$$
\frac{\partial xq_1}{\partial Q_1}=x_1,
\qquad
\frac{\partial yq_1}{\partial Q_1}=y_1,
\qquad
\frac{\partial zq_1}{\partial Q_1}=z_1,
$$

we obtain

$$
\boxed{
\frac{\partial L}{\partial Q_1}
===============================

x_1\frac{\partial L}{\partial xq_1}
+
y_1\frac{\partial L}{\partial yq_1}
+
z_1\frac{\partial L}{\partial zq_1}
}
$$

We now need to determine the gradients arriving at the individual query elements.

Consider $xq_1$. It contributes to all three attention scores belonging to token $x$:

$$
xS_x=
\frac{xq_1xk_1+xq_2xk_2}{\sqrt{2}},
$$

$$
xS_y=
\frac{xq_1yk_1+xq_2yk_2}{\sqrt{2}},
$$

$$
xS_z=
\frac{xq_1zk_1+xq_2zk_2}{\sqrt{2}}.
$$

Therefore,

$$
\frac{\partial xS_x}{\partial xq_1}
===================================

\frac{xk_1}{\sqrt{2}},
$$

$$
\frac{\partial xS_y}{\partial xq_1}
===================================

\frac{yk_1}{\sqrt{2}},
$$

$$
\frac{\partial xS_z}{\partial xq_1}
===================================

\frac{zk_1}{\sqrt{2}}.
$$

The gradient arriving at $xq_1$ is consequently

$$
\boxed{
\frac{\partial L}{\partial xq_1}
================================

\frac{1}{\sqrt{2}}
\left(
xk_1\frac{\partial L}{\partial xS_x}
+
yk_1\frac{\partial L}{\partial xS_y}
+
zk_1\frac{\partial L}{\partial xS_z}
\right)
}
$$

Similarly,

$$
\boxed{
\frac{\partial L}{\partial yq_1}
================================

\frac{1}{\sqrt{2}}
\left(
xk_1\frac{\partial L}{\partial yS_x}
+
yk_1\frac{\partial L}{\partial yS_y}
+
zk_1\frac{\partial L}{\partial yS_z}
\right)
}
$$

and

$$
\boxed{
\frac{\partial L}{\partial zq_1}
================================

\frac{1}{\sqrt{2}}
\left(
xk_1\frac{\partial L}{\partial zS_x}
+
yk_1\frac{\partial L}{\partial zS_y}
+
zk_1\frac{\partial L}{\partial zS_z}
\right).
}
$$

The remaining question is where gradients such as

$$
\frac{\partial L}{\partial xS_x},
\qquad
\frac{\partial L}{\partial xS_y},
\qquad
\frac{\partial L}{\partial xS_z}
$$

come from.

These gradients pass backwards through the softmax operation. Because each attention weight produced by a softmax depends on every score supplied to that softmax, the three score gradients for token $x$ must take into account all three of its attention-weight gradients:

$$
[xW_x,xW_y,xW_z]
================

\operatorname{softmax}([xS_x,xS_y,xS_z]).
$$

This is the next stage of the backward pass.
