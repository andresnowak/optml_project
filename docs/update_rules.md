# Update Rules for Spectral Muon Variants

This note defines the exact update rules we want to compare. All methods start
from the same momentum update and differ only in how they choose the singular
values of the matrix direction.

## Common Setup

For a matrix parameter $W_t \in \mathbb{R}^{m \times n}$ with gradient $G_t$,
the DynMuon reference keeps a sum-form momentum buffer $B_t$:

$$
B_t = \mu B_{t-1} + G_t.
$$

With Nesterov momentum, the proposed matrix update is:

$$
M_t = G_t + \mu B_t.
$$

Without Nesterov:

$$
M_t = B_t.
$$

Some Muon-style optimizers in this repository store the equivalent EMA-scaled
buffer $\tilde B_t=(1-\mu)B_t$ and form $\tilde M_t=(1-\mu)M_t$. This is the
same lookahead direction up to a positive scalar, which does not affect Muon,
RelMuon, InputMuon, or Kaon after their spectral normalization/shaping. AdamW
fallback parameters use Adam moments, not Nesterov.

All spectral methods take the thin SVD:

$$
M_t = U_t \Sigma_t V_t^T,
$$

where $\Sigma_t = \mathrm{diag}(\sigma_1, \dots, \sigma_r)$ and
$r = \min(m,n)$.

They build a shaped direction:

$$
D_t = U_t \widehat{\Sigma}_t V_t^T,
$$

and update:

$$
W_{t+1} = W_t - \eta_t s(W_t) D_t,
$$

where $\eta_t$ is the learning rate and $s(W_t)$ is the shape LR scaling, for
example $\sqrt{m/n}$ in the current DynMuon-style code. Weight decay, if used, is
decoupled and applied separately.

## 1. DynMuon-Route

DynMuon-Route chooses a layer-specific spectral exponent $p_{t,l}$ and sets:

$$
\widehat{\Sigma}_{t,l} = \Sigma_{t,l}^{p_{t,l}}.
$$

The direction for layer $l$ is therefore:

$$
D_{t,l} = U_{t,l}\Sigma_{t,l}^{p_{t,l}}V_{t,l}^T.
$$

The global DynMuon clock is:

$$
q_t = \frac{t}{T},
$$

$$
p_t = p_{\min} + (p_{\max} - p_{\min})
\frac{1}{1 + \exp\left(\frac{q_t - \tau}{w}\right)}.
$$

In our configs:

$$
p_{\min}=-0.25,\quad p_{\max}=1.0.
$$

The routed exponent is:

$$
p_{t,l} =
\mathrm{clip}
\left(
p_t + \beta (g_{t,l} - \tilde{g}_t),
-0.25,
1.0
\right).
$$

Here $g_{t,l}$ is the layer-local routing proxy. The default proxy is stable rank:

$$
g_{t,l}
= \mathrm{sr}(M_{t,l})
= \frac{\|M_{t,l}\|_F^2}{\sigma_1(M_{t,l})^2}.
$$

The reference $\tilde{g}_t$ is a running network-wide average:

$$
\bar{g}_t = \frac{1}{L}\sum_{l=1}^{L} g_{t,l},
$$

$$
\tilde{g}_t = \rho \tilde{g}_{t-1} + (1-\rho)\bar{g}_t.
$$

Pseudocode:

```text
for each step t:
    compute gradients G_l for each matrix layer l
    B_l <- mu * B_l + G_l
    if nesterov:
        M_l <- G_l + mu * B_l
    else:
        M_l <- B_l

    for each layer l:
        U_l, Sigma_l, V_l^T <- svd(M_l)
        g_l <- ||M_l||_F^2 / sigma_1(M_l)^2

    update network average g_tilde from all g_l
    p_t <- global DynMuon schedule(t)

    for each layer l:
        p_tl <- clip(p_t + beta * (g_l - g_tilde), -0.25, 1.0)
        D_l <- U_l * Sigma_l^p_tl * V_l^T
        W_l <- W_l - lr_t * shape_scale_l * D_l
```

## 2. RelMuon

RelMuon keeps the singular vectors of the proposed update $M_t$, but replaces the
singular values of $M_t$ with singular values taken from the current weights
$W_t$.

Take:

$$
M_t = U_M \Sigma_M V_M^T,
$$

and:

$$
W_t = U_W \Sigma_W V_W^T.
$$

Let $r_M = \mathrm{rank\ dimension}(M_t)$ and define:

$$
\widehat{\sigma}_i =
\frac{\sigma_i(W_t)}{c_t + \epsilon}
\quad \text{for } i=1,\dots,r_M.
$$

A practical normalization is needed so the update scale does not explode. The
cleanest default is to match the Frobenius norm of the Muon direction:

$$
c_t =
\sqrt{
\frac{\sum_{i=1}^{r_M}\sigma_i(W_t)^2}{r_M}
}.
$$

Then:

$$
\widehat{\Sigma}_t
= \mathrm{diag}(\widehat{\sigma}_1,\dots,\widehat{\sigma}_{r_M}),
$$

and:

$$
D_t = U_M \widehat{\Sigma}_t V_M^T.
$$

So RelMuon can be read as:

> Use the direction subspaces of the update, but give the update the relative
> spectrum of the current parameter matrix.

If $W_t$ has fewer available singular values because of shape conventions, use the
first $r_M$ singular values from the thin SVD. If needed, clamp:

$$
\widehat{\sigma}_i \leftarrow \mathrm{clip}(\widehat{\sigma}_i, a, b)
$$

for numerical stability.

Pseudocode:

```text
for each matrix parameter W:
    B <- mu * B + G
    M <- G + mu * B              # or B without Nesterov

    U_M, Sigma_M, V_M^T <- svd(M)
    U_W, Sigma_W, V_W^T <- svd(W)

    r <- min(number of singular values of M, number of singular values of W)
    scale <- sqrt(mean(Sigma_W[1:r]^2))
    Sigma_hat <- Sigma_W[1:r] / (scale + eps)

    D <- U_M[:,1:r] * diag(Sigma_hat) * V_M[1:r,:]^T
    W <- W - lr_t * shape_scale * D
```

## 3. Random Spectra

Random spectra keep the singular vectors of $M_t$ but replace the singular values
with sampled positive values.

The most controlled default should be uniform random singular values normalized to
the same average energy as Muon:

$$
z_i \sim \mathrm{Uniform}(0,1),
$$

$$
\widehat{\sigma}_i =
\frac{z_i}{\sqrt{\frac{1}{r}\sum_{j=1}^r z_j^2} + \epsilon}.
$$

Then:

$$
D_t = U_t \widehat{\Sigma}_t V_t^T.
$$

This makes the random spectrum positive and scale-controlled:

$$
\frac{1}{r}\sum_{i=1}^r \widehat{\sigma}_i^2 \approx 1.
$$

Alternative random distributions can be ablations, but they should be named
explicitly:

- `random_uniform`: $z_i \sim \mathrm{Uniform}(0,1)$.
- `random_lognormal`: $z_i \sim \mathrm{LogNormal}(0,s^2)$.
- `random_permutation`: randomly permute the original singular values of $M_t$.

For the first implementation, use `random_uniform` because it is simple, positive,
bounded before normalization, and easy to reproduce with a seed.

Pseudocode:

```text
U, Sigma, V^T <- svd(M)
z_i <- Uniform(0, 1) for i = 1..r
z <- z / (sqrt(mean(z^2)) + eps)
D <- U * diag(z) * V^T
W <- W - lr_t * shape_scale * D
```

## 4. Inverted Spectra

Inverted spectra keep the singular vectors of $M_t$ but reverse which directions
receive large singular values.

The safest definition is rank-order inversion, not reciprocal inversion.

Let:

$$
\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r.
$$

Define:

$$
\widehat{\sigma}_i = \sigma_{r+1-i}.
$$

Then normalize to match Muon-scale average energy:

$$
\widehat{\sigma}_i
\leftarrow
\frac{\widehat{\sigma}_i}
{\sqrt{\frac{1}{r}\sum_{j=1}^r \widehat{\sigma}_j^2} + \epsilon}.
$$

The update is:

$$
D_t = U_t \widehat{\Sigma}_t V_t^T.
$$

This means the top singular vector of $M_t$ receives the smallest original
singular value, while the weakest singular direction receives the largest original
singular value.

Avoid defining inverted spectra as $1/\sigma_i$ for the main experiment. Reciprocal
inversion can explode when small singular values are near zero. If tested, it
should be a separate ablation:

$$
\widehat{\sigma}_i =
\frac{1}{\sigma_i + \epsilon},
$$

followed by aggressive normalization and clipping.

Pseudocode:

```text
U, Sigma, V^T <- svd(M)
Sigma_hat <- reverse(Sigma)
Sigma_hat <- Sigma_hat / (sqrt(mean(Sigma_hat^2)) + eps)
D <- U * diag(Sigma_hat) * V^T
W <- W - lr_t * shape_scale * D
```

## Summary Table

| Method | Singular vectors | Singular values |
| --- | --- | --- |
| Muon | from $M_t$ | all ones |
| DynMuon | from $M_t$ | $\sigma_i(M_t)^{p_t}$ |
| DynMuon-Route | from $M_t$ | $\sigma_i(M_t)^{p_{t,l}}$ |
| RelMuon | from $M_t$ | normalized $\sigma_i(W_t)$ |
| Random spectra | from $M_t$ | normalized random positive values |
| Inverted spectra | from $M_t$ | normalized reversed $\sigma_i(M_t)$ |
