# Lec 2 / Part 1 — Backprop Lab

ME2605 / DDA6204, used in the **first hour** of the two-hour lecture: backpropagation
itself, before anything about initialization. Parts 2 and 3 (`nn-init-lab`,
`nn-signal-lab`) are for the second hour.

One small network, one example, three phases:

1. **forward** — `h = Wz + b`, `z = phi(h)`, left to right until `p`
2. **backward** — the error moves back one layer per press:
   `e_l = D_l (W_(l+1))^T e_(l+1)`, starting from `e_L` at the output
3. **update** — one weight layer per press, **from the output layer back to the input
   layer**: `W_l <- W_l - eta e_l (z_(l-1))^T`

## The diagram is the chain of the notes' Fig. 3.2

One block per layer, with two bands: the forward activations along the top
(`z_0, z_1, ...`) and the backward errors along the bottom. The point of the picture is
the walking order — nothing for layer `l+1` is computed before layer `l` is finished in
the forward pass, and nothing for layer `l` before `e_(l+1)` is ready in the backward
one. A layer's gradient is finished the moment that layer's error arrives.

**Above each layer block sit three lines**, so the panel and the picture always agree:

```
W_2  <-  W_2 - eta e_2 z_1^T
W_2   =  [-0.5 1 ; 0.5 1]
W_2'  =  [-0.58 1.08 ; 0.79 0.71]
```

Only the layers the walk has already reached show numbers; the others stay dimmed to the
formula.

## Notation — fixed once, for the whole course

- A **layer index is a subscript**: `W_1, W_2, z_l, h_l, e_l, D_l`. Superscripts are
  reserved for powers and transposes (`2^L`, `(w^7-1)^2`, `ᵀ`).
- A **matrix is written row by row**, rows separated by `;` — `[a b ; c d]`. It is never
  printed as a flat vector.
- A **vector component goes in brackets**: `e_2[1]` is the first entry of `e_2`.
- In the gradient chip `dL/dW_l = e_l (z_(l-1))^T` the `e` is red and the `z` is blue, and
  **two leader arrows follow each one to where it is made**: the red arrow drops from the
  chip to that layer's `e_l` label in the backward band, the blue one runs up the gap
  between blocks to the `z_(l-1)` label in the forward band. That is why the formula is
  written in this order, and the arrows are what makes the panel and the picture one
  object rather than two.

## Why the numbers are simple

The whole design goal is that everything on screen can be checked by hand, with as few
digits as possible.

- Inputs are exactly **+/-1** and weights are drawn from **{+/-0.5, +/-1}**, with every
  bias starting at 0. So `h = Wz + b` is exact: e.g. `h2 = 1*(-1) + 0.5*(-1) + 0 = -1.5`.
- Numbers are shown to **two decimals with trailing zeros stripped**: an exact 1 prints
  as `1`, not `1.00`; `0.5` prints as `0.5`.
- The worked update line is arithmetic you can do in your head:
  `W3[1,1] = 1 - 1 x 0.39 x (-0.88) = 1.34`.
- Default net is 2 -> 2 -> 2 -> 1 (three weight layers) with tanh and `eta = 1`.
- The two-page ledger is collapsed by default: `?reveal=999` on the deep link opens it.
- Exact integers all the way through are impossible with a saturating non-linearity:
  two inputs and integer weights force either `h = 0` (a dead column) or `|h| >= 3`
  where tanh saturates and the gradient collapses. Hence halves, and `phi = linear` is
  offered in the dropdown for anyone who wants fully exact arithmetic.

## The draw is validated, not just random

A candidate draw is rejected unless: the four examples stay distinct at every layer; no
hidden unit is a clone or mirror image of another (rows not proportional); no
pre-activation is near 0 or large enough to saturate; no activation, error or gradient
is tiny; and **every weight changes visibly at the displayed precision** for both
eta = 1 and eta = 0.5. That last condition exists because a weight that appears not to
move reads as a bug, which is the one thing this page must not look like.
Three draws are offered (seeds 1-3); all three pass.

## Controls

- hidden layers 1-3, width 2-4, phi in {tanh, sigmoid, ReLU, linear}, eta in {0.5, 1, 2}
- the four examples as buttons; there is **no training loop and no data plot** on this
  page, on purpose — it is about the mechanics
- `next` walks the phases, then one layer per press (output layer first), then the next
  example; `back` from the update phase **restores the weights exactly** (verified)
- arrow keys / PageUp-PageDown step as well, for use at the lectern

Deep links:

```
?hl=2&w=2&act=tanh&eta=1&seed=1&ex=0&phase=3&step=1&reveal=999
```

`phase` is 1..3; `step` (or `up`) is how many layers are already updated, counted from
the output; `reveal` sets how many rows of the ledger are shown.

## Verification

`backward()` checked against central finite differences: worst relative error ~2e-9 for
tanh / sigmoid / linear with every entry inside 1e-6; the only ReLU outliers are genuine
kinks. The per-layer update reproduces a single SGD step exactly (max difference 0) and
moves every layer. A scripted click-through confirms the phase order, that the updates
persist across an example change, and that `back` restores the weights exactly
(`RESTORE_EXACT=true`).

The **diagram** carries its own test, because the geometry depends on depth and width: a
DOM shim drives the page's real render functions over **324 configurations x every phase
x every step — 3240 renders, 0 failures** — and asserts that no label is ever empty or
`NaN`, that every layer index in the figure is a real subscript, that the update lines
above each block number exactly `3 x L`, and that each matrix is printed with its row
separator. Width 4 with depth 3 is the tightest case and is rendered and looked at by
hand; a tile that will not fit its column is shortened to `...` rather than allowed to
overlap its neighbour.
