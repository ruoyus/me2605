# Lec 2 / Part 1 - Backprop Lab

ME2605 / DDA6204, used in the **first hour** of the two-hour lecture: backpropagation
itself, before anything about initialization. Parts 2 and 3 (`nn-init-lab`,
`nn-signal-lab`) are for the second hour.

One small network, one example, three phases:

1. **forward** - `h = Wz + b`, `z = phi(h)`, left to right until `p`
2. **backward** - every edge gets `dL/dw = e * z`; colour = sign, thickness = size
3. **update** - one weight layer per press, **from the output layer back to the input
   layer**. The outer product `dL/dW = e (z)^T` is written out in full, one entry is
   worked digit by digit, then `W <- W - eta dL/dW`. The layer being updated turns
   green on the diagram and its edges show `old -> new`.

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
- The two-page ledger is collapsed by default: `?open=1` on the deep link opens it.
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
  page, on purpose - it is about the mechanics
- `next` walks the phases, then one layer per press (output layer first), then the next
  example; `back` from the update phase **restores the weights exactly** (verified)
- arrow keys / PageUp-PageDown step as well, for use at the lectern

Deep links: `?hl=2&w=2&act=tanh&eta=1&seed=1&ex=0&phase=3&up=1&open=1&reveal=999`
(`phase` 1..3, `up` = how many layers are already updated, counted from the output).

## Verification

`backward()` checked against central finite differences: worst relative error ~2e-9 for
tanh / sigmoid / linear with every entry inside 1e-6; the only ReLU outliers are genuine
kinks. The per-layer update reproduces a single SGD step exactly (max difference 0) and
moves every layer. A scripted click-through confirms the phase order, that the updates
persist across an example change, and that `back` restores the weights exactly
(`RESTORE_EXACT=true`).
