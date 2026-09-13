# Lec 2 / Part 1 - Backprop Lab

ME2605 / DDA6204, used in the **first hour** of the two-hour lecture: backpropagation
itself, before anything about initialization. Parts 2 and 3 (`nn-init-lab`,
`nn-signal-lab`) are for the second hour and are linked from the header.

One tiny network, one example, three phases you step through:

1. **forward** - `h = Wz + b`, `z = phi(h)`, left to right until `p`
2. **backward** - every edge gets `dL/dw = e * z`; colour = sign, thickness = size
3. **update** - **one weight layer at a time**: the outer product
   `dL/dW = e (z)^T` is written out in full, one entry worked out digit by digit,
   then `W <- W - eta dL/dW`. The layer being updated turns green on the diagram and
   its edges show `old -> new`.

## Why the numbers are hand-checkable

- The inputs are exactly **+/-1** (the four corners of the square, labelled by XOR)
  and the weights are drawn on the grid **{-0.5, -0.25, 0.25, 0.5}**, with every
  bias starting at 0. So `h^(1) = W^(1) x` is a sum of the weights with signs - an
  exact multiple of 0.25 - and the first layer can be checked mentally.
- Everything is printed to **four decimals**, and the node table writes out the
  expansion term by term, e.g.
  `h^(1)_1 = 0.2500*(-1.0000) - 0.5000*(-1.0000) + 0 = 0.2500`.
- The draw is validated: a candidate is rejected unless the four examples stay
  distinct at every layer, no unit is frozen, and **every weight has a non-zero
  gradient on all four examples**. A weight that legitimately does not move (because
  some activation is exactly 0) is therefore never mistaken for a bug.
- Default net: 2 -> 2 -> 2 -> 1, tanh, eta = 0.5 - three weight layers, two hidden
  of width 2, so each `dL/dW` is a 2x2 matrix you can multiply out by hand.

## Controls

- hidden layers 1-3, width 2-4, phi in {tanh, sigmoid, ReLU, linear}, eta, three draws
- the four examples as buttons; **train/test data and any training loop were removed
  on purpose** - this page is about the mechanics, not about convergence
- `next` walks the phases, then one layer per press, then the next example;
  `back` from the update phase **restores the weights exactly** (verified)
- arrow keys / PageUp-PageDown step as well, for use at the lectern

Deep links: `?hl=2&w=2&act=tanh&lr=0.5&seed=1&ex=0&phase=3&up=1&reveal=999&auto=1`
(`phase` 1..3, `up` = how many layers have already been updated).

## Verification

`backward()` checked against central finite differences: worst relative error
~2e-9 for tanh / sigmoid / linear with all entries inside 1e-6; the only ReLU
outliers are genuine kinks (|h| < 1e-3). The layer-at-a-time update reproduces a
single SGD step exactly (max|difference| = 0) and moves every layer.
