# Demo 1 - Backprop Lab

Part 1 of the ME2605 / DDA6204 Week 3 demos, used in the **first** part of the
lecture: backpropagation itself, before anything about initialization.

One tiny network, one training example, three phases you step through:

1. **forward** - z = Wa + b, a = phi(z), left to right, until p
2. **backward** - every edge gets dL/dw; colour = sign, thickness = size
3. **update** - w <- w - eta dL/dw, then the next example

- `back` / `next phase` / `Auto play` (`?auto=1`), and the phase is always recomputed
  from scratch, so stepping back is exact
- the network diagram shows forward values with their pre-activations, the error
  signal of every node, the gradient of every edge, and every bias
- a **node-by-node table** in the CS231n style: one row per node, forward expression,
  value, then the backward expression and its value, in reverse order
- `?train=N` pre-trains N steps so a prepared lecture link already shows a descent
- data: four clusters labelled by XOR - chosen because a network with `phi = linear`
  can never fit it (the loss stalls exactly on log 2), so the hidden layer has to be
  doing something

Defaults: 2-3-1, tanh, eta = 0.5, 400 steps -> loss 0.04, 0% training error.
(The 0.05 tail of runs - seed 5, say - gets stuck in an XOR local minimum at log 2;
that is real, and it is a useful thing to point at.)

Deep links: `?hl=1&w=3&act=tanh&lr=-0.30&phase=2&reveal=99&train=200&auto=1`

All parameters are 64-bit floats, so the numbers shown agree with a hand
calculation; `backward()` was verified against central finite differences
(worst relative error ~1e-9 for tanh/sigmoid/linear, with the only outliers being
genuine ReLU kinks).
