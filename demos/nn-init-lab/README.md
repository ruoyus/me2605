# Demo 1 - Initialization Lab

Part 1 of the ME2605 / DDA6204 Week 3 demos. Used *before* the theory is
introduced: nothing on the page mentions what the subject of Part 2 is. The
student picks the initialization of **every layer**, plus the learning rate, and
the only feedback is the training / test error.

- up to 8 hidden layers + the output layer, each configured independently
- four families per layer:
  - **constant** -- every entry equal to one tunable number
  - **0/1** -- every entry 0 or 1, tunable fraction of ones
  - **N(mu, sigma^2)** -- i.i.d. Gaussian, tunable mean and variance
  - **U(mu, sigma^2)** -- i.i.d. uniform, tunable mean and variance
- the learning rate (and momentum) sit in the same panel, so the initialization
  is tuned together with the rest of the training setup
- a 12x12 heat map plus the realized mean and variance of each layer's matrix
- data: **two interleaved spirals**, 400 train / 400 test, generated in the page
- train in the browser; the last five runs are kept so one change can be compared
- biases are initialized to zero and learned (the initialization is about weights only)

Part 2 is at `../nn-signal-lab/`.

Deep links:
`?d=24&L=6&act=relu&seed=1&lr=-1.30&mom=0.90&init=g:0.00:-1.20|...&out=g:0.00:-1.38&run=1`
(`g`/`u` encode mean and log10-variance, `b` the fraction of ones, `c` the constant.)

Measured on the shipped defaults (d=24, L=6, ReLU, lr=0.05, 900 steps, 5 seeds),
mean test error, all layers sharing one Gaussian variance:

| variance | <=0.01 | 0.02 | 0.031 | 0.042 | 0.0625 | 0.09 | 0.125 | 0.25 | >=0.5 |
|---|---|---|---|---|---|---|---|---|---|
| test error | 50% | 31% | 6.5% | 1.4% | 1.5% | 1.3% | 1.4% | 23% | 50% / diverged |

The naive `N(0, 1)` default diverges on every seed; the 0/1 and constant families
cannot work at this width at all (variance capped at 0.25, and both are rank-1).
