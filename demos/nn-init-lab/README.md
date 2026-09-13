# Demo 1 - Initialization Lab

Part 1 of the ME2605 / DDA6204 Week 3 demos. Used *before* signal propagation is
introduced: the student picks the initialization of **every layer** and the only
feedback is the training / test error.

- up to 8 hidden layers + the output layer, each configured independently
- three families per layer:
  - **0/1** -- every entry is 0 or 1, tunable fraction of ones
  - **N(mu, sigma^2)** -- i.i.d. Gaussian, tunable mean and variance
  - **U(mu, sigma^2)** -- i.i.d. uniform, tunable mean and variance
- a 12x12 heat map plus the realized mean and variance of each layer's matrix
- train on two moons in the browser; train / test **error** curves, and the last
  five runs are kept so one change can be compared directly
- no activation or gradient dashboard on this page, on purpose

Part 2 (with the signal-propagation dashboard) is at `../nn-signal-lab/`.

Deep links: `?d=16&L=6&act=relu&seed=1&init=g:0.00:-1.20|...&out=g:0.00:-1.20&run=1`
(`g`/`u` encode mean and log10-variance, `b` encodes the fraction of ones).
