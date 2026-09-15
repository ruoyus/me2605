# Demo — Initialization Lab (Lec 2 · Part 2)

Part 2 of the ME2605 / DDA6204 Week 3 demos, and the **exploration before the
answer**: nothing on the page names the principle, the hyper-parameter that
decides it, or the classical initializations. Finding a setting that works is the
exercise.

## What the student does

Six scenarios, each fixing the depth and the width. Every one of them opens on a
setting that **does not train**, and the job is to get the **training error**
below that scenario's target:

| scenario | hidden layers | width | target training error |
|---|---|---|---|
| S1 | 2 | 10 | < 15 % |
| S2 | 4 | 10 | < 12 % |
| S3 | 6 | 10 | < 12 % |
| S4 | 4 | 16 | < 10 % |
| S5 | 6 | 16 | < 12 % |
| S6 | 8 | 16 | < 20 % |

Clearing all six puts a congratulation banner on the page, at which point the
class raises hands. Progress is remembered in the browser (`localStorage`); the
**reset progress** button clears it.

## What is tunable, and what is not

The page is deliberately a **two-instrument** search.

- **One initialization for the whole network** — a family
  (`N(μ, σ²)`, `U(μ, σ²)`, constant, 0/1) plus a mean and a **variance** on one
  logarithmic slider. Every hidden layer *and* the output layer share it, so
  there is exactly one place to tune. This is the dial that decides whether
  training works, and it is also the answer the page is hiding.
- the learning rate and the momentum, on the same panel.

Fixed: batch 32, 900 SGD steps, no gradient clipping. The activation, the random
seed and the weight draw are switchable, but they are not the point.

The **only feedback is the training error and the test error** — no per-layer
readout, no histogram, no spectrum. Those two curves are the whole instrument,
which is what makes the finding the student's own rather than a checklist item.

## The shipped default does not work — on purpose

The page opens on `N(0, 1e-4)`, i.e. the familiar "randn times 0.01". That is a
real choice a beginner makes, not a straw man: measured on this exact code it
leaves **all six** scenarios at exactly 50 %, the chance level for this data set.
The variance slider starts at the bottom of its range, so the first move has to
be to raise it.

The scenario targets were not guessed. Sweeping the initialization variance
against the learning rate on this exact code, every target is comfortably
reachable — the best setting found for each scenario lands between **0.0 % and
1.0 %** training error, well inside its target. The targets are deliberately
generous so that the exercise is *find something that works*, not *find the
optimum*.

## Data and model

- two interleaved spirals, 400 train / 400 test, generated in the page from a
  fixed seed — nothing is downloaded
- ReLU by default; `tanh` and `linear` also selectable
- biases start at 0 and are learned; the initialization concerns weights only
- training runs in the browser; the last five runs keep their curves, so one
  change can be compared directly
- the decision boundary is drawn for the most recent run

## Deep links

```
?sc=1..6&act=relu|tanh|linear&seed=1..5&lr=-1.30&mom=0.9&init=g:mu:log10var&run=1
```

- **`sc` is 1–6** — the scenario *number*, not an index.
- **`lr` is a log10 exponent**, not a learning rate: it is clamped to
  `[-2.3, -0.3]` and fed through `10^`, matching the slider. The default
  `lr=-1.30` is `0.05`.
- `init` takes `g` / `u` (Gaussian / uniform: mean, then log10 variance), `c`
  (a constant value), or `b` (0/1 with probability `p`). The shipped default is
  `g:0:-4`, i.e. `N(0, 10⁻⁴)`.
- `run=1` starts training immediately — useful in a lecture where the page is
  opened six times in a row.

## Neighbours

- `../gd1d-lab/` — the one-dimensional step-size warm-up that comes first
- `../backprop-lab/` — Part 1
- `../nn-signal-lab/` — Part 3, which *answers* the question this page asks
