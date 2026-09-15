# Lec 2 / Part 2 — Step-Size Lab

ME2605 / DDA6204. The **warm-up that opens the second hour**: one scalar function, one
update rule, and two things the student chooses. It comes before any theory about depth
or initialization, and it deliberately shows **no picture of the function** — finding out
what goes wrong is the exercise.

## The function

```
F(w) = (w^7 - 1)^2          F'(w) = 14 w^6 (w^7 - 1)
w    <- w - eta F'(w)
```

Smooth, one minimum at `w = 1`, and neither of the two usual suspects is the obstacle.
Yet one fixed step size cannot serve both ends:

| | |
|---|---|
| `F'(2)` | `= 14 * 2^6 * (2^7 - 1) = 113 792` |
| `F'(1/2)` | `~ -0.217` |
| largest stable `eta` at the minimum | `2 / F''(1) = 2 / 98 ~ 0.02` |

so `eta = 0.01` moves `w` by about 1140 in one step from `w = 2` and the run diverges,
while from `w = -1` it creeps into the flat band around `w = 0` and stalls (measured:
it stops at about `-0.235`). The same number that diverges at one end stalls at the
other — that is the whole point of the page.

## What the student does

Two dials and nothing else:

- the **starting point** `w_0`, range `[-1.5, 2.5]`
- the **step size** `eta`, on a logarithmic slider

A run "works" if the iterates reach `w ~ 1`. The page reports the iteration count, the
final value, the distance to 1 and the gradient size along the way; the **attempt log**
keeps every run in the browser (`localStorage`, key `me2605.gd1d.log`) so the class can
compare settings side by side, and there is a button to clear it.

The page **opens on `w_0 = 2`, `eta = 0.01`** — a real choice a beginner makes, and one
that diverges in 2 steps. The step-size slider starts at the bottom of its range, so the
first move has to be to raise it.

The function is drawn **only after the student asks for it** (`?reveal=1` shows it),
because the geometry page in the deck is the first time the class sees the flat band,
the two steep walls and the nice region between them.

## Data

`w_0 = 2, eta = 0.01` diverging in 2 steps, `w_0 = 1/2` converging in 55, `w_0 = -1`
stalling at `-0.235`, and the `0.02` stability threshold are all **measured on this
page's own code**, not estimated.

## Deep links

```
?w0=2&eta=0.01&run=0&reveal=1
```

`w0` is clamped to `[-1.5, 2.5]`; `eta` is clamped to the slider's range
(`10^-5 .. 10^-1.3`). `run=0` renders without starting, `reveal=1` shows the function.

## Neighbours

- `../backprop-lab/` — Part 1, the first hour
- `../nn-init-lab/` — the same question in high dimension: which settings make a real
  network train
- `../nn-signal-lab/` — Part 3, which explains *why* the tuned scale works
