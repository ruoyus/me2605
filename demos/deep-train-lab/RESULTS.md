# deep-train-lab — measured results

Numbers below were produced by running the page's own code (the same `makeMoons` /
`makeNet` / `sgdStep` in `index.html`) outside the browser, and again through the page
itself with a DOM stub. They are what the lab will show; do not write any number into
`index.html` that is not in this file.

**Metric (changed 2026-09-22).** Dashboard A plots the per-layer signal strength
`S_l = (1/n) Σ_k ‖z_l[k]‖` — for each data point take the norm of the whole
post-activation vector, then average over the n points. (It used to plot the per-unit
RMS, `sqrt( (1/(d n)) Σ_{k,j} z_j[k]² )`, which is smaller by about `√d`.) The
pre-activation `h_l` is no longer computed or drawn. The code was checked against a
brute-force recomputation of `S_l` on the same net: max difference 1.2 · 10⁻⁸.

**Fixed throughout:** two moons, noise 0.13, 400 train / 400 test · no bias, no
normalization, no skip connections · mini-batch SGD, batch 32, lr 0.03, momentum 0.9,
global gradient clipping at norm 1 · width d = 24.

## How far the initial signal gets, d = 24, seed 1 (no training)

`S_l / S_0`, where `S_0 = 0.992`:

| gain g | activation | L = 4 | L = 10 | L = 20 | L = 50 | L = 100 | L = 200 |
|--------|-----------|-------|--------|--------|--------|---------|---------|
| √2 | ReLU (He)       | 2.12   | 0.97   | 0.14   | 0.23   | 2.4 · 10⁻³ | 1.8 · 10⁻⁵ |
| 1  | tanh (LeCun)    | 1.36   | 0.73   | 0.57   | 0.15   | 2.6 · 10⁻² | 2.0 · 10⁻⁴ |
| 1  | linear          | 2.50   | 1.63   | 1.49   | 0.49   | 9.3 · 10⁻² | 7.3 · 10⁻⁴ |

⚠️ The ratio is not 1 at a healthy layer: `S_l` is a **vector norm**, so a well-scaled
layer of width d sits near `√(d/d_in) ×` the input level (≈ 3.5 for d = 24, d_in = 2).
The verdict box in the page compares the ratio against that reference, not against 1.

## ReLU, gain g = √2 (He), 400 steps, d = 24

| L  | S_L / S₀ before training | loss (start → end) | test error |
|----|--------------------------|--------------------|------------|
| 4  | 2.12                     | 0.695 → 0.254      | 12.3 %     |
| 10 | 0.97                     | 0.621 → 0.245      | 12.5 %     |
| 20 | 0.14                     | 0.695 → 0.254      | 13.3 %     |
| 100| 2.4 · 10⁻³                | 0.693 → 0.456      | 50.0 %     |
| 200| 1.8 · 10⁻⁵                | 0.693 → 0.693      | 50.0 %     |

Same settings, 500 steps, seeds 1 / 2 / 7 (test error):

| L  | d = 24                     |
|----|----------------------------|
| 10 | 13 % / 13 % / 12 %         |
| 50 | 50 % / 12 % / 50 %         |
| 100| 50 % / 50 % / 50 %         |
| 200| 50 % / 50 % / 50 %         |

So the transition is not a sharp line and it moves with the seed: somewhere between
L = 20 and L = 100 at d = 24 the loss stops coming down. **No slide should quote a
precise frontier** — the lab is where the student finds it.

## L = 200, d = 24, seeds 1 … 12 (the two numbers quoted on the page)

| setting | S_L / S₀ before training | final loss | test error |
|---------|--------------------------|-----------|------------|
| ReLU, g = √2 | 1.2 · 10⁻⁶ … 1.9 · 10⁻², **median 1.7 · 10⁻⁴** | 6 seeds stay at 0.693, 6 dip to ≈ 0.46 | **50.0 % for all 12** |
| tanh, g = 1  | 2.0 · 10⁻⁴ … 1.1 · 10⁻¹, **median 3.6 · 10⁻³** | 0.23 … 0.39 | 5.5 % … 11.3 %, median 10.5 % |

⚠️ This is the reason the verdict box makes **no claim that a small initial signal
stops training**: for tanh the initial signal at L = 200 is only ~20× larger than the
ReLU case, and training still works. ReLU dies because φ′(h) = 0 wherever h ≤ 0, so a
network that starts dead stays dead; tanh has φ′ ≈ 1 near 0 and revives during training.

## After training the signal comes back (this is not a bug)

At d = 16, L = 105, ReLU, g = √2, seed 3: `S_L/S_0` goes from 1.8 · 10⁻³ before
training to 1.79 after — the trained net's top layers are *healthier* than at init.
Training repairs the decayed layers, and a trained ReLU net can overshoot the initial
level. Measured max `S_l` at L = 45: 6.9 (d = 16), 9.9 (d = 32), 21.0 (d = 64) — that is
`O(√d)`, i.e. the healthy level for that width, not an explosion.

## Cost (node, single core; the browser is the same order)

| L  | d  | 400 steps |
|----|----|-----------|
| 20 | 24 | 0.6 s     |
| 100| 24 | 2.8 s     |
| 200| 24 | 5.6 s     |

"Compare shallower / deeper" runs up to three nets and clamps d to 32; at L = 20 it
costs about 3 s (L = 4 / 20 / 100).
