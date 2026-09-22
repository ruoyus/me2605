# deep-train-lab — measured results

Numbers below were produced by running the page's own code (the same `makeMoons` /
`makeNet` / `sgdStep` in `index.html`) outside the browser, and again through the page
itself with a DOM stub. They are what the lab will show; do not write any number into
`index.html` that is not in this file.

**Metric (final form, 2026-09-22).** Dashboard A plots the per-layer **RMS**

```
RMS_l = sqrt( (1 / (n · d_l)) · Σ_k Σ_j  z_{l,j}[k]² )        index 0 = the input x
```

the root mean square of every entry of the post-activation `z_l`, over all n data
points. This is the quantity Claim 1 is about: `(1/d_out) E‖Wx‖²` is exactly
`E[RMS(Wx)²]`, so the claim reads *the expected squared RMS is preserved*. **A norm is a
different number** — the L2 norm of one sample is `√d_l · RMS_l`, so a healthy layer
would read ≈ `√d_l` (≈ 5 at d = 24), not ≈ 1. The lab must not plot the norm: then the
curve starts ≈ √(d/d_in) above the input-scale reference line, and nothing in the deck
would match it. The pre-activation `h_l` is not computed or drawn.

Code checked against a brute-force recomputation of `RMS_l` on the same net:
**max difference 5.2 · 10⁻⁹** (d = 24, L = 20, seed 3).

**Fixed throughout:** two moons, noise 0.13, 400 train / 400 test · no bias, no
normalization, no skip connections · mini-batch SGD, batch 32, lr 0.03, momentum 0.9,
global gradient clipping at norm 1 · width d = 24.

## How far the initial signal gets, d = 24, seed 1 (no training)

`RMS_l / RMS_0`, where `RMS_0 = 0.7594`:

| gain g | activation | L = 4 | L = 10 | L = 20 | L = 50 | L = 100 | L = 200 |
|--------|-----------|-------|--------|--------|--------|---------|---------|
| √2 | ReLU (He)    | 0.61 | 0.29 | 0.042 | 0.071 | 7.5 · 10⁻⁴ | 5.6 · 10⁻⁶ |
| 1  | tanh (LeCun) | 0.37 | 0.20 | 0.15  | 0.045 | 7.9 · 10⁻³ | 5.9 · 10⁻⁵ |
| 1  | linear       | 0.71 | 0.46 | 0.42  | 0.16  | 3.0 · 10⁻² | 2.2 · 10⁻⁴ |

A well-scaled net keeps the curve flat near the input's own RMS — that is what
`RMS_l ≈ 1` means once the input is normalised to unit per-entry variance (here the
input's RMS is 0.76, so "preserved" means ≈ 0.76, and the grey reference line sits there).

## ReLU, gain g = √2 (He), 400 steps, d = 24

| L  | RMS_L / RMS_0 before training | loss (start → end) | test error |
|----|-------------------------------|--------------------|------------|
| 4  | 0.61                          | 0.695 → 0.254      | 12.3 %     |
| 10 | 0.29                          | 0.621 → 0.245      | 12.5 %     |
| 20 | 0.042                         | 0.695 → 0.254      | 13.3 %     |
| 100| 7.5 · 10⁻⁴                     | 0.693 → 0.456      | 50.0 %     |
| 200| 5.6 · 10⁻⁶                     | 0.693 → 0.693      | 50.0 %     |

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

| setting | RMS_L / RMS_0 before training | final loss | test error |
|---------|-------------------------------|-----------|------------|
| ReLU, g = √2 | 3.7 · 10⁻⁷ … 5.8 · 10⁻³, **median 6.1 · 10⁻⁵** | 6 seeds stay at 0.693, 6 dip to ≈ 0.46 | **50.0 % for all 12** |
| tanh, g = 1  | 5.9 · 10⁻⁵ … 3.0 · 10⁻², **median 1.0 · 10⁻³** | 0.23 … 0.39 | 5.5 % … 11.3 %, median 10.5 % |

Seed 1 is the lab's default, and it is the number the page's caveat quotes:
**5.6 · 10⁻⁶ (ReLU)** and **5.9 · 10⁻⁵ (tanh)**.

⚠️ This is the reason the verdict box makes **no claim that a small initial signal stops
training**: for tanh the initial signal at L = 200 is only ~10× larger than the ReLU
case, and training still works. ReLU dies because φ′(h) = 0 wherever h ≤ 0, so a network
that starts dead stays dead; tanh has φ′ ≈ 1 near 0 and revives during training.

## After training the top-layer signal comes back (this is not a bug)

At d = 16, L = 105, ReLU, g = √2, seed 3 the top-layer RMS runs `5.2 · 10⁻⁴` before
training and **0.74** after — it recovers to the input's own RMS (0.76). Training repairs
the layers that the initialisation had killed. On a shallower, wider net the trained net
overshoots a little: trained `RMS_L` at L = 45 is 0.66 (d = 16), 2.98 (d = 32), 3.29
(d = 64) — a few times the input level, which is what a ReLU net looks like once it has
sharpened its decision boundary. **Not an explosion**, and not a bug.

## Cost (node, single core; the browser is the same order)

| L  | d  | 400 steps |
|----|----|-----------|
| 20 | 24 | 0.6 s     |
| 100| 24 | 2.8 s     |
| 200| 24 | 5.6 s     |

"Compare shallower / deeper" runs up to three nets and clamps d to 32; at L = 20 it
costs about 3 s (L = 4 / 20 / 100).
