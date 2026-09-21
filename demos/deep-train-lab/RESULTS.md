# deep-train-lab — measured results

Numbers below were produced by running the page's own code (the same `makeMoons` /
`makeNet` / `sgdStep` in `index.html`) outside the browser, and again through the page
itself with a DOM stub. They are what the lab will show; do not write any number into
`index.html` that is not in this file.

**Fixed throughout:** two moons, noise 0.13, 400 train / 400 test · no bias, no
normalization, no skip connections · mini-batch SGD, batch 32, lr 0.03, momentum 0.9,
global gradient clipping at norm 1 · width d = 24 · seed 1 unless stated.

## ReLU, gain g = √2 (He), 400 steps

| L  | signal at layer L / input, **before** training | loss (start → end) | test error |
|----|-----------------------------------------------|--------------------|------------|
| 4  | –                                             | 0.695 → 0.254      | 12.3 %     |
| 10 | –                                             | 0.621 → 0.245      | 12.5 %     |
| 20 | 0.042                                         | 0.695 → 0.254      | 13.3 %     |
| 100| –                                             | 0.693 → 0.456      | 50.0 %     |
| 200| 5.6 · 10⁻⁶                                    | 0.693 → 0.693      | 50.0 %     |

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

## tanh, gain g = 1 (LeCun), 400 steps, d = 24

| L  | signal at layer L / input, **before** training | loss (start → end) | test error |
|----|-----------------------------------------------|--------------------|------------|
| 20 | –                                             | 0.693 → 0.332      | 9.5 %      |
| 100| 0.008                                         | 0.693 → 0.338      | 9.8 %      |
| 200| 5.9 · 10⁻⁵                                    | 0.693 → 0.343      | 9.5 %      |

⚠️ This is the reason the verdict box in the lab makes **no claim about training**:
for tanh the initial signal at L = 200 is just as dead (6 · 10⁻⁵) as the ReLU case, and
training still works. ReLU dies because φ′(h) = 0 wherever h ≤ 0, so a network that
starts dead stays dead; tanh has φ′ ≈ 1 near 0 and revives during training.

## Cost (node, single core; the browser is the same order)

| L  | d  | 400 steps |
|----|----|-----------|
| 20 | 24 | 0.6 s     |
| 100| 24 | 2.8 s     |
| 200| 24 | 5.6 s     |

"Compare shallower / deeper" runs up to three nets and clamps d to 32; at L = 20 it
costs about 3 s (L = 4 / 20 / 100).
