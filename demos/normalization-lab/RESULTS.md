# Normalization vs. the weight-constraining alternatives — measured

Lecture 3 claims that the four ways of forcing a norm requirement on the
**parameters** (projection, regularizer, constraint, reparameterization) are not
what people use, and that normalization — which acts on the **activations** —
is. This is the experiment behind that claim, and the data for the demo site.

Run it with the course package's own interpreter, on the course package's own
MNIST (read-only; nothing is downloaded):

```bash
P="/Users/ruoyusun/Desktop/重要研究项目_2022起/0_临时_Teaching 教学/\
2026_Fall_ME2605_DDA6204_LLM_training_当前课程/Week 1/Lab Session/ME2605_Week1_Lab_Package"
"$P/.venv/bin/python" experiment.py --feature-scale 2.0 --layers 16 \
    --steps 1200 --every 300 --lr 0.05 --out results/arm_illscaled_raw.json
```

## The seven arms

Model and optimizer are identical across arms; only the mechanism changes.
MLP, no bias, He init (`W ~ N(0, 2/fan_in)`), ReLU, plain SGD, 16 hidden layers
of width 128, batch 128, 20 000 MNIST training images, 2 000 test images.

| arm | what it does |
|---|---|
| `none` | He init only |
| `ln` | LayerNorm before each activation |
| `bn` | BatchNorm before each activation |
| `rms` | RMSNorm before each activation |
| `proj` | **projection**: after every step, rescale each weight row to ‖w‖ = √2 |
| `reg` | **regularizer**: add λ Σ (‖w_row‖ − √2)² to the loss |
| `wn` | **reparameterization**: W = √2 · v/‖v‖ row-wise (WeightNorm) |

√2 is not arbitrary: it is the row norm He initialization already produces, so
the three weight-side arms **start at a point that already satisfies their own
constraint**. They are not handicapped by their initialization.

## Result

Validation accuracy after 1200 steps:

| input | `none` | `ln` | `bn` | `rms` | `proj` | `reg` | `wn` |
|---|---|---|---|---|---|---|---|
| **plain MNIST** | 0.933 | 0.945 | 0.928 | 0.935 | 0.935 | 0.939 | 0.929 |
| **ill-scaled features** | **NaN** | **0.872** | **0.882** | **0.883** | **0.104** | **NaN** | **NaN** |
| **ill-scaled features, then standardized** | 0.936 | 0.929 | 0.932 | 0.935 | 0.925 | 0.932 | 0.928 |

"Ill-scaled" means each of the 784 input features is multiplied by a fixed
log-uniform factor spanning 10⁻² … 10² — the incoherent feature scales of the
condition-number motivation, applied to MNIST.

Three things to read out of it:

1. **On well-scaled input there is nothing to see.** All seven arms land in
   0.93–0.95. Normalization is not magic when the problem is already easy, and
   the demo says so.
2. **On ill-scaled input the four weight-side mechanisms fail exactly like the
   baseline** — projection reaches 0.10, the regularizer and the
   reparameterization diverge. Normalizing the activations is what survives
   (0.87–0.88). This is the comparison the lecture needs.
3. **Standardizing the input also fixes it** — because that *is* the same idea
   applied at layer 0, which is precisely what the motivation says. Having done
   it, the layer-wise versions no longer buy anything.

The mechanism is visible in the same runs. Forward pre-activation scale
(√mean of h²) at the first and last hidden layer:

| arm | fwd[0] | fwd[L] |
|---|---|---|
| `none` | NaN | NaN |
| `ln` | 0.998 | 1.019 |
| `bn` | 0.980 | 1.055 |
| `rms` | 1.003 | 1.017 |
| `proj` | 115.9 | 3.97 |
| `reg` | NaN | NaN |
| `wn` | NaN | NaN |

The three normalization arms hold the scale at 1.00 by construction. The
weight-side arms do not touch it at all — `proj` even amplifies it 116× at the
first layer, because constraining rows to a fixed norm says nothing about the
variance of the activations those rows produce.

## What this does *not* show

- **It does not show LayerNorm > BatchNorm.** On plain MNIST they are within
  1 point of each other and the ordering flips with the seed and the step size.
  The step-size behaviour does separate them: at L=32 with lr = 0.05 the plain
  net, LayerNorm and RMSNorm all collapse while BatchNorm still reaches 0.90 —
  BatchNorm buys robustness to a large step at the price of depending on the
  batch. Worth a slide; not worth over-claiming.
- **It does not rescue depth 64.** At L=64 even the normalization arms sit near
  0.12–0.20. Pinning the forward scale does not repair the backward signal,
  which is what the lecture says and what the residual connections of the next
  lecture are for.

## Reproducing the other arms

```bash
# plain MNIST
"$P/.venv/bin/python" experiment.py --layers 16 --steps 1200 --every 300 \
    --lr 0.05 --out results/arm_plain_mnist.json
# ill-scaled, then standardized
"$P/.venv/bin/python" experiment.py --feature-scale 2.0 --standardize-input \
    --layers 16 --steps 1200 --every 300 --lr 0.05 \
    --out results/arm_illscaled_standardized.json
```

---

# The in-page replica: the four settings, and why they are these ones

`index.html` runs its **own** training loop (vanilla JS, no server, no network).
The student picks one of four settings, picks one mechanism, and presses Train.
The page's job is to make one contrast legible: **on this data, plain ReLU + He
does not train, and adding LayerNorm does.**

That is a claim about the replica, not a consequence of the code compiling — so
it was checked against the page's own maths before shipping. Constants:

| | |
|---|---|
| inputs | 48; each scaled by a fixed log-uniform factor over `10⁻² … 10²` |
| data | 1 536 train / 512 test, linear teacher, labels `s ≥ 0` |
| net | ReLU, **zero-init biases on every layer** (learned), He init `W ~ N(0, 2/fan_in)` |
| optimisation | plain SGD, batch 24, η = 0.05, 300 steps |
| evaluation | 256 test samples every 20 steps |

**The four settings** (all width 64, varying depth), measured on the page's own
seeds and re-measured on three further seed sets:

| setting | plain (ReLU + He) | + LayerNorm |
|---|---|---|
| L = 8,  d = 64 | diverges, gradient non-finite at step 6 | **0.035** |
| L = 12, d = 64 | diverges at step 5 | **0.035** |
| L = 16, d = 64 | diverges at step 4 | **0.035** |
| L = 20, d = 64 | saturates to Inf in 10 of 20 layers, sits at 0.559 (chance) | **0.078** |

The verdict holds on all four settings across four independent seed sets
(data / initialisation / minibatch stream): plain fails every time, LayerNorm is
at or below 0.15 every time.

### What the sweep ruled out, and what it fixed

- **Width alone is not the axis.** A *depth* ladder is what separates the two
  arms; at fixed depth the effect is much weaker. Narrow nets (d = 8, 10, the
  "d = 10, L = 50" shape) fail for a different reason — capacity — and LayerNorm
  does not rescue them either, so they are not shipped as settings.
- **The step size had to be measured, not guessed.** At η ≤ 0.02 the *plain* net
  starts to succeed — a small enough step survives the bad conditioning — which
  destroys the contrast. η = 0.05 is inside the window where plain always fails;
  η ≥ 0.5 breaks LayerNorm's own stability at depth.
- **The comparison needed zero-initialised biases.** Without them a deep
  no-bias net is structurally handicapped and *both* arms fail from L ≈ 16, so
  there was no depth range in which the claim was true. Adding learned biases
  (a thing a real network has) opened L ≈ 8…20.
- **LayerNorm does not rescue unbounded depth.** L = 24 works on 3 of 4 seed
  sets, L = 28 fails on all of them. This matches the MNIST table above: pinning
  the forward scale does not repair the backward signal. The lab stops at L = 20
  rather than shipping a setting whose caption would be false.

### Honest notes on the other two mechanisms

- **The regularizer behaves like the baseline**: it diverges on all four
  settings, exactly as plain does.
- **Gradient projection does *not* collapse on this replica** — it reaches
  0.063 / 0.066 / 0.086 / 0.211 across the four settings. The MNIST table shows
  it collapsing (0.104). A 48-input, 64-wide, 300-step replica is not the same
  experiment; the table is the one the lecture quotes. This discrepancy is
  stated on the page rather than tuned away.

### Reproducing the page's numbers

The page's numeric block (RNG, data, forward/backward, the mechanisms) has no DOM
in it, so it can be lifted out and driven in Node:

```bash
python3 - <<'PY'
s = open('index.html', encoding='utf-8').read()
a = s.index('function mulberry32(a){')
b = s.index('   2. the page')
open('/tmp/core.js','w').write(s[a:s.rindex('/* ====', 0, b)])
PY
node -e "eval(require('fs').readFileSync('/tmp/core.js','utf8')); /* drive runBatch / sgdStep */"
```

Two checks are worth re-running after any change to the core:

1. **finite-difference gradient check** over every parameter — weights, the
   per-layer biases, and the LayerNorm `γ` / `β` — judged as
   `|num − ana| / max(1e-6, |num| + |ana|)`, and counting only entries whose
   analytic gradient is above `1e-6`. A ReLU unit whose gate is shut has an
   analytic gradient of exactly 0 while a finite difference still "wakes" it once
   `h` exceeds the margin; without that filter a correct backward pass reads as a
   failure. Worst relative error on the shipped core: **2.5e-6**.
2. **the design assertion** — for all four settings, plain must fail and
   LayerNorm must land at or below 0.15.
