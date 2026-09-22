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
