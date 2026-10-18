# Normalization II: classroom lab

The updated `index.html` is the Lecture 4 lab. Page order: the three-model comparison comes first; the fixed-sample review of BN versus LN sits at the bottom under *Review BN and LN*; the old weight-constraint experiment is preserved in `legacy.html`, and `RESULTS.md` and `experiment.py` describe that older experiment. *Models to compare* offers four models — no normalization, LayerNorm, BatchNorm and, switched off by default, weight norm (direction plus one learned magnitude per neuron). The *Scale γ (LN) / g (WN)* switch offers a per-feature scale (vector) or one scale per layer (scalar) and applies to LN and weight norm only; BN keeps its per-feature pair.

## Suggested 20–30 minute sequence

1. **One fixed sample (5 min).** Change Batch A to Batch B. LN's output for Sample 1 stays fixed; training BN's output changes. Then choose B = 1. The centered BN values become zero and the affine output is β, not one. Standard fully connected BN training rejects this case; stored-statistics inference can process a single sample. Point out that computed μ and σ are not learned γ and β.
2. **Gentle baseline (5 min).** Ask whether the unnormalized network can learn. All three models are selected by default in *Models to compare*; run them with identical initial weights and batch indices. Read the test loss, accuracy, and final decision maps. Normalization is not necessary to obtain useful performance in this setting.
3. **Batch-size experiment (5–10 min).** Switch to the small-batch preset, which changes only B from 32 to 2 relative to the baseline. Ask why BN is more sensitive. Compare the two BN inference rules. Same number of updates does not mean the same number of training examples processed: B × updates differs. This caveat applies when comparing across presets, not between models within a run.
4. **Scale/depth experiment (5–10 min).** The stress preset changes both depth and input scale. Do not attribute their combined effect to either one alone. Then change only scale, or only depth, to isolate a factor. Repeat with seeds 11 and 23 and a different learning rate. A method need not win every run.

Optional: use *Models to compare* to switch a model off. It is then not trained at all, so the run is shorter and the curves are sparser; a kept model reproduces the same numbers it has in a full three-model run, because each model draws its own initial weights and batch indices from the same seed and never reads another model's state. Turn learned affine parameters off, change placement, or reduce width to 2. This is an MLP before/after-activation convention, not Transformer pre-LN/post-LN. Narrow-layer results are not representative of wide LLM layers.

## Measured checkpoints

1280 training and 384 test points, laid uniformly per unit arc length along each strand; 600 SGD updates, width 16, learning rate 0.05, post-activation normalization, learned feature-wise γ and β. The data are two spiral strands winding 1¼ turns around a shared centre, deeply interlocked; the best possible threshold on a single input coordinate reaches only about 59% test accuracy (measured 59 / 58 / 59 across seeds 7 / 11 / 23). Values are test accuracy using standard evaluation, not fabricated curves. The UI computes these again from scratch.

| Preset | Seed | No normalization | LN | BN (stored statistics) | BN (test-batch diagnostic) |
|---|---:|---:|---:|---:|---:|
| Gentle baseline | 7 | 97.1% | 100.0% | 100.0% | 99.5% |
| Gentle baseline | 11 | 96.4% | 99.7% | 100.0% | 98.2% |
| Gentle baseline | 23 | 96.4% | 99.2% | 99.7% | 98.2% |
| Scale + depth | 7 | 50.0% | 48.7% | 54.7% | 67.7% |
| Scale + depth | 11 | Diverged | 50.3% | 66.7% | 71.4% |
| Scale + depth | 23 | 50.3% | 52.9% | 65.1% | 70.8% |
| Small batch | 7 | 78.6% | 77.6% | 42.2% | 63.8% |
| Small batch | 11 | 74.7% | 64.3% | 63.0% | 63.0% |
| Small batch | 23 | 75.5% | 74.2% | 42.4% | 45.6% |

Weight norm on the gentle baseline (same settings, 600 updates): 98.7% / 90.4% / 96.6% across seeds 7 / 11 / 23 — usable here, and worth comparing against LN because its per-neuron magnitude plays the role of LN's γ. On B = 2, batch norm can spike mid-training (one seed here reaches a training loss above 4000 before partially recovering), which the loss curves show directly.

Each preset now disqualifies a different method. On the gentle baseline every model is usable. Under depth 8 with one input coordinate 100× larger, no normalization and LN both collapse to near chance while BN stays the most useful: BN standardizes each feature across the batch, so its first layer can undo a per-coordinate input scale, while LN normalizes within a sample and cannot — consistent with the invariance table of Ba et al. (2016): LN is invariant to rescaling a whole sample, not to rescaling a single coordinate. On B = 2 the ranking flips: BN's batch statistics become too noisy and LN is the best normalized method.

These are illustrative seeds, not a statistical benchmark. Test batches are deterministically shuffled, not forced to have balanced classes. The test set becomes a validation set if used repeatedly to choose hyperparameters. Decision-map dots show 128 test points; metrics use all 384. Maps show initialization during training and are replaced by final predictions at completion; the background shading is the model score through a fixed softening (sigmoid of the logit over 3), not the raw probability, so a saturated network still shows a gradient near its boundary. Inference and map generation never update running statistics. A table row is unchanged if the other models are switched off in *Models to compare*; only the number of charts and cards changes.

## Numerical and browser checks

From this folder:

```sh
node test-engine.cjs
node benchmark.cjs
```

The numerical test checks 368 finite differences covering W, linear bias, feature-wise γ and β for no normalization, LN and BN in both placements, plus the scalar-γ option for LN. A centered difference there is compared against the backward pass re-evaluated at v±h, which also covers the case where a sample kills an entire layer and a bias sits exactly on the ReLU kink — there the centered difference is the average of the two one-sided slopes. It also checks shared initialization, the BN running-variance update, immutable evaluation, singleton inference and data scaling. Largest measured absolute gradient discrepancy: 2.10 × 10⁻⁸.

Serve the repository root over HTTP on port 8765 and run `node demos/normalization-lab/test-browser.cjs` with Playwright available. Optionally set `CHROME_PATH` to an installed Chrome executable. This checks both the default and small-batch numerical results, matrix controls, the *Models to compare* picker (a two-model and a one-model run reproduce the per-model numbers of the three-model run; an empty selection trains nothing and reports it), stop, JSON export, English/Chinese switching, mobile overflow and browser errors. Browser workers require HTTP; opening `index.html` directly with a file URL is not supported.

## Implementation boundaries

- No external ML runtime, CDN, backend, analytics or downloaded dataset.
- `engine.js`: double-precision forward/backward computations, SGD, synthetic data, evaluation.
- `worker.js`: actual training outside the UI thread; it trains the models selected in the UI (all three by default), and those share sampled batch indices.
- `lab.js`, `lab.css`: interactive teaching UI, curves, decision maps and export.
- BN uses ε = 10⁻⁵ and EMA coefficient 0.1. Training forward uses the population batch variance; running variance receives the unbiased estimate. Inference normally uses stored statistics.
- Weight norm sets each row to w = g·v/‖v‖ with one learned g per neuron (√2 at init); the scalar switch shares one g per layer. The engine-level test covers it with a training smoke run rather than finite differences.
- LN uses each sample's current feature statistics in both modes. Neither method learns separate parameters for individual samples.

References: [PyTorch BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html), [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html), [Batch Normalization](https://arxiv.org/abs/1502.03167), [Layer Normalization](https://arxiv.org/abs/1607.06450).
