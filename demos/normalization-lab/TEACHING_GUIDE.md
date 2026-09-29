# Normalization II: classroom lab

The updated `index.html` is the Lecture 4 lab. The previous experiment is preserved in `legacy.html`; `RESULTS.md` and `experiment.py` describe that older experiment, not the new two-moons runs.

## Suggested 20–30 minute sequence

1. **One fixed sample (5 min).** Change Batch A to Batch B. LN's output for Sample 1 stays fixed; training BN's output changes. Then choose B = 1. The centered BN values become zero and the affine output is β, not one. Standard fully connected BN training rejects this case; stored-statistics inference can process a single sample. Point out that computed μ and σ are not learned γ and β.
2. **Gentle baseline (5 min).** Ask whether the unnormalized network can learn. Run all three models with identical initial weights and batch indices. Read the held-out loss, accuracy, and final decision maps. Normalization is not necessary to obtain useful performance in this setting.
3. **Batch-size experiment (5–10 min).** Switch to the small-batch preset, which changes only B from 32 to 2 relative to the baseline. Ask why BN is more sensitive. Compare the two BN inference rules. Same number of updates does not mean the same number of training examples processed: B × updates differs. This caveat applies when comparing across presets, not between models within a run.
4. **Scale/depth experiment (5–10 min).** The stress preset changes both depth and input scale. Do not attribute their combined effect to either one alone. Then change only scale, or only depth, to isolate a factor. Repeat with seeds 11 and 23 and a different learning rate. A method need not win every run.

Optional: turn learned affine parameters off, change placement, or reduce width to 2. This is an MLP before/after-activation convention, not Transformer pre-LN/post-LN. Narrow-layer results are not representative of wide LLM layers.

## Measured checkpoints

300 SGD updates, width 16, learning rate 0.03, post-activation normalization, learned feature-wise γ and β. Values are held-out accuracy using standard evaluation, not fabricated curves. The UI computes these again from scratch.

| Preset | Seed | No normalization | LN | BN (stored statistics) | BN (test-batch diagnostic) |
|---|---:|---:|---:|---:|---:|
| Gentle baseline | 7 | 93.4% | 98.4% | 99.2% | 98.8% |
| Gentle baseline | 11 | 89.1% | 98.0% | 100.0% | 99.6% |
| Gentle baseline | 23 | 92.6% | 96.5% | 98.8% | 98.8% |
| Scale + depth | 7 | 85.2% | 83.2% | 87.1% | 85.9% |
| Scale + depth | 11 | Diverged | 83.2% | 83.2% | 86.3% |
| Scale + depth | 23 | 89.8% | 87.1% | 89.1% | 89.5% |
| Small batch | 7 | 90.2% | 94.9% | 87.1% | 68.8% |
| Small batch | 11 | 84.0% | 96.9% | 79.7% | 67.6% |
| Small batch | 23 | 91.8% | 95.3% | 64.5% | 74.6% |

These are illustrative seeds, not a statistical benchmark. Test batches are deterministically shuffled, not forced to have balanced classes. The held-out set becomes a validation set if used repeatedly to choose hyperparameters. Decision-map dots show 96 held-out points; metrics use all 256. Maps show initialization during training and are replaced by final predictions at completion. Inference and map generation never update running statistics.

## Numerical and browser checks

From this folder:

```sh
node test-engine.cjs
node benchmark.cjs
```

The numerical test checks 286 finite differences covering W, linear bias, feature-wise γ and β for no normalization, LN and BN, in both placements. It also checks shared initialization, the BN running-variance update, immutable evaluation, singleton inference and data scaling. Largest measured absolute gradient discrepancy: 2.11 × 10⁻¹⁰.

Serve the repository root over HTTP on port 8765 and run `node demos/normalization-lab/test-browser.cjs` with Playwright available. Optionally set `CHROME_PATH` to an installed Chrome executable. This checks both the default and small-batch numerical results, matrix controls, stop, JSON export, English/Chinese switching, mobile overflow and browser errors. Browser workers require HTTP; opening `index.html` directly with a file URL is not supported.

## Implementation boundaries

- No external ML runtime, CDN, backend, analytics or downloaded dataset.
- `engine.js`: double-precision forward/backward computations, SGD, synthetic data, evaluation.
- `worker.js`: actual training outside the UI thread; all methods share sampled batch indices.
- `lab.js`, `lab.css`: interactive teaching UI, curves, decision maps and export.
- BN uses ε = 10⁻⁵ and EMA coefficient 0.1. Training forward uses the population batch variance; running variance receives the unbiased estimate. Inference normally uses stored statistics.
- LN uses each sample's current feature statistics in both modes. Neither method learns separate parameters for individual samples.

References: [PyTorch BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html), [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html), [Batch Normalization](https://arxiv.org/abs/1502.03167), [Layer Normalization](https://arxiv.org/abs/1607.06450).
