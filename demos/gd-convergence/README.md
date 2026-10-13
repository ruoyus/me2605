# GD Convergence across Data Distributions

Live: <https://ruoyus.github.io/me2605/demos/gd-convergence/>

Eight datasets drawn from different probability distributions (isotropic and stretched
Gaussians, uniform squares, Laplace clouds, blob-vs-ring, two moons), one fixed optimiser,
run until ‖∇L‖ ≤ 10⁻³ or 10⁶ updates. Non-interactive by design: every curve is computed
offline by `generate.cjs` and embedded in `index.html`, so the page has no runtime compute
and works from a static file.

## The setup

- Model: linear least squares on features (1, x, y) with the ±1 class label as target —
  a strictly convex quadratic, so the Hessian ΦᵀΦ/n never moves and the per-step decay of
  each eigendirection is exactly (1 − λᵢ/L), the setting of the Lecture 5 convergence bound.
- Optimiser: full-batch gradient descent, η = 1/L, L = λ_max(ΦᵀΦ/n) from a power iteration.
- The κ ladder is built by stretching one input coordinate of the same Gaussian blobs by
  10 / 100 / 1000 / 3000: the optimum is unchanged, only the conditioning degrades. The
  measured updates-to-target divided by κ is 4.03 / 4.16 / 4.27 across three decades —
  the constant of the bound r ≥ (κ/2)·ln(1/ε). The ×3000 run does not reach 10⁻³ inside
  10⁶ updates and is drawn dashed.

## Files

- `generate.cjs` — draws the datasets (mulberry32, seed 7), runs GD, writes `curves.json`.
- `template.html` + `curves.json` → `index.html` (the JSON replaces `__CURVES_JSON__`;
  rebuild with a one-line string replace after rerunning the generator).
- `index.html` — the published page.
