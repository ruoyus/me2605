# Condition Number Lab — ME2605 Lecture 4, Part 2

Live: <https://ruoyus.github.io/me2605/demos/condition-number-lab/>

One page, no build step, no network: the 17-house regression of the lecture, with the
condition number of `X Xᵀ` computed as you move the scale slider, and gradient descent
run live so the two step counts can be compared with the theorem's prediction.

## What the page computes

- `X` is 2 × 17: first row the living areas times the chosen scale, second row all ones;
  the target is the price divided by ten, exactly as the lecture scales it.
- The eigenvalues of `X Xᵀ` come from the closed form for a 2 × 2 matrix, so `κ` is exact
  to double precision — no iteration, nothing to converge.
- `κ` is constant along the run (the model is linear, so the Hessian does not move). The
  page says so, because the last bullet of the lecture is that inside a network it does
  move, which is why the layer acts at every layer.
- The prediction uses the bound part 1 states: **r ≥ (κ+1)/2 · ln(1/ε)** on the *relative
  gap* `(F − F*)/(F₀ − F*)`. The measurement uses the same yardstick; using `(F − F*)/F*`
  instead puts the two numbers on different scales and makes the run look slower than the
  bound, which is wrong, not interesting.

## The data

Reconstructed from the professor's own `fitting1.png` (2014). The marker positions were
extracted from the plot pixels, then the six values published on the data slide were
snapped back in:

```
area (ft²) : 1030 1100 1101 1139 1210 1572 2280 2490 2576 3210 3210 3241 3440 4340 5000 5719 6320
price (1000$) : 95  120  287  141  100  167  195  205  227  290  410  345  355  480  405  567  545
```

With this data the two scalings reproduce the numbers on the slides:

| areas × | λ<sub>max</sub> | λ<sub>min</sub> | κ | predicted steps to ε = 10⁻⁴ | measured |
|---|---|---|---|---|---|
| 0.01   | 1.86 × 10⁴ | 4.11 | 4,530 | 20,867 | 8,442 |
| 0.0001 | 18.45      | 0.415 | 44.4 | 210 | 171 |

`F* = 267.86` in both cases: the two runs solve the same problem, and only the speed
changes. After the lecture's 1,000 steps the first run is at 2,944, i.e. 11× above `F*`.

The 2014 slide's λ pair for the first try, (0.0004, 1.856), does not come from any scaling of
this data — a 100× rescaling of the area column changes κ by ≈100 here, not by 10⁴ — so the
figures and the demo use the computed pair above. The slide's κ for the same two runs
(4,640 and 46) agrees with the computed 4,530 and 44.4 to within 3%, and the lecture now
quotes the computed pair on the diagnosis page and in this table.

## Regenerating the figures from the same data
`Lecture_3_SP_2026/build/make_house_figs.py` writes `objective_bad/good.png`,
`fit_bad/good.png` and `house_data.png` into `build/assets/`, and prints the table above.

## Checking the maths
The numeric core sits in one block marked `numeric core (no DOM)` in `index.html`. Extract
it and run it under Node:

```bash
python3 - <<'PY'
s = open('index.html').read()
a = s.index('<script>') + len('<script>')
b = s.index('/* ======================= charts')
open('/tmp/core.js', 'w').write(s[a:b] + "\nmodule.exports = {stats, optimum, loss, grad, run, predicted, stepsTo};\n")
PY
node -e 'const M=require("/tmp/core.js");
  console.log(M.stats(0.0001).kappa, Math.hypot(...M.grad(0.0001, M.optimum(0.0001))));'
```

Checked: the gradient against a central difference (worst relative error 3.5 × 10⁻⁹), the
gradient at the computed optimum (1.3 × 10⁻¹², so the normal equations are right), and both
step counts above.
