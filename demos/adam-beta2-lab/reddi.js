/* Reddi et al. (ICLR 2018) stochastic counterexample, used in Lab 3 panel 1.
   x ∈ [−1, 1];  g_t = C with probability p = (1+δ)/(C+1), else −1.  E[g] = δ > 0, so x* = −1.
   Adam (projected): m ← β1 m + (1−β1) g ; v ← β2 v + (1−β2) g² ; x ← Π(x − α_t m̂/(√v̂ + ε)). */
(function (root) {
'use strict';
function rng(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function run(o) {
  var r = rng(o.seed || 1), C = o.C, p = (1 + o.delta) / (C + 1), x = o.x0 == null ? 0 : o.x0, m = 0, v = 0, b1 = o.beta1, b2 = o.beta2, eps = 1e-8;
  var xs = [x], sumStep = 0, sumG = 0, sB = 0, nB = 0, sS = 0, nS = 0, sgd = o.sgd, xsS = [x], xS = x;
  for (var t = 1; t <= o.T; t++) {
    var g = r() < p ? C : -1;
    m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g;
    var mh = m / (1 - Math.pow(b1, t)), vh = v / (1 - Math.pow(b2, t));
    var dir = mh / (Math.sqrt(vh) + eps);              /* Adam moves by −α·dir */
    var a = o.alpha / (o.decay ? Math.sqrt(t) : 1);
    x = Math.max(-1, Math.min(1, x - a * dir));
    if (sgd) xS = Math.max(-1, Math.min(1, xS - a * g));
    if (t > o.T / 2) { sumStep += -dir; sumG += -g; if (g > 0) { sB += -dir; nB++; } else { sS += -dir; nS++; } }    /* stationary averages (second half) */
    if (t % (o.every || 10) === 0) { xs.push(x); if (sgd) xsS.push(xS); }
  }
  return { xs: xs, x: x, xsSgd: sgd ? xsS : null, xSgd: xS, stepBig: nB ? sB / nB : 0, stepSmall: nS ? sS / nS : 0, meanStep: sumStep / (o.T / 2), meanNegGrad: sumG / (o.T / 2), p: p, expectedGrad: p * C - (1 - p) };
}
var api = { run: run };
if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Reddi = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
