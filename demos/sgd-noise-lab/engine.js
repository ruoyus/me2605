/* Optimization labs engine (ME2605 Lecture 6).
   Adapted from the normalization-lab engine: same seeded RNG, He initialization and
   LayerNorm / BatchNorm code, extended with a softmax output (K classes), Adam, and
   learning-rate schedules. Double precision, no DOM, no external dependencies:
   the same file runs in the browser worker and in Node for the measured tables. */
(function (root) {
"use strict";

/* ---------- random numbers ---------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    var t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gauss(r) {
  var u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ---------- data ----------
   Every dataset is {X: Float64Array(n*nin), y: Int32Array(n), n, nin, K}. */
function spirals(seed, count, turns, noise) {
  /* Two interlocking spiral strands around a shared centre (as in the normalization lab),
     with the number of turns as the complexity knob. Points are uniform per arc length. */
  var r = mulberry32(seed), X = new Float64Array(count * 2), y = new Int32Array(count);
  var R0 = 0.12, R1 = 1.5, SIGMA = noise == null ? 0.04 : noise;
  function radiusFromUniform(v) { var D = R1 - R0; return (-R0 + Math.sqrt(R0 * R0 + D * (R0 + R1) * v)) / D; }
  for (var k = 0; k < count; k++) {
    var cls = k % 2, u = radiusFromUniform(r());
    var ang = u * 2 * turns * Math.PI + cls * Math.PI, rad = R0 + (R1 - R0) * u;
    X[2 * k] = rad * Math.cos(ang) + SIGMA * gauss(r);
    X[2 * k + 1] = rad * Math.sin(ang) + SIGMA * gauss(r);
    y[k] = cls;
  }
  for (var a = count - 1; a > 0; a--) {
    var j = Math.floor(r() * (a + 1)), tmp = y[a]; y[a] = y[j]; y[j] = tmp;
    for (var f = 0; f < 2; f++) { tmp = X[2 * a + f]; X[2 * a + f] = X[2 * j + f]; X[2 * j + f] = tmp; }
  }
  return { X: X, y: y, n: count, nin: 2, K: 2 };
}
/* MNIST: bytes = Uint8Array of n*196 pixels (14x14, 0..255) followed by n labels.
   Returns train/test splits, pixels scaled to [0,1] then standardized with train statistics. */
function mnistFromBytes(bytes, nTotal, nTrain, nTest) {
  var P = 196, i, k;
  function take(start, n) {
    var X = new Float64Array(n * P), y = new Int32Array(n);
    for (k = 0; k < n; k++) {
      for (i = 0; i < P; i++) X[k * P + i] = bytes[(start + k) * P + i] / 255;
      y[k] = bytes[nTotal * P + start + k];
    }
    return { X: X, y: y, n: n, nin: P, K: 10 };
  }
  var tr = take(0, nTrain), te = take(nTotal - nTest, nTest);
  var mu = new Float64Array(P), sd = new Float64Array(P);
  for (k = 0; k < tr.n; k++) for (i = 0; i < P; i++) mu[i] += tr.X[k * P + i];
  for (i = 0; i < P; i++) mu[i] /= tr.n;
  for (k = 0; k < tr.n; k++) for (i = 0; i < P; i++) { var v = tr.X[k * P + i] - mu[i]; sd[i] += v * v; }
  for (i = 0; i < P; i++) sd[i] = Math.sqrt(sd[i] / tr.n) + 0.1;   /* +0.1: border pixels are ~constant */
  [tr, te].forEach(function (D) {
    for (k = 0; k < D.n; k++) for (i = 0; i < P; i++) D.X[k * P + i] = (D.X[k * P + i] - mu[i]) / sd[i];
  });
  return { train: tr, test: te };
}

/* ---------- network ----------
   x -> [Linear -> ReLU -> (Norm)] x L -> Linear -> softmax cross-entropy.
   arm: "none" | "ln" | "bn". Normalization placement "post" (after ReLU) as in Lecture 4. */
function makeNet(cfg, nin, K) {
  var r = mulberry32((cfg.seed || 1) + 1000), L = cfg.depth, d = cfg.width, arm = cfg.arm || "none";
  var dims = [nin]; for (var i = 0; i < L; i++) dims.push(d); dims.push(K);
  var layers = [];
  for (var m = 0; m < dims.length - 1; m++) {
    var nIn = dims[m], nOut = dims[m + 1];
    var ly = { nin: nIn, nout: nOut, W: new Float64Array(nOut * nIn), gW: new Float64Array(nOut * nIn),
               b: new Float64Array(nOut), gb: new Float64Array(nOut) };
    var sd = Math.sqrt(2 / nIn);
    for (var q = 0; q < nOut * nIn; q++) ly.W[q] = gauss(r) * sd;
    if (arm !== "none" && m < L) {
      ly.gam = new Float64Array(nOut).fill(1); ly.bet = new Float64Array(nOut);
      ly.ggam = new Float64Array(nOut); ly.gbet = new Float64Array(nOut);
      if (arm === "bn") { ly.runningMean = new Float64Array(nOut); ly.runningVar = new Float64Array(nOut).fill(1); }
    }
    layers.push(ly);
  }
  var net = { layers: layers, L: L, width: d, nin: nin, K: K, arm: arm, t: 0 };
  /* parameter list (value, gradient) pairs, with optimizer state attached lazily */
  net.params = [];
  layers.forEach(function (ly) {
    net.params.push({ p: ly.W, g: ly.gW, kind: "W" }); net.params.push({ p: ly.b, g: ly.gb, kind: "b" });
    if (ly.gam) { net.params.push({ p: ly.gam, g: ly.ggam, kind: "gamma" }); net.params.push({ p: ly.bet, g: ly.gbet, kind: "beta" }); }
  });
  net.nParams = net.params.reduce(function (s, e) { return s + e.p.length; }, 0);
  return net;
}

var EPS = 1e-5;
function normForward(arm, h, B, d, ly, train) {
  var o = new Float64Array(B * d), raw = new Float64Array(B * d), b, i;
  if (arm === "ln") {
    var mu = new Float64Array(B), inv = new Float64Array(B);
    for (b = 0; b < B; b++) {
      var m1 = 0; for (i = 0; i < d; i++) m1 += h[b * d + i]; m1 /= d; mu[b] = m1;
      var v1 = 0; for (i = 0; i < d; i++) { var t = h[b * d + i] - m1; v1 += t * t; }
      inv[b] = 1 / Math.sqrt(v1 / d + EPS);
      for (i = 0; i < d; i++) { var rv = (h[b * d + i] - m1) * inv[b]; raw[b * d + i] = rv; o[b * d + i] = ly.gam[i] * rv + ly.bet[i]; }
    }
    return { out: o, raw: raw, inv: inv, axis: "row" };
  }
  /* BatchNorm: batch statistics in training, stored statistics at evaluation */
  var muc = new Float64Array(d), invc = new Float64Array(d);
  if (train && B < 2) throw new Error("Training BN needs at least two samples per batch.");
  for (i = 0; i < d; i++) {
    var m2 = 0; for (b = 0; b < B; b++) m2 += h[b * d + i]; m2 /= B;
    var v2 = 0; for (b = 0; b < B; b++) { var t2 = h[b * d + i] - m2; v2 += t2 * t2; }
    var variance = v2 / B;
    if (train) {
      ly.runningMean[i] = 0.9 * ly.runningMean[i] + 0.1 * m2;
      ly.runningVar[i] = 0.9 * ly.runningVar[i] + 0.1 * v2 / (B - 1);
    } else { m2 = ly.runningMean[i]; variance = ly.runningVar[i]; }
    muc[i] = m2; invc[i] = 1 / Math.sqrt(variance + EPS);
    for (b = 0; b < B; b++) { var rv2 = (h[b * d + i] - m2) * invc[i]; raw[b * d + i] = rv2; o[b * d + i] = ly.gam[i] * rv2 + ly.bet[i]; }
  }
  return { out: o, raw: raw, inv: invc, axis: "col" };
}
function normBackward(dOut, N, B, d, ly) {
  var g = new Float64Array(B * d), dY = new Float64Array(B * d), b, i;
  for (b = 0; b < B; b++) for (i = 0; i < d; i++) {
    var q = b * d + i;
    ly.ggam[i] += dOut[q] * N.raw[q]; ly.gbet[i] += dOut[q];
    dY[q] = dOut[q] * ly.gam[i];
  }
  if (N.axis === "row") {
    for (b = 0; b < B; b++) {
      var s1 = 0, s2 = 0;
      for (i = 0; i < d; i++) { s1 += dY[b * d + i]; s2 += dY[b * d + i] * N.raw[b * d + i]; }
      s1 /= d; s2 /= d;
      for (i = 0; i < d; i++) g[b * d + i] = (dY[b * d + i] - s1 - N.raw[b * d + i] * s2) * N.inv[b];
    }
    return g;
  }
  for (i = 0; i < d; i++) {
    var c1 = 0, c2 = 0;
    for (b = 0; b < B; b++) { c1 += dY[b * d + i]; c2 += dY[b * d + i] * N.raw[b * d + i]; }
    c1 /= B; c2 /= B;
    for (b = 0; b < B; b++) g[b * d + i] = (dY[b * d + i] - c1 - N.raw[b * d + i] * c2) * N.inv[i];
  }
  return g;
}

/* forward (+ backward when grad=true) over the samples idx of D.
   Returns mean loss and error rate; gradients are left in net.layers[*].g*. */
function runBatch(net, D, idx, B, train, grad) {
  var L = net.L, d = net.width, nin = net.nin, K = net.K, arm = net.arm, b, i, j, o, m, q;
  var Z = [], H = [], N = [];
  var z0 = new Float64Array(B * nin);
  for (b = 0; b < B; b++) { var src = idx[b] * nin; for (i = 0; i < nin; i++) z0[b * nin + i] = D.X[src + i]; }
  Z[0] = z0;
  for (m = 0; m < L; m++) {
    var ly = net.layers[m], W = ly.W, dd = ly.nin, zin = Z[m];
    var h = new Float64Array(B * d);
    for (b = 0; b < B; b++) {
      var zoff = b * dd;
      for (o = 0; o < d; o++) {
        var s = ly.b[o], woff = o * dd;
        for (j = 0; j < dd; j++) s += W[woff + j] * zin[zoff + j];
        h[b * d + o] = s;
      }
    }
    var a = new Float64Array(B * d);
    for (q = 0; q < B * d; q++) a[q] = h[q] > 0 ? h[q] : 0;
    H[m] = h;
    if (arm !== "none") { N[m] = normForward(arm, a, B, d, ly, train); Z[m + 1] = N[m].out; }
    else Z[m + 1] = a;
  }
  var lo = net.layers[L], zL = Z[L], logits = new Float64Array(B * K), prob = new Float64Array(B * K);
  var loss = 0, err = 0;
  for (b = 0; b < B; b++) {
    var mx = -Infinity;
    for (o = 0; o < K; o++) {
      var s3 = lo.b[o], w3 = o * d;
      for (j = 0; j < d; j++) s3 += lo.W[w3 + j] * zL[b * d + j];
      logits[b * K + o] = s3; if (s3 > mx) mx = s3;
    }
    var se = 0, arg = 0;
    for (o = 0; o < K; o++) { var e = Math.exp(logits[b * K + o] - mx); prob[b * K + o] = e; se += e; if (logits[b * K + o] > logits[b * K + arg]) arg = o; }
    for (o = 0; o < K; o++) prob[b * K + o] /= se;
    var yb = D.y[idx[b]];
    loss += Math.log(se) + mx - logits[b * K + yb];
    if (arg !== yb) err++;
  }
  loss /= B; err /= B;
  if (!grad) return { loss: loss, err: err, logits: logits };

  for (m = 0; m <= L; m++) {
    var l2 = net.layers[m]; l2.gW.fill(0); l2.gb.fill(0);
    if (l2.ggam) { l2.ggam.fill(0); l2.gbet.fill(0); }
  }
  var dOut = new Float64Array(B * K);
  for (b = 0; b < B; b++) for (o = 0; o < K; o++) dOut[b * K + o] = (prob[b * K + o] - (D.y[idx[b]] === o ? 1 : 0)) / B;
  for (b = 0; b < B; b++) for (o = 0; o < K; o++) {
    var go = dOut[b * K + o]; if (go === 0) continue;
    lo.gb[o] += go;
    for (j = 0; j < d; j++) lo.gW[o * d + j] += go * zL[b * d + j];
  }
  var dZ = new Float64Array(B * d);
  for (b = 0; b < B; b++) for (j = 0; j < d; j++) {
    var s4 = 0; for (o = 0; o < K; o++) s4 += dOut[b * K + o] * lo.W[o * d + j];
    dZ[b * d + j] = s4;
  }
  for (m = L - 1; m >= 0; m--) {
    var ly2 = net.layers[m], dd2 = ly2.nin;
    var dA = N[m] ? normBackward(dZ, N[m], B, d, ly2) : dZ;
    var dH = new Float64Array(B * d);
    for (q = 0; q < B * d; q++) dH[q] = H[m][q] > 0 ? dA[q] : 0;
    for (b = 0; b < B; b++) {
      var base = b * dd2;
      for (o = 0; o < d; o++) {
        var g5 = dH[b * d + o]; if (g5 === 0) continue;
        ly2.gb[o] += g5;
        var wof = o * dd2;
        for (j = 0; j < dd2; j++) ly2.gW[wof + j] += g5 * Z[m][base + j];
      }
    }
    if (m > 0) {
      var dZp = new Float64Array(B * dd2);
      for (b = 0; b < B; b++) for (o = 0; o < d; o++) {
        var g6 = dH[b * d + o]; if (g6 === 0) continue;
        var wo = o * dd2, bo = b * dd2;
        for (j = 0; j < dd2; j++) dZp[bo + j] += g6 * ly2.W[wo + j];
      }
      dZ = dZp;
    }
  }
  return { loss: loss, err: err };
}

/* ---------- learning-rate schedules ----------
   s = {type, lr, steps, warmup, minFrac, milestones, gamma, decayFrac} ; t = 1..steps.
   The plateau rule is stateful and handled by the trainer (see Trainer.observe). */
function scheduleLR(s, t) {
  var T = s.steps, lr = s.lr, wu = s.warmup || 0, mn = s.minFrac == null ? 0 : s.minFrac;
  if (wu > 0 && t <= wu && s.type !== "invsqrt" && s.type !== "invt") return lr * t / wu;
  switch (s.type) {
    case "constant": case "plateau": return lr;
    case "step": {
      var f = 1; (s.milestones || [0.5, 0.75]).forEach(function (p) { if (t > p * T) f *= (s.gamma || 0.1); });
      return lr * f;
    }
    case "cosine": { var p = (t - wu) / Math.max(1, T - wu); return lr * (mn + (1 - mn) * 0.5 * (1 + Math.cos(Math.PI * p))); }
    case "linear": { var p2 = (t - wu) / Math.max(1, T - wu); return lr * (mn + (1 - mn) * (1 - p2)); }
    case "wsd": {
      var dStart = T * (1 - (s.decayFrac || 0.2));
      if (t <= dStart) return lr;
      var p3 = (t - dStart) / Math.max(1, T - dStart);
      return lr * (mn + (1 - mn) * (1 - p3));
    }
    case "invt": return lr / (1 + (t - 1) / (s.t0 || 100));            /* α_t ∝ 1/t (square-summable) */
    case "invsqrt": return lr / Math.sqrt(1 + (t - 1) / (s.t0 || 100)); /* α_t ∝ 1/√t */
  }
  return lr;
}

/* ---------- optimizers ---------- */
function applyUpdate(net, o, lr) {
  net.t++;
  var t = net.t;
  net.params.forEach(function (e) {
    var p = e.p, g = e.g, n = p.length, i;
    if (o.type === "adam") {
      if (!e.m) { e.m = new Float64Array(n); e.v = new Float64Array(n); }
      var b1 = o.beta1, b2 = o.beta2, eps = o.eps == null ? 1e-8 : o.eps;
      var c1 = o.biasCorrection === false ? 1 : 1 - Math.pow(b1, t);
      var c2 = o.biasCorrection === false ? 1 : 1 - Math.pow(b2, t);
      for (i = 0; i < n; i++) {
        e.m[i] = b1 * e.m[i] + (1 - b1) * g[i];
        e.v[i] = b2 * e.v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= lr * (e.m[i] / c1) / (Math.sqrt(e.v[i] / c2) + eps);
      }
    } else if (o.momentum) {
      if (!e.m) e.m = new Float64Array(n);
      for (i = 0; i < n; i++) { e.m[i] = o.momentum * e.m[i] + g[i]; p[i] -= lr * e.m[i]; }
    } else {
      for (i = 0; i < n; i++) p[i] -= lr * g[i];
    }
  });
}

/* ---------- full-dataset evaluation (forward only, evaluation mode) ---------- */
function evaluate(net, D, chunk) {
  var size = chunk || 256, tot = 0, er = 0;
  for (var s = 0; s < D.n; s += size) {
    var n = Math.min(size, D.n - s), idx = new Int32Array(n);
    for (var i = 0; i < n; i++) idx[i] = s + i;
    var q = runBatch(net, D, idx, n, false, false);
    tot += q.loss * n; er += q.err * n;
  }
  return { loss: tot / D.n, acc: 1 - er / D.n };
}
/* squared norm of the full-batch gradient (train mode off: evaluation-mode network) */
function fullGradNorm2(net, D) {
  var acc = net.params.map(function (e) { return new Float64Array(e.p.length); });
  var size = 256;
  for (var s = 0; s < D.n; s += size) {
    var n = Math.min(size, D.n - s), idx = new Int32Array(n);
    for (var i = 0; i < n; i++) idx[i] = s + i;
    runBatch(net, D, idx, n, net.arm !== "bn", true);
    net.params.forEach(function (e, k) { for (var j = 0; j < e.g.length; j++) acc[k][j] += e.g[j] * n / D.n; });
  }
  var t = 0; acc.forEach(function (a) { for (var j = 0; j < a.length; j++) t += a[j] * a[j]; });
  return t;
}

/* gradient noise at the current weights: σ²(w) = (1/m) Σ_i ‖∇f_i(w) − ḡ‖², ḡ = (1/m) Σ_i ∇f_i(w),
   over m fixed training samples (per-sample gradients, evaluation-mode network; none/LN only).
   This is the σ² of the constant-stepsize theorem: it stays positive when the network cannot
   fit the data and goes to 0 when every sample is fit (interpolation). */
function gradNoise(net, D, sampleIdx) {
  var m = sampleIdx.length, sizes = net.params.map(function (e) { return e.p.length; });
  var mean = sizes.map(function (n) { return new Float64Array(n); }), sumSq = 0, one = new Int32Array(1);
  var saved = net.layers.map(function (l) { return l.runningMean ? [l.runningMean.slice(), l.runningVar.slice()] : null; });
  for (var s = 0; s < m; s++) {
    one[0] = sampleIdx[s];
    runBatch(net, D, one, 1, false, true);
    net.params.forEach(function (e, k) { var g = e.g, mk = mean[k]; for (var j = 0; j < g.length; j++) { mk[j] += g[j] / m; sumSq += g[j] * g[j] / m; } });
  }
  net.layers.forEach(function (l, k) { if (saved[k]) { l.runningMean.set(saved[k][0]); l.runningVar.set(saved[k][1]); } });
  var meanSq = 0; mean.forEach(function (a) { for (var j = 0; j < a.length; j++) meanSq += a[j] * a[j]; });
  return { noise: Math.max(0, sumSq - meanSq), gradSq: meanSq };
}

/* mini-batch indices: sampling with replacement from a seeded stream shared by all runs
   that use the same seed and batch size (as in the normalization lab) */
function indices(seed, step, size, n) {
  var r = mulberry32(seed * 7919 + 90000 + step);
  var idx = new Int32Array(size);
  for (var i = 0; i < size; i++) idx[i] = Math.floor(r() * n);
  return idx;
}

/* ---------- trainer ----------
   cfg: {depth,width,arm,seed, batch, steps, opt:{type,beta1,beta2,eps,momentum,biasCorrection},
         sched:{type,lr,warmup,minFrac,milestones,gamma,decayFrac,patience,factor}, evalEvery}
   Reduce-on-plateau follows PyTorch's ReduceLROnPlateau (mode=min, threshold=1e-4 rel):
   the train loss is checked at every evaluation; after `patience` evaluations without
   improvement the lr is multiplied by `factor`. */
function Trainer(cfg, train, test) {
  this.cfg = cfg; this.train = train; this.test = test;
  this.net = makeNet(cfg, train.nin, train.K);
  this.step = 0; this.lrScale = 1; this.best = Infinity; this.bad = 0; this.drops = [];
  this.failed = false;
}
Trainer.prototype.lr = function () {
  var s = this.cfg.sched;
  return scheduleLR(s, Math.max(1, this.step)) * (s.type === "plateau" ? this.lrScale : 1);
};
Trainer.prototype.advance = function (n) {
  var c = this.cfg, out = null;
  for (var k = 0; k < n && this.step < c.steps && !this.failed; k++) {
    this.step++;
    var idx = indices(c.seed, this.step, c.batch, this.train.n);
    var q = runBatch(this.net, this.train, idx, c.batch, true, true);
    if (!isFinite(q.loss)) { this.failed = true; this.failedAt = this.step; break; }
    var lr = this.lr();
    applyUpdate(this.net, c.opt, lr);
    out = { loss: q.loss, lr: lr };
  }
  return out;
};
Trainer.prototype.observe = function (trainSet, testSet) {
  /* train/test metrics (full sets by default, or the given evaluation subsets); drives the plateau rule */
  var tr = evaluate(this.net, trainSet || this.train), te = (testSet || this.test) ? evaluate(this.net, testSet || this.test) : null;
  if (!isFinite(tr.loss)) { this.failed = true; this.failedAt = this.step; }
  var s = this.cfg.sched;
  if (s.type === "plateau" && this.step > 0) {
    if (tr.loss < this.best * (1 - 1e-4)) { this.best = tr.loss; this.bad = 0; }
    else if (++this.bad > (s.patience || 5)) {
      this.lrScale *= (s.factor || 0.1); this.bad = 0; this.drops.push(this.step);
    }
  }
  return { step: this.step, train: tr, test: te, lr: this.lr() };
};

var api = { mulberry32: mulberry32, gauss: gauss, spirals: spirals, mnistFromBytes: mnistFromBytes,
  makeNet: makeNet, runBatch: runBatch, applyUpdate: applyUpdate, scheduleLR: scheduleLR,
  evaluate: evaluate, fullGradNorm2: fullGradNorm2, gradNoise: gradNoise, indices: indices, Trainer: Trainer, eps: EPS };
if (typeof module !== "undefined" && module.exports) module.exports = api;
else root.OptEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
