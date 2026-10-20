/* One training run per worker, off the UI thread. No external runtime or network calls.
   Message in: {bytes, ntrain, ntest, cfg, every, evalN, noiseEvery, noiseM}
   Messages out: {type:'progress', ...} every `every` updates, then {type:'done', ...}. */
importScripts('engine.js');
self.onmessage = function (event) {
  var c = event.data, E = OptEngine;
  try {
    var M = E.mnistFromBytes(new Uint8Array(c.bytes), 10000, c.ntrain, c.ntest);
    function head(D, n) { n = Math.min(n, D.n); return { X: D.X.subarray(0, n * D.nin), y: D.y.subarray(0, n), n: n, nin: D.nin, K: D.K }; }
    var trE = head(M.train, c.evalN), teE = head(M.test, c.evalN);
    var noiseIdx = new Int32Array(c.noiseM || 0);
    for (var i = 0; i < noiseIdx.length; i++) noiseIdx[i] = (i * 31) % M.train.n;
    var T = new E.Trainer(c.cfg, M.train, null), steps = c.cfg.steps;
    var mbSum = 0, mbCnt = 0;
    function measure(final) {
      var o = T.observe(final ? M.train : trE, final ? M.test : teE);
      var msg = { type: final ? 'done' : 'progress', step: T.step, train: o.train, test: o.test, lr: o.lr,
                  mb: mbCnt ? mbSum / mbCnt : null, drops: T.drops.slice(), failed: T.failed, failedAt: T.failedAt };
      if (noiseIdx.length && (final || (c.noiseEvery && T.step % c.noiseEvery === 0)) && !T.failed) {
        var g = E.gradNoise(T.net, M.train, noiseIdx); msg.noise = g.noise; msg.gradSq = g.gradSq;
      }
      mbSum = 0; mbCnt = 0;
      return msg;
    }
    self.postMessage(measure(false));
    while (T.step < steps && !T.failed) {
      var stop = Math.min(steps, T.step + c.every);
      while (T.step < stop && !T.failed) { var q = T.advance(1); if (q) { mbSum += q.loss; mbCnt++; } }
      if (T.failed) break;
      if (T.step < steps) self.postMessage(measure(false));
    }
    if (T.failed) self.postMessage({ type: 'done', step: T.step, failed: true, failedAt: T.failedAt, drops: T.drops.slice() });
    else self.postMessage(measure(true));
  } catch (err) { self.postMessage({ type: 'error', message: String(err && err.message || err) }); }
};
