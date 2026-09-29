/* Numerical core adapted from the original course lab. No DOM or external dependencies. */
(function(root){
"use strict";
function mulberry32(a){
  return function(){
    a |= 0; a = a + 0x6D2B79F5 | 0;
    var t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gauss(r){
  var u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function fillHe(r, W, rows, cols){
  var sd = Math.sqrt(2 / cols);
  for (var i = 0; i < rows * cols; i++) W[i] = gauss(r) * sd;
}
function fillV(r, V, rows, cols){
  var sd = Math.sqrt(1 / cols);              /* ‖v_row‖ ≈ 1, so ‖w_row‖ ≈ √2 */
  for (var i = 0; i < rows * cols; i++) V[i] = gauss(r) * sd;
}

/* --- data: a linear teacher, with per-feature scales we control --- */
function buildData(seed, nin, nTrain, nTest, spread){
  var r = mulberry32(seed), i, k;
  var scale = new Float64Array(nin), w = new Float64Array(nin);
  for (i = 0; i < nin; i++) scale[i] = Math.pow(10, (r() * 2 - 1) * spread);
  for (i = 0; i < nin; i++) w[i] = gauss(r);
  function gen(n){
    var X = new Float64Array(n * nin), y = new Float64Array(n);
    for (k = 0; k < n; k++){
      var s = 0;
      for (i = 0; i < nin; i++){
        var v = gauss(r) * scale[i];
        X[k * nin + i] = v; s += v * w[i];
      }
      y[k] = s >= 0 ? 1 : 0;
    }
    return {X: X, y: y, n: n};
  }
  return {train: gen(nTrain), test: gen(nTest), nin: nin, scale: scale};
}
function standardize(data, nin){
  var d, i, k, n = data.train.n;
  var mu = new Float64Array(nin), sd = new Float64Array(nin);
  for (k = 0; k < n; k++) for (i = 0; i < nin; i++) mu[i] += data.train.X[k * nin + i];
  for (i = 0; i < nin; i++) mu[i] /= n;
  for (k = 0; k < n; k++) for (i = 0; i < nin; i++){
    var v = data.train.X[k * nin + i] - mu[i]; sd[i] += v * v;
  }
  for (i = 0; i < nin; i++) sd[i] = Math.sqrt(sd[i] / n) + 1e-8;
  [data.train, data.test].forEach(function(D){
    for (k = 0; k < D.n; k++) for (i = 0; i < nin; i++){
      D.X[k * nin + i] = (D.X[k * nin + i] - mu[i]) / sd[i];
    }
  });
}

/* --- the network --- */
var ARMS = ["none", "ln", "bn", "rms", "proj", "reg", "wn"];
var ARM_LABEL = {
  none: "none", ln: "ln (LayerNorm)", bn: "bn (BatchNorm)", rms: "rms (RMSNorm)",
  proj: "proj (projection)", reg: "reg (regularizer)", wn: "wn (reparameterization)"
};
function makeNet(L, width, nin, arm, r){
  var dims = [nin], i;
  for (i = 0; i < L; i++) dims.push(width);
  dims.push(1);
  var isNorm = arm === "ln" || arm === "bn" || arm === "rms";
  var layers = [];
  for (var m = 0; m < dims.length - 1; m++){
    var nIn = dims[m], nOut = dims[m + 1];
    var ly = {nin: nIn, nout: nOut, W: new Float64Array(nOut * nIn), gW: new Float64Array(nOut * nIn),
              b: new Float64Array(nOut), gb: new Float64Array(nOut)};
    if (arm === "wn"){ ly.V = new Float64Array(nOut * nIn); ly.gV = new Float64Array(nOut * nIn); fillV(r, ly.V, nOut, nIn); }
    else fillHe(r, ly.W, nOut, nIn);
    /* The affine pair of the normalization, one entry per feature of the layer
       output: γ starts at 1, β starts at 0, and the page can switch either off.
       The output layer is not normalized, so it carries no pair. */
    if (isNorm && m < L){
      ly.gam  = new Float64Array(nOut); ly.gam.fill(1);
      ly.bet  = new Float64Array(nOut);
      ly.ggam = new Float64Array(nOut); ly.gbet = new Float64Array(nOut);
      if (arm === "bn") { ly.runningMean = new Float64Array(nOut); ly.runningVar = new Float64Array(nOut).fill(1); }
    }
    layers.push(ly);
  }
  return {layers: layers, L: L, width: width, nin: nin, arm: arm};
}
function effW(net, m){
  var ly = net.layers[m];
  if (net.arm !== "wn") return ly.W;
  var nOut = ly.nout, nIn = ly.nin, W = new Float64Array(nOut * nIn), S = Math.SQRT2;
  for (var o = 0; o < nOut; o++){
    var nrm = 0, i;
    for (i = 0; i < nIn; i++){ var v = ly.V[o * nIn + i]; nrm += v * v; }
    nrm = Math.sqrt(nrm) + 1e-12;
    for (i = 0; i < nIn; i++) W[o * nIn + i] = S * ly.V[o * nIn + i] / nrm;
  }
  return W;
}

/* --- forward + backward over a batch; returns loss and the per-layer scale --- */
var EPS = 1e-5;
function runBatch(net, D, idx, B, train, opt){
  var L = net.L, d = net.width, nin = net.nin;
  var Z = [], P = [], H = [], N = [];
  Z[0] = B * nin;
  var zbuf = new Float64Array(B * nin);
  for (var b = 0; b < B; b++) for (var i = 0; i < nin; i++) zbuf[b * nin + i] = D.X[idx[b] * nin + i];
  Z[0] = zbuf;

  for (var m = 0; m < L; m++){
    var ly = net.layers[m], W = effW(net, m), dd = (m === 0 ? nin : d);
    var h = new Float64Array(B * d);
    for (var b2 = 0; b2 < B; b2++){
      var zoff = b2 * dd;
      for (var o = 0; o < d; o++){
        var s = 0, woff = o * dd;
        for (var j = 0; j < dd; j++) s += W[woff + j] * Z[m][zoff + j];
        h[b2 * d + o] = s + ly.b[o];
      }
    }
    var post = net.placement === "post";
    var activation = Float64Array.from(h, function(v){ return Math.max(0,v); });
    var p = post ? activation : h, nrm = null;
    if (net.arm === "ln" || net.arm === "bn" || net.arm === "rms"){
      if (ly.gam && !(opt && opt.scaleOnly)){ ly.ggam.fill(0); ly.gbet.fill(0); }
      nrm = normForward(net.arm, post ? activation : h, B, d, ly, train, opt && opt.batchStats);
      p = nrm.out;
    }
    H[m] = h; N[m] = nrm;
    var z = new Float64Array(B * d);
    for (var q = 0; q < B * d; q++) z[q] = post && nrm ? p[q] : Math.max(0,p[q]);
    P[m] = p; Z[m + 1] = z;
  }
  /* output layer */
  var lw = net.layers[L], Wc = effW(net, L);
  var zL = Z[L], out = new Float64Array(B), loss = 0, err = 0;
  for (b = 0; b < B; b++){
    var o2 = 0;
    for (j = 0; j < d; j++) o2 += Wc[j] * zL[b * d + j];
    out[b] = o2 + net.layers[L].b[0];
    var yb = D.y[idx[b]];
    var p1 = 1 / (1 + Math.exp(-out[b]));
    loss += Math.max(out[b], 0) - yb * out[b] + Math.log1p(Math.exp(-Math.abs(out[b])));
    err += ((p1 >= 0.5 ? 1 : 0) !== yb) ? 1 : 0;
  }
  loss /= B; err /= B;

  /* per-layer scale of what entered the activation.
     Computed with the peak factored out: summing the squares directly overflows
     to Infinity once a layer has exploded, which would report a real magnitude
     as "non-finite". */
  var scale = [];
  for (m = 0; m < L; m++){
    var pk = 0, acc = 0;
    for (q = 0; q < B * d; q++){ var av = Math.abs(P[m][q]); if (av > pk) pk = av; }
    if (pk > 0 && isFinite(pk)){
      for (q = 0; q < B * d; q++){ var rr = P[m][q] / pk; acc += rr * rr; }
      scale.push(pk * Math.sqrt(acc / (B * d)));
    } else {
      s = 0;
      for (q = 0; q < B * d; q++) s += P[m][q] * P[m][q];
      scale.push(Math.sqrt(s / (B * d)));
    }
  }
  if (opt && opt.scaleOnly) return {loss: loss, err: err, scale: scale, logits: out};

  /* ---- backward ---- */
  for (m = 0; m < net.layers.length; m++){ net.layers[m].gW.fill(0); net.layers[m].gb.fill(0); if (net.arm === "wn") net.layers[m].gV.fill(0); }
  var dOut = new Float64Array(B);
  for (b = 0; b < B; b++){
    var pb = 1 / (1 + Math.exp(-out[b]));
    dOut[b] = (pb - D.y[idx[b]]) / B;
  }
  var gWlast = net.layers[L].gW;
  for (b = 0; b < B; b++) for (j = 0; j < d; j++) gWlast[j] += dOut[b] * zL[b * d + j];
  for (b = 0; b < B; b++) net.layers[L].gb[0] += dOut[b];
  var dZ = new Float64Array(B * d);
  for (b = 0; b < B; b++) for (j = 0; j < d; j++) dZ[b * d + j] = dOut[b] * Wc[j];

  for (m = L - 1; m >= 0; m--){
    var ly2 = net.layers[m], dd2 = (m === 0 ? nin : d);
    var dP = new Float64Array(B * d);
    for (q = 0; q < B * d; q++) dP[q] = net.placement === "post" && N[m] ? dZ[q] : (P[m][q] > 0 ? dZ[q] : 0);
    var dH = dP;
    if (N[m]) dH = normBackward(net.arm, dP, N[m], B, d, net, m);
    if (net.placement === "post" && N[m]) for(q=0;q<B*d;q++) dH[q] *= H[m][q] > 0 ? 1 : 0;
    var gWm = ly2.gW;
    for (b = 0; b < B; b++){
      var zo = b * dd2;
      for (var o3 = 0; o3 < d; o3++){
        var go = dH[b * d + o3];
        if (go === 0) continue;
        var wof = o3 * dd2, base = b * dd2;
        for (j = 0; j < dd2; j++) gWm[wof + j] += go * Z[m][base + j];
      }
    }
    for (var o4 = 0; o4 < d; o4++){
      var s4 = 0;
      for (var b4 = 0; b4 < B; b4++) s4 += dH[b4 * d + o4];
      ly2.gb[o4] += s4;
    }
    if (m > 0){
      var Wm = effW(net, m), dZprev = new Float64Array(B * dd2);
      for (b = 0; b < B; b++) for (j = 0; j < dd2; j++){
        var s2 = 0;
        for (o3 = 0; o3 < d; o3++) s2 += dH[b * d + o3] * Wm[o3 * dd2 + j];
        dZprev[b * dd2 + j] = s2;
      }
      dZ = dZprev;
    }
  }
  return {loss: loss, err: err, scale: scale};
}

function normForward(arm, h, B, d, ly, train, batchStats){
  var useG = !!(ly && ly.gam) && ly.useGam !== false, useB = !!(ly && ly.bet) && ly.useBet !== false;
  if (arm === "rms"){
    var out = new Float64Array(B * d), raw = new Float64Array(B * d), inv = new Float64Array(B);
    for (var b = 0; b < B; b++){
      var s = 0;
      for (var i = 0; i < d; i++){ var v = h[b * d + i]; s += v * v; }
      inv[b] = 1 / Math.sqrt(s / d + EPS);
      for (i = 0; i < d; i++){
        var rv = h[b * d + i] * inv[b];
        raw[b * d + i] = rv;
        out[b * d + i] = useG ? ly.gam[i] * rv : rv;
      }
    }
    /* RMSNorm keeps the scale and has no shift, so β never applies here */
    return {out: out, raw: raw, inv: inv, h: h, useG: useG, useB: false};
  }
  var mu = new Float64Array(arm === "ln" ? B : d), inv2 = new Float64Array(arm === "ln" ? B : d);
  var o2 = new Float64Array(B * d), r2 = new Float64Array(B * d), i2, k;
  if (arm === "ln"){
    for (b = 0; b < B; b++){
      var m1 = 0;
      for (i2 = 0; i2 < d; i2++) m1 += h[b * d + i2];
      m1 /= d; mu[b] = m1;
      var v1 = 0;
      for (i2 = 0; i2 < d; i2++){ var t = h[b * d + i2] - m1; v1 += t * t; }
      inv2[b] = 1 / Math.sqrt(v1 / d + EPS);
      for (i2 = 0; i2 < d; i2++){
        var rv1 = (h[b * d + i2] - m1) * inv2[b];
        r2[b * d + i2] = rv1;
        o2[b * d + i2] = (useG ? ly.gam[i2] * rv1 : rv1) + (useB ? ly.bet[i2] : 0);
      }
    }
  } else { /* Internal storage: sample-major. The classroom matrix is its transpose. */
    if (!ly.runningMean) { ly.runningMean = new Float64Array(d); ly.runningVar = new Float64Array(d).fill(1); }
    if (train && B < 2) throw new Error("Training BN needs at least two samples per feature.");

    for (i2 = 0; i2 < d; i2++){
      var m2 = 0;
      for (b = 0; b < B; b++) m2 += h[b * d + i2];
      m2 /= B; mu[i2] = m2;
      var v2 = 0;
      for (b = 0; b < B; b++){ var t2 = h[b * d + i2] - m2; v2 += t2 * t2; }
      var variance = v2 / B;
      if (train) {
        ly.runningMean[i2] = 0.9 * ly.runningMean[i2] + 0.1 * m2;
        ly.runningVar[i2] = 0.9 * ly.runningVar[i2] + 0.1 * v2 / (B - 1);
      } else if (!batchStats) {
        m2 = ly.runningMean[i2]; variance = ly.runningVar[i2]; mu[i2] = m2;
      }
      inv2[i2] = 1 / Math.sqrt(variance + EPS);
      for (b = 0; b < B; b++){
        var rv2 = (h[b * d + i2] - m2) * inv2[i2];
        r2[b * d + i2] = rv2;
        o2[b * d + i2] = (useG ? ly.gam[i2] * rv2 : rv2) + (useB ? ly.bet[i2] : 0);
      }
    }
  }
  return {out: o2, raw: r2, mu: mu, inv: inv2, h: h, axis: arm === "ln" ? "row" : "col",
          useG: useG, useB: useB};
}
function normBackward(arm, dOut, N, B, d, net, m){
  var g = new Float64Array(B * d), dY = dOut, b, i;
  var ly = net.layers[m];
  /* The pair sits between the standardization and everything above it, so its
     own gradients are elementwise and the rest of the rule sees dOut ⊙ γ. */
  if (ly && ly.gam && (N.useG || N.useB)){
    if (N.useG){
      for (b = 0; b < B; b++) for (i = 0; i < d; i++)
        ly.ggam[i] += dOut[b * d + i] * N.raw[b * d + i];
      dY = new Float64Array(B * d);
      for (b = 0; b < B; b++) for (i = 0; i < d; i++)
        dY[b * d + i] = dOut[b * d + i] * ly.gam[i];
    }
    if (N.useB && arm !== "rms"){
      for (b = 0; b < B; b++) for (i = 0; i < d; i++) ly.gbet[i] += dOut[b * d + i];
    }
  }
  if (arm === "rms"){
    for (b = 0; b < B; b++){
      var hd = 0, s = 0;
      for (i = 0; i < d; i++){ hd += N.h[b * d + i] * dY[b * d + i]; s += N.h[b * d + i] * N.h[b * d + i]; }
      var inv = N.inv[b], inv3 = inv * inv * inv;
      for (i = 0; i < d; i++)
        g[b * d + i] = dY[b * d + i] * inv - N.h[b * d + i] * hd * inv3 / d;
    }
    return g;
  }
  if (N.axis === "row"){
    for (b = 0; b < B; b++){
      var s1 = 0, s2 = 0;
      for (i = 0; i < d; i++){ s1 += dY[b * d + i]; s2 += dY[b * d + i] * N.raw[b * d + i]; }
      s1 /= d; s2 /= d;
      for (i = 0; i < d; i++)
        g[b * d + i] = (dY[b * d + i] - s1 - N.raw[b * d + i] * s2) * N.inv[b];
    }
    return g;
  }
  for (i = 0; i < d; i++){
    var c1 = 0, c2 = 0;
    for (b = 0; b < B; b++){ c1 += dY[b * d + i]; c2 += dY[b * d + i] * N.raw[b * d + i]; }
    c1 /= B; c2 /= B;
    for (b = 0; b < B; b++)
      g[b * d + i] = (dY[b * d + i] - c1 - N.raw[b * d + i] * c2) * N.inv[i];
  }
  return g;
}

/* --- one SGD step, including the mechanism --- */
var REG_LAMBDA = 1.0, REG_TARGET = Math.SQRT2;
function sgdStep(net, D, idx, B, eta){
  var res = runBatch(net, D, idx, B, true, null);
  var layers = net.layers;
  for (var m = 0; m < layers.length; m++){
    var ly = layers[m], nOut = ly.nout, nIn = ly.nin, o, i;
    /* the affine pair of the normalization, updated like any other parameter */
    if (ly.gam){
      if (ly.useGam !== false) for (i = 0; i < nOut; i++) ly.gam[i] -= eta * ly.ggam[i];
      if (ly.useBet !== false) for (i = 0; i < nOut; i++) ly.bet[i] -= eta * ly.gbet[i];
    }
    for (i = 0; i < nOut; i++) ly.b[i] -= eta * ly.gb[i];
    if (net.arm === "wn"){
      var W = effW(net, m);
      for (o = 0; o < nOut; o++){
        var nrm = 0;
        for (i = 0; i < nIn; i++){ var v = ly.V[o * nIn + i]; nrm += v * v; }
        nrm = Math.sqrt(nrm) + 1e-12;
        var dot = 0;
        for (i = 0; i < nIn; i++) dot += ly.gW[o * nIn + i] * ly.V[o * nIn + i];
        for (i = 0; i < nIn; i++){
          var u = ly.V[o * nIn + i] / nrm;
          var gr = (Math.SQRT2 / nrm) * (ly.gW[o * nIn + i] - u * (dot / nrm));
          ly.V[o * nIn + i] -= eta * gr;
        }
      }
      continue;
    }
    for (o = 0; o < nOut; o++){
      var rowNrm = 0;
      for (i = 0; i < nIn; i++){ var w2 = ly.W[o * nIn + i]; rowNrm += w2 * w2; }
      rowNrm = Math.sqrt(rowNrm) + 1e-12;
      for (i = 0; i < nIn; i++){
        var gr2 = ly.gW[o * nIn + i];
        if (net.arm === "reg") gr2 += 2 * REG_LAMBDA * (rowNrm - REG_TARGET) * ly.W[o * nIn + i] / rowNrm;
        ly.W[o * nIn + i] -= eta * gr2;
      }
      if (net.arm === "proj"){
        var nn = 0;
        for (i = 0; i < nIn; i++){ var w3 = ly.W[o * nIn + i]; nn += w3 * w3; }
        nn = Math.sqrt(nn) + 1e-12;
        for (i = 0; i < nIn; i++) ly.W[o * nIn + i] *= REG_TARGET / nn;
      }
    }
  }
  return res;
}


function create(config, arm) {
  var net = makeNet(config.depth, config.width, 2, arm, mulberry32(config.seed + 1000));
  net.placement = config.placement;
  net.layers.forEach(function(ly){ ly.useGam = config.affine !== false; ly.useBet = config.affine !== false; });
  return net;
}
function moons(seed, count, scale) {
  var r=mulberry32(seed), X=new Float64Array(count*2), y=new Float64Array(count);
  for(var k=0;k<count;k++){
    var cls=k%2, t=r()*Math.PI;
    X[2*k]=(cls ? 1-Math.cos(t) : Math.cos(t)) + 0.12*gauss(r);
    X[2*k+1]=((cls ? 0.5-Math.sin(t) : Math.sin(t)) + 0.12*gauss(r))*scale;
    y[k]=cls;
  }
  // Shuffle rather than forcing each size-2 test batch to contain both classes.
  for(var a=count-1;a>0;a--){
    var j=Math.floor(r()*(a+1)),tmp=y[a];y[a]=y[j];y[j]=tmp;
    for(var f=0;f<2;f++){tmp=X[2*a+f];X[2*a+f]=X[2*j+f];X[2*j+f]=tmp;}
  }
  return {X:X,y:y,n:count};
}
function evaluate(net, data, batchStats, batchSize) {
  var total=0, errors=0, scales=null, logits=new Float64Array(data.n);
  var size=batchStats ? (batchSize || 32) : 128;
  for(var start=0;start<data.n;start+=size){
    var n=Math.min(size,data.n-start), idx=Int32Array.from({length:n},function(_,i){return start+i;});
    var q=runBatch(net,data,idx,n,false,{scaleOnly:true,batchStats:batchStats});
    total+=q.loss*n; errors+=q.err*n; logits.set(q.logits,start);
    if(!scales) scales=q.scale.map(function(){return 0;});
    q.scale.forEach(function(v,i){scales[i]+=v*n/data.n;});
  }
  return {loss:total/data.n,err:errors/data.n,scale:scales,logits:logits};
}
function indices(seed, step, size, n) {
  var r=mulberry32(seed+90000+step);
  return Int32Array.from({length:size},function(){return Math.floor(r()*n);});
}
var api={create:create,moons:moons,evaluate:evaluate,indices:indices,step:sgdStep,
 forward:runBatch,normForward:normForward,normBackward:normBackward,eps:EPS};
if(typeof module!=="undefined" && module.exports) module.exports=api;
else root.NormEngine=api;
})(typeof globalThis!=="undefined"?globalThis:this);
