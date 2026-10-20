/* Shared UI helpers for the Lecture 6 labs: language toggle, canvas charts, a pool of
   training workers (one run per worker), data loading and JSON export. No dependencies. */
(function (root) {
'use strict';
var K = {};
K.lang = 'en';
K.t = function (en, zh) { return K.lang === 'zh' ? zh : en; };
K.$ = function (id) { return document.getElementById(id); };
K.applyLang = function () {
  document.documentElement.lang = K.lang === 'en' ? 'en' : 'zh-CN';
  document.querySelectorAll('[data-en]').forEach(function (el) { el.innerHTML = el.dataset[K.lang]; });
  var b = K.$('lang'); if (b) b.textContent = K.lang === 'en' ? '中文' : 'English';
};
K.initLang = function (onChange) {
  var b = K.$('lang'); if (!b) return;
  b.onclick = function () { K.lang = K.lang === 'en' ? 'zh' : 'en'; K.applyLang(); if (onChange) onChange(); };
};

/* ---- number formatting ---- */
K.fmt = function (v, d) {
  if (v == null || !isFinite(v)) return '—';
  var a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(d == null ? 1 : d);
  return v.toFixed(d == null ? 3 : d);
};
K.pct = function (v) { return v == null || !isFinite(v) ? '—' : (100 * v).toFixed(1) + '%'; };

/* ---- canvas line chart ----
   series: [{color, label, data:[[x,y],...], dash, width, alpha}]
   o: {xMax, yMin, yMax, log, xLabel, yFmt, labels(bool), vlines:[{x,color,label}], empty} */
K.chart = function (id, series, o) {
  var el = K.$(id); if (!el) return;
  var g = el.getContext('2d'), w = el.width, h = el.height;
  g.clearRect(0, 0, w, h);
  var showLabels = !!o.labels, left = 58, right = showLabels ? 128 : 16, top = 14, bottom = 40;
  var log = !!o.log;
  var tr = function (v) { return log ? Math.log10(Math.max(1e-12, v)) : v; };
  var lo = tr(o.yMin), hi = tr(o.yMax); if (hi - lo < 1e-9) hi = lo + 1;
  var tx = function (x) { return left + (w - left - right) * x / Math.max(1, o.xMax); };
  var ty = function (y) { var v = Math.min(hi, Math.max(lo, tr(y))); return h - bottom - (h - top - bottom) * (v - lo) / (hi - lo); };
  g.font = '13px system-ui, -apple-system, sans-serif'; g.lineWidth = 1;
  var nTicks = 4;
  if (log) {
    var a = Math.floor(lo), b = Math.ceil(hi), step = Math.max(1, Math.ceil((b - a) / 5));
    for (var e = a; e <= b; e += step) {
      var val = Math.pow(10, e); if (tr(val) < lo - 1e-9 || tr(val) > hi + 1e-9) continue;
      var y = ty(val);
      g.strokeStyle = '#e4e9f1'; g.beginPath(); g.moveTo(left, y); g.lineTo(w - right, y); g.stroke();
      g.fillStyle = '#66748a'; g.textAlign = 'right'; g.fillText(o.yFmt ? o.yFmt(val) : (e >= -2 && e <= 3 ? String(+val.toPrecision(1)) : '1e' + e), left - 8, y + 4);
    }
  } else {
    for (var i = 0; i <= nTicks; i++) {
      var v2 = o.yMin + (o.yMax - o.yMin) * i / nTicks, y2 = ty(v2);
      g.strokeStyle = '#e4e9f1'; g.beginPath(); g.moveTo(left, y2); g.lineTo(w - right, y2); g.stroke();
      g.fillStyle = '#66748a'; g.textAlign = 'right'; g.fillText(o.yFmt ? o.yFmt(v2) : (+v2.toPrecision(3)).toString(), left - 8, y2 + 4);
    }
  }
  g.textAlign = 'center'; g.fillStyle = '#66748a';
  for (var j = 0; j <= 4; j++) { var x = o.xMax * j / 4; g.fillText(Math.round(x).toLocaleString('en-US'), tx(x), h - bottom + 19); }
  g.fillText(o.xLabel || '', (left + w - right) / 2, h - 3);
  (o.vlines || []).forEach(function (vl) {
    g.save(); g.strokeStyle = vl.color || '#9aa6b8'; g.setLineDash([4, 4]); g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(tx(vl.x), top); g.lineTo(tx(vl.x), h - bottom); g.stroke(); g.restore();
    if (vl.label) { g.fillStyle = vl.color || '#9aa6b8'; g.textAlign = 'left'; g.font = '11px system-ui'; g.fillText(vl.label, tx(vl.x) + 3, top + 10); g.font = '13px system-ui'; }
  });
  var any = false;
  series.forEach(function (s) {
    if (!s.data || !s.data.length) return; any = true;
    g.save(); g.globalAlpha = s.alpha == null ? 1 : s.alpha;
    g.strokeStyle = s.color; g.lineWidth = s.width || 2.6; if (s.dash) g.setLineDash(s.dash);
    g.beginPath(); var moved = false;
    s.data.forEach(function (p) {
      if (!isFinite(p[1]) || (log && p[1] <= 0)) { moved = false; return; }
      var X = tx(p[0]), Y = ty(p[1]); if (!moved) { g.moveTo(X, Y); moved = true; } else g.lineTo(X, Y);
    });
    g.stroke(); g.restore();
  });
  if (!any) { g.fillStyle = '#768399'; g.textAlign = 'center'; g.fillText(o.empty || K.t('Run the experiment to see measured curves', '运行实验，查看实测曲线'), (left + w - right) / 2, h / 2); }
  if (showLabels) {
    var pts = [];
    series.forEach(function (s) {
      if (!s.label || !s.data) return; var last = null;
      s.data.forEach(function (p) { if (isFinite(p[1]) && (!log || p[1] > 0)) last = p; });
      if (last) pts.push({ c: s.color, label: s.label, x: tx(last[0]), y0: ty(last[1]), y: ty(last[1]) });
    });
    pts.sort(function (p, q) { return p.y - q.y; }); var prev = -1e9;
    pts.forEach(function (p) { p.y = Math.min(h - bottom - 8, Math.max(top + 8, p.y)); if (p.y < prev + 15) p.y = prev + 15; prev = p.y; });
    if (pts.length && pts[pts.length - 1].y > h - bottom - 8) { var d = pts[pts.length - 1].y - (h - bottom - 8); pts.forEach(function (p) { p.y -= d; }); }
    g.font = '600 12px system-ui'; g.textAlign = 'left'; g.textBaseline = 'middle';
    pts.forEach(function (p) {
      g.fillStyle = p.c; g.beginPath(); g.arc(p.x, p.y0, 2.6, 0, 7); g.fill();
      g.strokeStyle = p.c; g.lineWidth = 1.1; g.beginPath(); g.moveTo(p.x + 2, p.y0); g.lineTo(p.x + 8, p.y); g.stroke();
      g.lineWidth = 3.2; g.strokeStyle = '#fff'; g.strokeText(p.label, p.x + 10, p.y); g.fillStyle = p.c; g.fillText(p.label, p.x + 10, p.y);
    });
    g.textBaseline = 'alphabetic';
  }
};
K.range = function (vals, lo0, hi0, log) {
  var v = vals.filter(function (x) { return isFinite(x) && (!log || x > 0); });
  if (!v.length) return [lo0, hi0];
  var mn = Math.min.apply(null, v), mx = Math.max.apply(null, v);
  if (log) return [Math.pow(10, Math.floor(Math.log10(mn) * 2) / 2), Math.pow(10, Math.ceil(Math.log10(mx) * 2) / 2)];
  return [Math.min(lo0, mn), Math.max(hi0, mx)];
};

/* ---- data ---- */
var dataPromise = null;
K.loadMnist = function () {
  if (!dataPromise) dataPromise = fetch('../lec6-data/mnist14.bin').then(function (r) {
    if (!r.ok) throw new Error('Could not load ../lec6-data/mnist14.bin (' + r.status + ')');
    return r.arrayBuffer();
  });
  return dataPromise;
};

/* ---- worker pool: one training run per worker, at most `parallel` at a time ---- */
K.Pool = function (onUpdate, onDone) { this.workers = []; this.onUpdate = onUpdate; this.onDone = onDone; this.running = false; };
K.Pool.prototype.start = function (runs, common) {
  var self = this; this.stop(); this.running = true;
  var parallel = Math.max(1, Math.min(runs.length, (navigator.hardwareConcurrency || 4) - 1, 4));
  var next = 0, finished = 0;
  this.results = runs.map(function (r) { return { id: r.id, history: [], final: null, failed: false, error: null, started: false }; });
  return K.loadMnist().then(function (buf) {
    function launch() {
      if (!self.running || next >= runs.length) return;
      var k = next++, r = runs[k], res = self.results[k];
      var wk = new Worker('worker.js'); self.workers.push(wk); res.started = true;
      wk.onmessage = function (ev) {
        var m = ev.data;
        if (m.type === 'error') { res.error = m.message; res.failed = true; }
        else if (m.type === 'progress') res.history.push(m);
        else if (m.type === 'done') { if (!m.failed) res.history.push(m); res.final = m; res.failed = !!m.failed; res.failedAt = m.failedAt; }
        if (m.type === 'done' || m.type === 'error') {
          wk.terminate(); finished++;
          if (finished === runs.length) { self.running = false; self.onUpdate(self.results); self.onDone(self.results); return; }
          launch();
        }
        self.onUpdate(self.results);
      };
      wk.onerror = function (e) { res.error = e.message; res.failed = true; finished++; wk.terminate(); self.onUpdate(self.results); if (finished === runs.length) { self.running = false; self.onDone(self.results); } else launch(); };
      wk.postMessage(Object.assign({}, common, r.msg, { bytes: buf.slice(0) }));
    }
    for (var p = 0; p < parallel; p++) launch();
  });
};
K.Pool.prototype.stop = function () { this.running = false; this.workers.forEach(function (w) { w.terminate(); }); this.workers = []; };

/* ---- export ---- */
K.download = function (name, obj) {
  var blob = new Blob([JSON.stringify(obj, null, 1)], { type: 'application/json' });
  var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
};

root.LabKit = K;
})(typeof globalThis !== 'undefined' ? globalThis : this);
