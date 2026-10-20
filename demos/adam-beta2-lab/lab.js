'use strict';
/* Lab 3 · Adam and β₂ (ME2605 Lecture 6, Parts III–IV).
   Panel 1: Reddi et al. (2018) counterexample, expected vs stochastic directions.
   Panel 2: Adam on a real MLP (MNIST 14×14), sweep β₂ or a β₁ × β₂ grid. */
const K = LabKit, $ = K.$, t = K.t;
const COLORS = ['#d8543d', '#b7791f', '#3262ed', '#008577', '#7c3aed', '#64748b', '#0e7490'];
const colorOf = (list, v) => COLORS[list.indexOf(v) % COLORS.length];

/* ---------------- Panel 1 ---------------- */
const R_B2 = ['0.5', '0.9', '0.99', '0.999'];
let rChosen = R_B2.slice(), rRes = null;
function rList() {
  $('r-b2').innerHTML = R_B2.map(v => '<label class="sched"><input type="checkbox" value="' + v + '"' + (rChosen.includes(v) ? ' checked' : '') + '><span class="swatch" style="background:' + colorOf(R_B2, v) + '"></span><span>' + v + '</span></label>').join('');
  $('r-b2').querySelectorAll('input').forEach(cb => cb.onchange = () => { rChosen = R_B2.filter(v => v === cb.value ? cb.checked : rChosen.includes(v)); rRun(); });
}
function rRun() {
  const b1 = +$('r-b1').value, alpha = +$('r-alpha').value, T = +$('r-T').value, seed = Math.max(1, Math.min(99999, Math.round(+$('r-seed').value) || 1));
  const every = Math.max(1, Math.round(T / 400));
  rRes = { T, every, runs: rChosen.map((v, i) => Object.assign({ b2: v }, Reddi.run({ C: 10, delta: 0.1, beta1: b1, beta2: +v, alpha, T, seed, every, sgd: i === 0 && $('r-sgd').checked }))) };
  rRender();
}
function rRender() {
  if (!rRes) return;
  const { T, every, runs } = rRes;
  $('r-legend').innerHTML = runs.map(r => '<span style="color:' + colorOf(R_B2, r.b2) + '">● Adam β₂ = ' + r.b2 + '</span>').join('') + (runs[0] && runs[0].xsSgd ? '<span style="color:#182237">- - SGD</span>' : '') + '<span style="color:#9aa5b5">' + t('· x* = −1', '· x* = −1') + '</span>';
  const ser = runs.map(r => ({ color: colorOf(R_B2, r.b2), data: r.xs.map((x, i) => [i * every, x]), label: 'β₂ ' + r.b2 }));
  if (runs[0] && runs[0].xsSgd) ser.push({ color: '#182237', dash: [6, 4], data: runs[0].xsSgd.map((x, i) => [i * every, x]), label: 'SGD' });
  ser.push({ color: '#9aa5b5', dash: [2, 4], data: [[0, -1], [T, -1]] });
  K.chart('r-x', ser, { xMax: T, yMin: -1.05, yMax: 1.05, xLabel: t('iterations t', '迭代次数 t'), yFmt: v => v.toFixed(1), labels: true });
  rArrows(runs);
  $('r-table').innerHTML = '<table class="heat" style="width:100%;border-spacing:0"><tr><th>β₂</th><th>' + t('step when g = +10<br>(prob. 0.1)', 'g = +10 时的一步<br>（概率 0.1）') + '</th><th>' + t('step when g = −1<br>(prob. 0.9)', 'g = −1 时的一步<br>（概率 0.9）') + '</th><th>' + t('expected step', '期望一步') + '</th><th>' + t('vs −E[g]', '与 −E[g] 相比') + '</th><th>' + t('final x', '最终 x') + '</th></tr>' +
    runs.map(r => { const same = r.meanStep < 0; return '<tr><td style="color:' + colorOf(R_B2, r.b2) + '">' + r.b2 + '</td><td>' + r.stepBig.toFixed(2) + 'α</td><td>' + (r.stepSmall >= 0 ? '+' : '') + r.stepSmall.toFixed(2) + 'α</td><td><b>' + (r.meanStep >= 0 ? '+' : '') + r.meanStep.toFixed(3) + 'α</b></td><td style="color:' + (same ? '#008577' : '#d8543d') + '">' + (same ? t('same direction (0°)', '同向（0°）') : t('opposite (180°)', '反向（180°）')) + '</td><td>' + r.x.toFixed(2) + '</td></tr>'; }).join('') +
    '<tr><td>SGD</td><td>−10α</td><td>+1α</td><td><b>−0.100α</b></td><td>—</td><td>' + (runs[0] ? runs[0].xSgd.toFixed(2) : '—') + '</td></tr></table>' +
    '<p class="small">' + t('Steps are measured in units of α and averaged over the second half of the run. A positive expected step moves x towards +1, away from x* = −1.', '步长以 α 为单位，取运行后半段的平均。期望步长为正，表示 x 往 +1 走，远离 x* = −1。') + '</p>';
}
function rArrows(runs) {
  const cv = $('r-dir'), dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 1100, rowH = 46, H = 40 + rowH * (runs.length + 1);
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const L = 150, Rr = W - 30, mid = (L + Rr) / 2, half = (Rr - L) / 2;
  const maxStep = Math.max(3.2, ...runs.map(r => Math.max(Math.abs(r.stepBig), Math.abs(r.stepSmall))));
  const sc = v => mid + half * v / maxStep;
  g.font = '12px system-ui, sans-serif'; g.fillStyle = '#5b6578'; g.textAlign = 'center';
  g.fillText(t('← towards x* = −1', '← 朝向 x* = −1'), L + 70, 16); g.fillText(t('away from x* →', '远离 x* →'), Rr - 60, 16);
  g.strokeStyle = '#d5dbe5'; g.beginPath(); g.moveTo(mid, 24); g.lineTo(mid, H - 6); g.stroke();
  function arrow(y, v, color, width, label) {
    const x1 = sc(v); g.strokeStyle = color; g.fillStyle = color; g.lineWidth = width;
    g.beginPath(); g.moveTo(mid, y); g.lineTo(x1, y); g.stroke();
    const d = Math.sign(v) || 1; if (Math.abs(x1 - mid) > 3) { g.beginPath(); g.moveTo(x1, y); g.lineTo(x1 - d * 8, y - 5); g.lineTo(x1 - d * 8, y + 5); g.closePath(); g.fill(); }
    if (label) { g.font = '11px system-ui, sans-serif'; g.textAlign = v < 0 ? 'right' : 'left'; g.fillText(label, x1 + (v < 0 ? -10 : 10), y + 4); }
  }
  const rows = [{ name: 'SGD', c: '#182237', big: -10 / 1, small: 1, mean: -0.1, note: '' }].concat(runs.map(r => ({ name: 'β₂ ' + r.b2, c: colorOf(R_B2, r.b2), big: r.stepBig, small: r.stepSmall, mean: r.meanStep })));
  const f = Math.max(1, Math.min(8, Math.floor(0.8 * maxStep / Math.max(...rows.map(r => Math.abs(r.mean))))));
  rows.forEach((r, i) => {
    const y = 40 + i * rowH;
    g.fillStyle = r.c; g.font = '600 13px system-ui, sans-serif'; g.textAlign = 'left'; g.fillText(r.name, 8, y + 8);
    g.globalAlpha = 0.35; arrow(y, Math.max(-maxStep, r.big), r.c, 3, ''); arrow(y, r.small, r.c, 3, ''); g.globalAlpha = 1;
    arrow(y + 14, r.mean * f, r.c, 4, ''); g.fillStyle = r.c; g.font = '11px system-ui, sans-serif'; g.textAlign = 'left'; g.fillText(t('E[step] ×', 'E[步长] ×') + f, 8, y + 24);
  });
  g.font = '11px system-ui, sans-serif'; g.fillStyle = '#5b6578'; g.textAlign = 'left';
  g.fillText(t('faint arrows: the two stochastic steps (g = +10 left, g = −1 right) · solid: the expected step, magnified ×' + f + ' · the SGD step −10α is clipped', '淡色箭头：两种随机步（g = +10 向左，g = −1 向右）· 实线：期望一步，放大 ' + f + ' 倍 · SGD 的 −10α 被截断'), 8, H - 4);
}
['r-b1', 'r-alpha', 'r-T', 'r-seed', 'r-sgd'].forEach(id => $(id).addEventListener('change', rRun));
$('r-run').onclick = rRun;

/* ---------------- Panel 2 ---------------- */
const B2_ALL = ['0.3', '0.5', '0.7', '0.8', '0.9', '0.99', '0.999'];
const GRID_B1 = [0, 0.5, 0.8, 0.9, 0.95], GRID_B2 = [0.3, 0.5, 0.7, 0.8, 0.9, 0.99];
/* Presets: verified offline with the same engine, see RESULTS.md. */
const PRESETS = {
  blowup: { n: 1, en: 'Small β₂ blows up (4 × 64)', zh: '小 β₂ 会爆炸（4 × 64）', mode: 'compare', net: '4x64', batch: '8', lr: '0.001', sched: 'constant', steps: '2000', beta1: '0.9', b2: ['0.5', '0.7', '0.9', '0.999'],
    q: ['Predict: β₁ = 0.9 and β₂ ∈ {0.5, 0.7, 0.9, 0.999}. Which runs train, which blow up? Where is the threshold?', '预测：β₁ = 0.9，β₂ ∈ {0.5, 0.7, 0.9, 0.999}。哪些能训练、哪些会爆炸？阈值在哪里？'] },
  rms: { n: 2, en: 'RMSProp (β₁ = 0): larger β₂ is better', zh: 'RMSProp（β₁ = 0）：β₂ 越大越好', mode: 'compare', net: '2x64', batch: '8', lr: '0.001', sched: 'constant', steps: '2000', beta1: '0', b2: ['0.3', '0.7', '0.9', '0.99'],
    q: ['Predict: without momentum nothing diverges. Does β₂ still matter? Small β₂ makes Adam close to signSGD (Panel 1).', '预测：没有动量时不会发散。β₂ 还重要吗？小 β₂ 让 Adam 接近 signSGD（面板 1）。'] },
  ln: { n: 3, en: 'With LayerNorm (4 × 64 + LN)', zh: '加 LayerNorm（4 × 64 + LN）', mode: 'compare', net: '4x64-ln', batch: '8', lr: '0.001', sched: 'constant', steps: '2000', beta1: '0.9', b2: ['0.5', '0.7', '0.9', '0.999'],
    q: ['Predict: the same β₂ values as scenario 1, now with LayerNorm. Does β₂ = 0.5 still blow up? Is large β₂ still best?', '预测：与场景 1 相同的 β₂，但加了 LayerNorm。β₂ = 0.5 还会爆炸吗？大 β₂ 仍然最好吗？'] },
  grid: { n: 4, en: 'β₁ × β₂ phase diagram', zh: 'β₁ × β₂ 相图', mode: 'grid', net: '2x64', batch: '8', lr: '0.001', sched: 'constant', steps: '2000', beta1: '0.9', b2: ['0.5', '0.7', '0.9', '0.999'],
    q: ['Predict: colour the 5 × 6 grid before you run. Where is the red region? Compare with the curve β₁ = √β₂ from the slides.', '预测：运行前先给 5 × 6 网格涂色。红色区域在哪里？与幻灯片中的曲线 β₁ = √β₂ 比较。'] },
  sched: { n: 5, en: 'β₂ under a schedule', zh: '加上学习率调度后的 β₂', mode: 'compare', net: '4x64', batch: '8', lr: '0.001', sched: 'wcosine', steps: '2000', beta1: '0.9', b2: ['0.5', '0.7', '0.9', '0.999'],
    q: ['Predict: warmup + cosine shrinks α at the end. Can a good schedule rescue a bad β₂, or is β₂ still the deciding factor?', '预测：预热 + 余弦会在后期缩小 α。好的调度能挽救坏的 β₂ 吗？还是 β₂ 仍是决定因素？'] }
};
let preset = 'blowup', chosen = PRESETS.blowup.b2.slice(), pool = null, active = null, results = null, running = false;

function mode() { return $('mode').value; }
function schedSpec(c) {
  const s = { lr: c.lr, steps: c.steps };
  if (c.sched === 'plateau') return Object.assign(s, { type: 'plateau', patience: 10, factor: 0.1 });
  if (c.sched === 'wcosine') return Object.assign(s, { type: 'cosine', minFrac: 0, warmup: Math.round(0.05 * c.steps) });
  return Object.assign(s, { type: 'constant' });
}
function config() {
  return { mode: mode(), net: $('net').value, batch: +$('batch').value, lr: +$('lr').value, sched: $('sched').value, steps: +$('steps').value, seed: +$('seed').value, beta1: +$('beta1').value, b2: chosen.slice() };
}
function b2List() {
  $('b2vals').innerHTML = B2_ALL.map(v => '<label class="sched"><input type="checkbox" value="' + v + '"' + (chosen.includes(v) ? ' checked' : '') + (running ? ' disabled' : '') + '><span class="swatch" style="background:' + colorOf(B2_ALL, v) + '"></span><span>β₂ = ' + v + '</span></label>').join('');
  $('b2vals').querySelectorAll('input').forEach(cb => cb.onchange = () => {
    if (cb.checked && chosen.length >= 4) { cb.checked = false; $('status').textContent = t('At most four β₂ values per run.', '每次最多对比 4 个 β₂。'); return; }
    chosen = B2_ALL.filter(v => v === cb.value ? cb.checked : chosen.includes(v)); changed();
  });
}
function netParts(n) { const ln = n.endsWith('-ln'), [L, W] = n.replace('-ln', '').split('x').map(Number); return { L, W, arm: ln ? 'ln' : 'none' }; }
function showMode() {
  const g = mode() === 'grid';
  document.querySelectorAll('.compare-only').forEach(el => el.classList.toggle('hidden', g));
  document.querySelectorAll('.grid-only').forEach(el => el.classList.toggle('hidden', !g));
}
function explain() {
  const c = config(), p = PRESETS[preset], n = netParts(c.net);
  $('hypothesis').textContent = p ? t(p.q[0], p.q[1]) : '';
  $('architecture').textContent = 'x(196) → [Linear → ReLU' + (n.arm === 'ln' ? ' → LN' : '') + '] × ' + n.L + ' (width ' + n.W + ') → Linear(10) → softmax · Adam · α = ' + c.lr;
}
function presetButtons() {
  $('presets').innerHTML = Object.entries(PRESETS).map(([k, p]) => '<button data-preset="' + k + '"' + (k === preset ? ' class="selected"' : '') + '>' + p.n + '. ' + t(p.en, p.zh) + '</button>').join('');
  document.querySelectorAll('[data-preset]').forEach(b => { b.disabled = running; b.onclick = () => applyPreset(b.dataset.preset); });
}
function applyPreset(k) {
  preset = k; const p = PRESETS[k];
  ['mode', 'net', 'batch', 'lr', 'sched', 'steps', 'beta1'].forEach(id => $(id).value = p[id]);
  chosen = p.b2.slice(); b2List(); showMode(); presetButtons(); explain();
  results = null; active = null; render();
}
function changed() {
  preset = null; presetButtons(); showMode(); explain();
  if (results) $('status').textContent = t('Settings changed. The results still show the previous run.', '设置已改变。结果仍显示上一次运行。'); else render();
}
function render() {
  const c = active || config();
  if (c.mode === 'grid') return renderGrid(c);
  const keys = c.b2, res = results || [], steps = c.steps;
  $('legend').innerHTML = keys.map(v => '<span style="color:' + colorOf(B2_ALL, v) + '">● β₂ = ' + v + '</span>').join('') + '<span style="color:#5b6578">β₁ = ' + c.beta1 + '</span>';
  const ser = (fn, lbl) => keys.map((v, i) => ({ color: colorOf(B2_ALL, v), label: lbl ? 'β₂ ' + v : null, data: (res[i] ? res[i].history : []).map(fn).filter(p => p && isFinite(p[1])) }));
  const loss = ser(m => m.train ? [m.step, Math.min(10, m.train.loss)] : null, true);
  K.chart('loss', loss, { xMax: steps, yMin: 0.01, yMax: 10, log: true, xLabel: t('Adam updates', 'Adam 更新次数'), labels: true });
  K.chart('acc', ser(m => m.test ? [m.step, m.test.acc] : null), { xMax: steps, yMin: 0, yMax: 1, xLabel: t('Adam updates', 'Adam 更新次数'), yFmt: v => Math.round(100 * v) + '%' });
  const finals = res.map(r => r && r.final && !r.failed ? r.final.train.loss : Infinity), best = Math.min(...finals);
  $('cards').innerHTML = keys.map((v, i) => {
    const x = res[i], f = x && x.final;
    let body;
    if (x && x.error) body = '<p class="small">' + x.error + '</p>';
    else if ((x && x.failed) || (f && f.train.loss > 10)) body = '<div class="metric" style="color:#d8543d">' + t('Blew up', '爆炸') + '</div><p class="small">' + (x.failed ? t('Non-finite loss at update ', '第 ') + x.failedAt + t('', ' 次更新出现非有限值') : t('Final training loss ', '最终训练损失 ') + K.fmt(f.train.loss, 2)) + '</p>' + (f ? '<dl class="kv"><dt>' + t('Train / test accuracy', '训练 / 测试准确率') + '</dt><dd>' + K.pct(f.train.acc) + ' / ' + K.pct(f.test.acc) + '</dd></dl>' : '');
    else body = '<div class="metric">' + (f ? K.fmt(f.train.loss, 3) : '—') + (f && f.train.loss === best && res.filter(r => r && r.final).length > 1 ? ' <span class="small">' + t('lowest', '最低') + '</span>' : '') + '</div><p class="small">' + t('Final training loss (all 4000 digits)', '最终训练损失（全部 4000 个样本）') + '</p>' +
      '<dl class="kv"><dt>' + t('Train / test accuracy', '训练 / 测试准确率') + '</dt><dd>' + (f ? K.pct(f.train.acc) + ' / ' + K.pct(f.test.acc) : '—') + '</dd><dt>β₁ &lt; √β₂ ?</dt><dd>' + (c.beta1 < Math.sqrt(+v) ? t('yes', '是') : t('no', '否')) + '</dd></dl>';
    return '<article class="model-card" style="--color:' + colorOf(B2_ALL, v) + '"><h3>β₂ = ' + v + '</h3>' + body + '</article>';
  }).join('');
}
function cellColor(L) {
  if (!isFinite(L) || L > 1) return '#f6c9bd';
  const s = Math.max(0, Math.min(1, (Math.log10(L) - Math.log10(0.08)) / (Math.log10(0.3) - Math.log10(0.08))));
  const a = [168, 230, 207], b = [252, 236, 200]; return 'rgb(' + a.map((x, i) => Math.round(x + (b[i] - x) * s)).join(',') + ')';
}
function renderGrid(c) {
  const res = results || [];
  let h = '<table class="heat"><tr><th>β₁ \\ β₂</th>' + GRID_B2.map(b => '<th>' + b + '</th>').join('') + '</tr>';
  GRID_B1.forEach((b1, i) => {
    h += '<tr><th>' + b1 + '</th>' + GRID_B2.map((b2, j) => {
      const r = res[i * GRID_B2.length + j]; let L = null, txt = '';
      if (r && (r.final || r.failed)) { L = r.failed ? Infinity : r.final.train.loss; txt = !isFinite(L) ? '∞' : L > 10 ? '≫1' : L.toFixed(L < 1 ? 3 : 1); }
      else if (r && r.history.length) txt = '…';
      const edge = b1 >= Math.sqrt(b2) ? 'box-shadow:inset 0 0 0 2px #182237;' : '';
      return '<td style="' + edge + 'background:' + (L == null ? '#eef1f5' : cellColor(L)) + '" title="β₁ ' + b1 + ', β₂ ' + b2 + '">' + txt + '</td>';
    }).join('') + '</tr>';
  });
  $('heat').innerHTML = h + '</table><p class="small">' + t('Outlined cells have β₁ ≥ √β₂ (outside the convergence region of the theory). Values: final training loss after ', '加框的格子满足 β₁ ≥ √β₂（在理论的收敛区域之外）。数值：') + c.steps + t(' updates.', ' 次更新后的最终训练损失。') + '</p>';
}
function setRunning(v) {
  running = v; $('run').disabled = v; $('stop').disabled = !v;
  document.querySelectorAll('#panel2 .controls select,#panel2 .controls input,[data-preset]').forEach(el => el.disabled = v);
}
function run() {
  const c = config();
  if (!Number.isInteger(c.seed) || c.seed < 1 || c.seed > 99999) { $('status').textContent = t('Seed must be an integer from 1 to 99999.', '种子必须是 1–99999 的整数。'); return; }
  if (c.mode === 'compare' && !c.b2.length) { $('status').textContent = t('Choose at least one β₂.', '至少选择一个 β₂。'); return; }
  active = c; results = null; setRunning(true); $('export').disabled = true; $('progress').value = 0;
  const n = netParts(c.net), grid = c.mode === 'grid';
  const pairs = grid ? GRID_B1.flatMap(b1 => GRID_B2.map(b2 => [b1, b2])) : c.b2.map(v => [c.beta1, +v]);
  const every = grid ? Math.max(50, Math.round(c.steps / 10)) : Math.max(10, Math.round(c.steps / 100));
  const runs = pairs.map(([b1, b2], i) => ({ id: i, msg: { cfg: { depth: n.L, width: n.W, arm: n.arm, seed: c.seed, batch: c.batch, steps: c.steps, opt: { type: 'adam', beta1: b1, beta2: b2, eps: 1e-8 }, sched: schedSpec(c) }, every } }));
  $('status').textContent = t('Loading the digits…', '正在加载数据……');
  pool = new K.Pool(res => {
    results = res;
    const done = res.reduce((s, r) => s + (r.final || r.failed ? 1 : 0), 0);
    $('progress').value = Math.round(100 * res.reduce((s, r) => s + (r.final || r.failed ? c.steps : (r.history.length ? r.history[r.history.length - 1].step : 0)), 0) / (c.steps * res.length));
    $('status').textContent = t('Training · ', '训练中 · ') + done + ' / ' + res.length + t(' runs finished', ' 次运行已完成'); render();
  }, res => { results = res; setRunning(false); $('export').disabled = false; $('progress').value = 100; $('status').textContent = t('Finished.', '完成。'); render(); });
  pool.start(runs, { ntrain: 4000, ntest: 2000, evalN: grid ? 500 : 1000, noiseM: 0 }).catch(err => { setRunning(false); $('status').textContent = String(err.message || err); });
}
$('run').onclick = run;
$('stop').onclick = () => { if (pool) pool.stop(); setRunning(false); $('export').disabled = !results; $('status').textContent = t('Stopped. The next run starts fresh.', '已停止。下次运行将从头开始。'); };
$('export').onclick = () => { if (!results) return; K.download('adam-beta2-lab-results.json', { lab: 'adam-beta2-lab', version: 1, settings: active, schedule: schedSpec(active), grid: active.mode === 'grid' ? { beta1: GRID_B1, beta2: GRID_B2 } : null, results: results.map(r => ({ failed: r.failed, failedAt: r.failedAt, final: r.final, history: r.history })) }); };
['mode', 'net', 'batch', 'lr', 'sched', 'steps', 'seed', 'beta1'].forEach(id => $(id).addEventListener('change', changed));

/* ---------------- panel switching ---------------- */
function show(p) {
  $('panel1').classList.toggle('hidden', p !== 1); $('panel2').classList.toggle('hidden', p !== 2);
  $('nav1').classList.toggle('active', p === 1); $('nav2').classList.toggle('active', p === 2);
  if (p === 1) rRender(); else render();
  window.scrollTo({ top: $('panel' + p).offsetTop - 70, behavior: 'smooth' });
}
$('to2').onclick = () => show(2); $('to1').onclick = () => show(1);
$('nav1').onclick = e => { e.preventDefault(); show(1); }; $('nav2').onclick = e => { e.preventDefault(); show(2); };
if (location.hash === '#panel2') setTimeout(() => show(2), 0);

K.initLang(() => { rList(); rRender(); presetButtons(); b2List(); explain(); render(); if (!running && !results) $('status').textContent = t('Ready. Choose a scenario, predict, then run.', '准备就绪。选择一个场景，先预测，再运行。'); });
$('status').textContent = t('Ready. Choose a scenario, predict, then run.', '准备就绪。选择一个场景，先预测，再运行。');
rList(); rRun(); applyPreset('blowup');
