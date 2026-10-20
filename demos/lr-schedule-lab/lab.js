'use strict';
/* Lab 2 · Learning-rate schedules on a real MLP (ME2605 Lecture 6, Part II). */
const K = LabKit, $ = K.$, t = K.t;
const SCHEDS = {
  constant:  { en: 'Constant α₀', zh: '常数 α₀', color: '#d8543d' },
  constant10:{ en: 'Constant α₀/10', zh: '常数 α₀/10', color: '#9aa5b5' },
  step:      { en: 'Step decay (50%, 75%)', zh: '阶梯衰减（50%、75%）', color: '#7c3aed' },
  plateau:   { en: 'Reduce on plateau', zh: '平台衰减', color: '#c2410c' },
  invt:      { en: '1/t decay', zh: '1/t 衰减', color: '#64748b' },
  cosine:    { en: 'Cosine', zh: '余弦', color: '#3262ed' },
  wcosine:   { en: 'Warmup + cosine', zh: '预热 + 余弦', color: '#008577' },
  wsd:       { en: 'WSD', zh: 'WSD', color: '#b7791f' }
};
/* Presets: verified with seeds 7, 11, 23 (see RESULTS.md). */
const PRESETS = {
  stuck: { n: 1, en: 'Stuck at a constant α → divide by 10', zh: '常数步长卡住 → 除以 10',
    net: '1x16', arm: 'none', lr: '0.4', batch: '8', steps: '5000', factor: '0.1', warmup: '0.05', scheds: ['constant', 'step', 'plateau'],
    q: ['Predict: with α = 0.4 the loss stops improving after a few hundred updates. What happens to the loss at the moment α is divided by 10?',
        '预测：α = 0.4 时，几百步之后损失就不再下降。α 被除以 10 的那一刻，损失会怎样？'] },
  cosine: { n: 2, en: 'Warmup + cosine vs the rest', zh: '预热 + 余弦 vs 其他调度',
    net: '1x16', arm: 'none', lr: '0.4', batch: '8', steps: '5000', factor: '0.1', warmup: '0.05', scheds: ['constant', 'constant10', 'step', 'wcosine'],
    q: ['Predict: four schedules, same peak α₀ = 0.4 and the same 5000 updates. Rank them by final training loss before you run.',
        '预测：四种调度，相同的峰值 α₀ = 0.4、相同的 5000 次更新。运行前，按最终训练损失给它们排个序。'] },
  deeper: { n: 3, en: 'A deeper network (4 × 32)', zh: '更深的网络（4 × 32）',
    net: '4x32', arm: 'none', lr: '0.15', batch: '8', steps: '4000', factor: '0.1', warmup: '0.05', scheds: ['constant', 'constant10', 'step', 'wcosine'],
    q: ['Predict: does the ranking from scenario 2 survive on a deeper network with a smaller peak stepsize?',
        '预测：在更深的网络、更小的峰值步长下，场景 2 的排序还成立吗？'] },
  wsd: { n: 4, en: 'Cosine vs WSD vs 1/t', zh: '余弦 vs WSD vs 1/t',
    net: '1x16', arm: 'none', lr: '0.4', batch: '8', steps: '5000', factor: '0.1', warmup: '0.05', scheds: ['wcosine', 'wsd', 'invt', 'plateau'],
    q: ['Predict: WSD stays at α₀ for 80% of training and decays only at the end; 1/t decays from the very start. Which one ends lower, and why?',
        '预测：WSD 在前 80% 保持 α₀，只在最后衰减；1/t 从一开始就衰减。哪一个最终更低？为什么？'] }
};
let preset = 'stuck', pool = null, active = null, results = null, running = false;
let chosen = PRESETS.stuck.scheds.slice();

function schedSpec(key, c) {
  const T = c.steps, s = { lr: c.lr, steps: T };
  if (key === 'constant') return Object.assign(s, { type: 'constant' });
  if (key === 'constant10') return Object.assign(s, { type: 'constant', lr: c.lr / 10 });
  if (key === 'step') return Object.assign(s, { type: 'step', milestones: [0.5, 0.75], gamma: c.factor });
  if (key === 'plateau') return Object.assign(s, { type: 'plateau', patience: 10, factor: c.factor });
  if (key === 'invt') return Object.assign(s, { type: 'invt', t0: Math.max(1, Math.round(T / 20)) });
  if (key === 'cosine') return Object.assign(s, { type: 'cosine', minFrac: 0 });
  if (key === 'wcosine') return Object.assign(s, { type: 'cosine', minFrac: 0, warmup: Math.round(T * c.warmup) });
  if (key === 'wsd') return Object.assign(s, { type: 'wsd', minFrac: 0, warmup: Math.round(T * c.warmup), decayFrac: 0.2 });
}
function config() {
  return { net: $('net').value, arm: $('arm').value, lr: +$('lr').value, batch: +$('batch').value, steps: +$('steps').value,
    seed: +$('seed').value, factor: +$('factor').value, warmup: +$('warmup').value, scheds: chosen.slice() };
}
function schedList() {
  $('sched-list').innerHTML = Object.entries(SCHEDS).map(([k, s]) => '<label class="sched"><input type="checkbox" value="' + k + '"' + (chosen.includes(k) ? ' checked' : '') + (running ? ' disabled' : '') + '><span class="swatch" style="background:' + s.color + '"></span><span>' + t(s.en, s.zh) + '</span></label>').join('');
  $('sched-list').querySelectorAll('input').forEach(cb => cb.onchange = () => {
    if (cb.checked && chosen.length >= 4) { cb.checked = false; $('status').textContent = t('At most four schedules per run.', '每次最多对比 4 种调度。'); return; }
    chosen = Object.keys(SCHEDS).filter(k => (k === cb.value ? cb.checked : chosen.includes(k)));
    changed();
  });
}
function explain() {
  const c = config(), p = PRESETS[preset], [L] = c.net.split('x');
  $('hypothesis').textContent = p ? t(p.q[0], p.q[1]) : '';
  $('architecture').textContent = 'x(196) → [Linear → ReLU' + (c.arm === 'ln' ? ' → LN' : '') + '] × ' + L + ' → Linear(10) → softmax · SGD · α₀ = ' + c.lr;
  $('exp-title').textContent = c.scheds.length === 0 ? t('Choose at least one schedule.', '至少选择一种调度。') : t('Train ' + c.scheds.length + ' schedule' + (c.scheds.length > 1 ? 's' : '') + ', side by side.', '并排训练 ' + c.scheds.length + ' 种调度。');
}
function presetButtons() {
  $('presets').innerHTML = Object.entries(PRESETS).map(([k, p]) => '<button data-preset="' + k + '"' + (k === preset ? ' class="selected"' : '') + '>' + p.n + '. ' + t(p.en, p.zh) + '</button>').join('');
  document.querySelectorAll('[data-preset]').forEach(b => { b.disabled = running; b.onclick = () => applyPreset(b.dataset.preset); });
}
function applyPreset(k) {
  preset = k; const p = PRESETS[k];
  ['net', 'arm', 'lr', 'batch', 'steps', 'factor', 'warmup'].forEach(id => $(id).value = p[id]);
  chosen = p.scheds.slice(); schedList(); presetButtons(); explain();
  if (results) $('status').textContent = t('Settings changed. The curves still show the previous run.', '设置已改变。曲线仍显示上一次运行。'); else render();
}
function changed() {
  preset = null; presetButtons(); explain();
  if (results) $('status').textContent = t('Settings changed. The curves still show the previous run.', '设置已改变。曲线仍显示上一次运行。'); else render();
}
function render() {
  const c = active || config(), keys = c.scheds, steps = c.steps, res = results || [];
  $('legend').innerHTML = keys.map(k => '<span style="color:' + SCHEDS[k].color + '">● ' + t(SCHEDS[k].en, SCHEDS[k].zh) + '</span>').join('');
  const ser = (fn, lbl) => keys.map((k, i) => ({ color: SCHEDS[k].color, label: lbl ? t(SCHEDS[k].en, SCHEDS[k].zh) : null, data: (res[i] ? res[i].history : []).map(fn).filter(p => p && isFinite(p[1])) }));
  const loss = ser(m => m.train ? [m.step, m.train.loss] : null, true), lv = loss.flatMap(s => s.data.map(p => p[1])), lr = K.range(lv, 0.03, 1, true);
  K.chart('loss', loss, { xMax: steps, yMin: Math.max(1e-4, lr[0]), yMax: Math.max(1, lr[1]), log: true, xLabel: t('SGD updates', 'SGD 更新次数'), labels: true });
  /* the planned schedule (exact for all but the plateau rule, which is drawn from the run) */
  const lrS = keys.map((k, i) => {
    let data;
    if (k === 'plateau') data = (res[i] ? res[i].history : []).filter(m => m.lr != null).map(m => [m.step, m.lr]);
    else { const sp = schedSpec(k, c); data = []; for (let s = 1; s <= steps; s += Math.max(1, Math.floor(steps / 300))) data.push([s, OptEngine.scheduleLR(sp, s)]); data.push([steps, OptEngine.scheduleLR(sp, steps)]); }
    return { color: SCHEDS[k].color, data: data.map(p => [p[0], Math.max(p[1], c.lr * 1e-3)]), dash: k === 'constant10' ? [6, 4] : null };
  });
  K.chart('lrc', lrS, { xMax: steps, yMin: c.lr * 1e-3, yMax: c.lr * 1.5, log: true, xLabel: t('SGD updates', 'SGD 更新次数'), yFmt: v => String(+v.toPrecision(1)) });
  K.chart('acc', ser(m => m.test ? [m.step, m.test.acc] : null), { xMax: steps, yMin: 0.6, yMax: 1, xLabel: t('SGD updates', 'SGD 更新次数'), yFmt: v => Math.round(100 * v) + '%' });
  const finals = res.map(r => r && r.final && !r.failed ? r.final.train.loss : Infinity), best = Math.min(...finals);
  $('cards').innerHTML = keys.map((k, i) => {
    const x = res[i], f = x && x.final, S = SCHEDS[k];
    let body;
    if (x && x.error) body = '<p class="small">' + x.error + '</p>';
    else if (x && x.failed) body = '<div class="metric">' + t('Diverged', '发散') + '</div><p class="small">' + t('Non-finite loss at update ', '第 ') + x.failedAt + t('', ' 次更新出现非有限值') + '</p>';
    else body = '<div class="metric">' + (f ? K.fmt(f.train.loss, 3) : '—') + (f && f.train.loss === best && res.filter(r => r && r.final).length > 1 ? ' <span class="small">' + t('lowest', '最低') + '</span>' : '') + '</div><p class="small">' + t('Final training loss (all 4000 digits)', '最终训练损失（全部 4000 个样本）') + '</p>' +
      '<dl class="kv"><dt>' + t('Train / test accuracy', '训练 / 测试准确率') + '</dt><dd>' + (f ? K.pct(f.train.acc) + ' / ' + K.pct(f.test.acc) : '—') + '</dd>' +
      (k === 'plateau' ? '<dt>' + t('α divided at update', 'α 缩小于第') + '</dt><dd>' + (x && x.history.length ? ((x.final || x.history[x.history.length - 1]).drops || []).join(', ') || '—' : '—') + '</dd>' : '') + '</dl>';
    return '<article class="model-card" style="--color:' + S.color + '"><h3>' + t(S.en, S.zh) + '</h3>' + body + '</article>';
  }).join('');
}
function setRunning(v) {
  running = v; $('run').disabled = v; $('stop').disabled = !v;
  document.querySelectorAll('.controls select,.controls input,[data-preset]').forEach(el => el.disabled = v);
}
function run() {
  const c = config();
  if (!Number.isInteger(c.seed) || c.seed < 1 || c.seed > 99999) { $('status').textContent = t('Seed must be an integer from 1 to 99999.', '种子必须是 1–99999 的整数。'); return; }
  if (!c.scheds.length) { $('status').textContent = t('Choose at least one schedule.', '至少选择一种调度。'); return; }
  active = c; results = null; setRunning(true); $('export').disabled = true; $('progress').value = 0;
  const every = Math.max(10, Math.round(c.steps / 100)), [L, W] = c.net.split('x').map(Number);
  $('status').textContent = t('Loading the digits…', '正在加载数据……');
  const runs = c.scheds.map((k, i) => ({ id: i, msg: { cfg: { depth: L, width: W, arm: c.arm, seed: c.seed, batch: c.batch, steps: c.steps, opt: { type: 'sgd' }, sched: schedSpec(k, c) }, every } }));
  pool = new K.Pool(res => {
    results = res;
    const done = res.reduce((s, r) => s + (r.final || r.error ? 1 : 0), 0);
    $('progress').value = Math.round(100 * res.reduce((s, r) => s + (r.final ? c.steps : (r.history.length ? r.history[r.history.length - 1].step : 0)), 0) / (c.steps * res.length));
    $('status').textContent = t('Training · ', '训练中 · ') + done + ' / ' + res.length + t(' schedules finished', ' 种调度已完成'); render();
  }, res => { results = res; setRunning(false); $('export').disabled = false; $('progress').value = 100; $('status').textContent = t('Finished. The lowest final training loss is marked.', '完成。最终训练损失最低的已标出。'); render(); });
  pool.start(runs, { ntrain: 4000, ntest: 2000, evalN: 1000, noiseM: 0 }).catch(err => { setRunning(false); $('status').textContent = String(err.message || err); });
}
$('run').onclick = run;
$('stop').onclick = () => { if (pool) pool.stop(); setRunning(false); $('export').disabled = !results; $('status').textContent = t('Stopped. The next run starts fresh.', '已停止。下次运行将从头开始。'); };
$('export').onclick = () => { if (!results) return; K.download('lr-schedule-lab-results.json', { lab: 'lr-schedule-lab', version: 1, settings: active, schedules: active.scheds.map(k => schedSpec(k, active)), results: results.map(r => ({ failed: r.failed, failedAt: r.failedAt, final: r.final, history: r.history })) }); };
['net', 'arm', 'lr', 'batch', 'steps', 'seed', 'factor', 'warmup'].forEach(id => $(id).addEventListener('change', changed));
K.initLang(() => { presetButtons(); schedList(); explain(); render(); if (!running && !results) $('status').textContent = t('Ready. Choose a scenario, predict, then run.', '准备就绪。选择一个场景，先预测，再运行。'); });
$('status').textContent = t('Ready. Choose a scenario, predict, then run.', '准备就绪。选择一个场景，先预测，再运行。');
applyPreset('stuck');
