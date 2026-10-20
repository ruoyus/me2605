'use strict';
/* Lab 1 · Constant-stepsize SGD: noise floor vs interpolation (ME2605 Lecture 6, Part I). */
const K = LabKit, $ = K.$, t = K.t;
const COLORS = ['#d8543d', '#3262ed', '#008577', '#7c3aed'];
const OPTIONS = { lr: ['0.025', '0.05', '0.1', '0.2', '0.4'], batch: ['4', '8', '16', '64'], arm: ['none', 'ln'], net: ['1x8', '1x16', '2x16', '2x128', '4x64'] };
const ARM_NAME = { none: ['No normalization', '无归一化'], ln: ['LayerNorm', 'LayerNorm'] };

/* Presets: verified with seeds 7 and 11 (see RESULTS.md). */
const PRESETS = {
  small_lr: { n: 1, en: 'Small network · three stepsizes', zh: '小网络 · 三个步长',
    net: '1x8', arm: 'none', lr: '0.2', batch: '16', steps: '5000', compare: 'lr', values: ['0.4', '0.2', '0.1'],
    q: ['Predict: a 1 × 8 network cannot fit 4000 digits. With a constant stepsize, will its training loss settle? How will the three curves differ?',
        '预测：1 × 8 的网络无法拟合 4000 个手写数字。常数步长下，它的训练损失会稳定下来吗？三条曲线会有什么不同？'] },
  small_batch: { n: 2, en: 'Small network · three batch sizes', zh: '小网络 · 三个批大小',
    net: '1x8', arm: 'none', lr: '0.2', batch: '16', steps: '5000', compare: 'batch', values: ['4', '16', '64'],
    q: ['Predict: same network and stepsize, batch size 4, 16 or 64. Which floor is lowest, and which curve jitters most?',
        '预测：同样的网络和步长，批大小取 4、16、64。哪条曲线的地板最低？哪条抖得最厉害？'] },
  wide_lr: { n: 3, en: 'Wide network · interpolation', zh: '宽网络 · 插值',
    net: '2x128', arm: 'none', lr: '0.1', batch: '16', steps: '5000', compare: 'lr', values: ['0.2', '0.1', '0.05'],
    q: ['Predict: a 2 × 128 network has 43k parameters for 4000 digits. Will constant-stepsize SGD still fluctuate? Watch σ².',
        '预测：2 × 128 的网络有 4.3 万个参数，只有 4000 个样本。常数步长的 SGD 还会振荡吗？注意观察 σ²。'] },
  small_ln: { n: 4, en: 'Small network + LayerNorm', zh: '小网络 + LayerNorm',
    net: '1x8', arm: 'ln', lr: '0.2', batch: '16', steps: '5000', compare: 'lr', values: ['0.4', '0.2', '0.1'],
    q: ['Predict: add LayerNorm to the small network. Does normalization remove the noise floor?',
        '预测：给小网络加上 LayerNorm。归一化能消除噪声地板吗？'] },
  small_ln_batch: { n: 5, en: 'Small network + LayerNorm · three batch sizes', zh: '小网络 + LayerNorm · 三个批大小',
    net: '1x8', arm: 'ln', lr: '0.2', batch: '16', steps: '5000', compare: 'batch', values: ['4', '16', '64'],
    q: ['Predict: with LayerNorm, does a larger batch still lower the floor?',
        '预测：加了 LayerNorm 之后，更大的批还能降低地板吗？'] },
  deep_ln_lr: { n: 6, en: 'Deep network + LayerNorm · three stepsizes', zh: '深网络 + LayerNorm · 三个步长',
    net: '4x64', arm: 'ln', lr: '0.05', batch: '16', steps: '5000', compare: 'lr', values: ['0.1', '0.05', '0.025'],
    q: ['Predict: the deep normalized network can fit the data. Do the three stepsizes now end at different floors?',
        '预测：带归一化的深网络能拟合数据。三个步长现在会停在不同的地板上吗？'] }
};
let preset = 'small_lr', pool = null, active = null, results = null, running = false;

function label(compare, v) {
  if (compare === 'lr') return 'α = ' + v;
  if (compare === 'batch') return 'B = ' + v;
  if (compare === 'arm') return t(ARM_NAME[v][0], ARM_NAME[v][1]);
  return v.replace('x', ' × ');
}
function optionText(compare, v) { return compare === 'arm' ? t(ARM_NAME[v][0], ARM_NAME[v][1]) : (compare === 'net' ? v.replace('x', ' × ') : v); }
function buildValueSelectors(values) {
  const c = $('compare').value, opts = OPTIONS[c];
  const vals = values || [opts[opts.length - 1], opts[Math.floor(opts.length / 2)], opts[0]];
  $('values').innerHTML = [0, 1, 2].map(i => '<select id="v' + i + '" aria-label="Run ' + (i + 1) + '"><option value="">—</option>' +
    opts.map(o => '<option value="' + o + '"' + (vals[i] === o ? ' selected' : '') + '>' + optionText(c, o) + '</option>').join('') + '</select>').join('');
  [0, 1, 2].forEach(i => { $('v' + i).style.borderTop = '3px solid ' + COLORS[i]; $('v' + i).onchange = changed; });
  ['lr', 'batch', 'arm', 'net'].forEach(k => { $(k).disabled = running || k === c; });
  $('values-note').textContent = t('Runs to compare (choose “—” to skip one). The setting above that is being compared is ignored.', '要对比的运行（选“—”可跳过）。被对比的那一项，上方的设置不起作用。');
}
function config() {
  const c = $('compare').value;
  const vals = [0, 1, 2].map(i => $('v' + i).value).filter(Boolean);
  const base = { net: $('net').value, arm: $('arm').value, lr: $('lr').value, batch: $('batch').value };
  return { compare: c, values: vals, base, steps: +$('steps').value, seed: +$('seed').value,
    runs: vals.map((v, i) => { const s = Object.assign({}, base); s[c] = v; return { id: i, label: label(c, v), color: COLORS[i], s }; }) };
}
function explain() {
  const c = config(), p = PRESETS[preset];
  $('hypothesis').textContent = p ? t(p.q[0], p.q[1]) : '';
  const s = c.runs.length ? c.runs[0].s : c.base, [L, W] = (c.compare === 'net' ? c.base.net : s.net).split('x');
  $('architecture').textContent = 'x(196) → [Linear → ReLU' + (s.arm === 'ln' && c.compare !== 'arm' ? ' → LN' : c.compare === 'arm' ? ' (→ LN)' : '') + '] × ' + (c.compare === 'net' ? 'L' : L) + ' → Linear(10) → softmax · SGD';
  $('exp-title').textContent = c.runs.length === 0 ? t('Choose at least one run.', '至少选择一次运行。') : c.runs.length === 1 ? t('Train one run.', '训练一次运行。') : t('Train ' + c.runs.length + ' runs, side by side.', '并排训练 ' + c.runs.length + ' 次运行。');
}
function presetButtons() {
  $('presets').innerHTML = Object.entries(PRESETS).map(([k, p]) => '<button data-preset="' + k + '"' + (k === preset ? ' class="selected"' : '') + '>' + p.n + '. ' + t(p.en, p.zh) + '</button>').join('');
  document.querySelectorAll('[data-preset]').forEach(b => b.onclick = () => applyPreset(b.dataset.preset));
}
function applyPreset(k) {
  preset = k; const p = PRESETS[k];
  ['net', 'arm', 'lr', 'batch', 'steps'].forEach(id => $(id).value = p[id]);
  $('compare').value = p.compare; buildValueSelectors(p.values);
  presetButtons(); explain();
  if (results) $('status').textContent = t('Settings changed. The curves still show the previous run.', '设置已改变。曲线仍显示上一次运行。'); else render();
}
function changed() {
  preset = null; presetButtons(); explain();
  if (results) $('status').textContent = t('Settings changed. The curves still show the previous run.', '设置已改变。曲线仍显示上一次运行。'); else render();
}

/* fluctuation = half the range of the training loss over the last 30% of checkpoints */
function stats(r) {
  const h = r.history.filter(m => m.type === 'progress' && m.train && isFinite(m.train.loss));
  if (h.length < 4) return null;
  const tail = h.slice(Math.floor(h.length * 0.7)).map(m => m.train.loss);
  const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
  return { mean, half: (Math.max(...tail) - Math.min(...tail)) / 2 };
}
function render() {
  const runs = active ? active.runs : config().runs, steps = active ? active.steps : +$('steps').value;
  $('legend').innerHTML = runs.map(r => '<span style="color:' + r.color + '">● ' + r.label + '</span>').join('');
  const res = results || [];
  const series = (fn) => runs.map((r, i) => ({ color: r.color, label: r.label, data: (res[i] ? res[i].history : []).filter(m => m.train || m.noise != null).map(fn).filter(p => p && isFinite(p[1])) }));
  const lossS = series(m => m.train ? [m.step, m.train.loss] : null);
  const lv = lossS.flatMap(s => s.data.map(p => p[1]));
  const lr = K.range(lv, 0.01, 1, true);
  K.chart('loss', lossS, { xMax: steps, yMin: Math.max(1e-5, lr[0]), yMax: Math.max(lr[1], 1), log: true, xLabel: t('SGD updates', 'SGD 更新次数'), labels: true });
  const nz = series(m => m.noise != null ? [m.step, m.noise] : null), nv = nz.flatMap(s => s.data.map(p => p[1])), nr = K.range(nv, 0.1, 100, true);
  K.chart('noise', nz, { xMax: steps, yMin: Math.max(1e-6, nr[0]), yMax: Math.max(nr[1], 10), log: true, xLabel: t('SGD updates', 'SGD 更新次数') });
  const ac = series(m => m.test ? [m.step, m.test.acc] : null);
  K.chart('acc', ac, { xMax: steps, yMin: 0.5, yMax: 1, xLabel: t('SGD updates', 'SGD 更新次数'), yFmt: v => Math.round(100 * v) + '%' });
  $('cards').innerHTML = runs.map((r, i) => {
    const x = res[i], f = x && x.final, st = x ? stats(x) : null;
    let body;
    if (x && x.error) body = '<p class="small">' + x.error + '</p>';
    else if (x && x.failed) body = '<div class="metric">' + t('Diverged', '发散') + '</div><p class="small">' + t('Non-finite loss at update ', '第 ') + x.failedAt + t('', ' 次更新出现非有限值') + '</p>';
    else body = '<div class="metric">' + (f ? K.fmt(f.train.loss, 3) : (st ? '≈' + K.fmt(st.mean, 3) : '—')) + '</div><p class="small">' + t(f ? 'Final training loss (all 4000 digits)' : 'Training loss (running)', f ? '最终训练损失（全部 4000 个样本）' : '训练损失（进行中）') + '</p>' +
      '<dl class="kv"><dt>' + t('Fluctuation, last 30%', '后 30% 的振幅') + '</dt><dd>' + (st ? '± ' + K.fmt(st.half, 3) : '—') + '</dd>' +
      '<dt>' + t('Gradient noise σ²', '梯度噪声 σ²') + '</dt><dd>' + (f && f.noise != null ? K.fmt(f.noise, 2) : '—') + '</dd>' +
      '<dt>' + t('Train / test accuracy', '训练 / 测试准确率') + '</dt><dd>' + (f ? K.pct(f.train.acc) + ' / ' + K.pct(f.test.acc) : '—') + '</dd></dl>';
    return '<article class="model-card" style="--color:' + r.color + '"><h3>' + r.label + ' <span class="small">· ' + [active && active.compare === 'net' ? null : r.s.net.replace('x', ' × '), r.s.arm === 'ln' && !(active && active.compare === 'arm') ? 'LN' : null, active && active.compare === 'lr' ? null : 'α ' + r.s.lr, active && active.compare === 'batch' ? null : 'B ' + r.s.batch].filter(Boolean).join(' · ') + '</span></h3>' + body + '</article>';
  }).join('');
}
function setRunning(v) {
  running = v; $('run').disabled = v; $('stop').disabled = !v;
  document.querySelectorAll('.controls select,.controls input,[data-preset]').forEach(el => el.disabled = v);
  if (!v) buildValueSelectorsKeep();
}
function buildValueSelectorsKeep() { const keep = [0, 1, 2].map(i => $('v' + i) ? $('v' + i).value : ''); buildValueSelectors(keep); }
function run() {
  const c = config();
  if (!Number.isInteger(c.seed) || c.seed < 1 || c.seed > 99999) { $('status').textContent = t('Seed must be an integer from 1 to 99999.', '种子必须是 1–99999 的整数。'); return; }
  if (!c.runs.length) { $('status').textContent = t('Choose at least one run.', '至少选择一次运行。'); return; }
  active = c; results = null; setRunning(true); $('export').disabled = true; $('progress').value = 0;
  const every = Math.max(10, Math.round(c.steps / 100)), noiseEvery = Math.max(every, Math.round(c.steps / 10 / every) * every);
  $('status').textContent = t('Loading the digits…', '正在加载数据……');
  const runs = c.runs.map(r => {
    const [L, W] = r.s.net.split('x').map(Number);
    return { id: r.id, msg: { cfg: { depth: L, width: W, arm: r.s.arm, seed: c.seed, batch: +r.s.batch, steps: c.steps, opt: { type: 'sgd' }, sched: { type: 'constant', lr: +r.s.lr, steps: c.steps } }, every, noiseEvery } };
  });
  pool = new K.Pool(res => {
    results = res;
    const done = res.reduce((s, r) => s + (r.final || r.error ? 1 : 0), 0);
    const prog = res.reduce((s, r) => s + (r.final ? c.steps : (r.history.length ? r.history[r.history.length - 1].step : 0)), 0) / (c.steps * res.length);
    $('progress').value = Math.round(100 * prog);
    $('status').textContent = t('Training · ', '训练中 · ') + done + ' / ' + res.length + t(' runs finished', ' 次运行已完成');
    render();
  }, res => {
    results = res; setRunning(false); $('export').disabled = false; $('progress').value = 100;
    $('status').textContent = t('Finished. Compare the fluctuation (±) and σ² on the cards.', '完成。请比较卡片上的振幅（±）和 σ²。'); render();
  });
  pool.start(runs, { ntrain: 4000, ntest: 2000, evalN: 1000, noiseM: 128 }).catch(err => { setRunning(false); $('status').textContent = String(err.message || err); });
}
$('run').onclick = run;
$('stop').onclick = () => { if (pool) pool.stop(); setRunning(false); $('export').disabled = !results; $('status').textContent = t('Stopped. The next run starts fresh.', '已停止。下次运行将从头开始。'); };
$('export').onclick = () => { if (!results) return; K.download('sgd-noise-lab-results.json', { lab: 'sgd-noise-lab', version: 1, settings: active, results: results.map(r => ({ failed: r.failed, failedAt: r.failedAt, final: r.final, history: r.history })) }); };
$('compare').onchange = () => { buildValueSelectors(); changed(); };
['net', 'arm', 'lr', 'batch', 'steps', 'seed'].forEach(id => $(id).addEventListener('change', changed));
K.initLang(() => { presetButtons(); buildValueSelectorsKeep(); explain(); render(); if (!running && !results) $('status').textContent = t('Ready. Choose a scenario, predict, then run.', '准备就绪。选择一个场景，先预测，再运行。'); });
$('status').textContent = t('Ready. Choose a scenario, predict, then run.', '准备就绪。选择一个场景，先预测，再运行。');
applyPreset('small_lr');
