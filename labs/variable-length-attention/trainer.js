"use strict";

const D = 16;
const MAX = 30;
const H = 11;
const BATCH = 32;
let state;
let task = null;
let goal = 0;
let randomState = 7;

function random() {
  randomState = (Math.imul(1664525, randomState) + 1013904223) >>> 0;
  return randomState / 4294967296;
}

function gaussian(scale) {
  return scale * Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

function matrix(rows, cols, scale = 0) {
  return Array.from({length: rows}, () =>
    Float64Array.from({length: cols}, () => scale ? gaussian(scale) : 0));
}

function moments(params) {
  return params.map(a => ArrayBuffer.isView(a) ? new Float64Array(a.length) : matrix(a.length, a[0].length));
}

function makeModel(params) {
  return {params, first: moments(params), second: moments(params)};
}

function initialize(vocab) {
  const E = matrix(vocab, D, 0.3);
  E[0].fill(0);
  const attention = [
    E, matrix(D, D, 1 / 4), matrix(D, D, 1 / 4),
    matrix(D, D, 1 / 4), matrix(vocab, D, 1 / 4)
  ];
  const mlp = [
    E.map(row => row.slice()), matrix(H, MAX * D, 1 / Math.sqrt(MAX * D)),
    new Float64Array(H), matrix(vocab, H, 1 / Math.sqrt(H)), new Float64Array(vocab)
  ];
  return {attention: makeModel(attention), mlp: makeModel(mlp)};
}

function dot(left, right) {
  let total = 0;
  for (let j = 0; j < left.length; j++) total += left[j] * right[j];
  return total;
}
function mv(weights, input) {
  return Float64Array.from(weights, row => dot(input, row));
}
function back(weights, grad) {
  const outputs = new Float64Array(weights[0].length);
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < outputs.length; j++) outputs[j] += weights[i][j] * grad[i];
  }
  return outputs;
}
function outerAdd(out, left, right) {
  for (let i = 0; i < left.length; i++) {
    for (let j = 0; j < right.length; j++) out[i][j] += left[i] * right[j];
  }
}
function add(out, src) {
  for (let i = 0; i < out.length; i++) out[i] += src[i];
}
function targetGradient(logits, target) {
  const peak = Math.max(...logits);
  const exps = Float64Array.from(logits, x => Math.exp(x - peak));
  const sum = exps.reduce((a, b) => a + b, 0);
  const loss = Math.log(sum) + peak - logits[target];
  for (let i = 0; i < exps.length; i++) exps[i] /= BATCH * sum;
  exps[target] -= 1 / BATCH;
  return [loss, exps];
}

function attentionGradient(params, grad, ids, target) {
  const [E, Wq, Wk, Wv, head] = params;
  const [dE, dWq, dWk, dWv, dHead] = grad;
  const last = ids[ids.length - 1], xq = E[last], q = mv(Wq, xq);
  const earlier = ids.slice(0, -1);
  const keys = earlier.map(id => mv(Wk, E[id]));
  const values = earlier.map(id => mv(Wv, E[id]));
  const scores = Float64Array.from(keys, k => dot(q, k));
  const z = new Float64Array(D);
  for (let i = 0; i < earlier.length; i++) {
    for (let j = 0; j < D; j++) z[j] += scores[i] * values[i][j];
  }
  const [loss, dy] = targetGradient(mv(head, z), target);
  outerAdd(dHead, dy, z);
  const dz = back(head, dy), dq = new Float64Array(D);
  for (let i = 0; i < earlier.length; i++) {
    const ds = dot(dz, values[i]);
    const dk = Float64Array.from(q, x => ds * x);
    const dv = Float64Array.from(dz, x => scores[i] * x);
    for (let j = 0; j < D; j++) dq[j] += ds * keys[i][j];
    const index = earlier[i];
    outerAdd(dWk, dk, E[index]);
    outerAdd(dWv, dv, E[index]);
    add(dE[index], back(Wk, dk));
    add(dE[index], back(Wv, dv));
  }
  outerAdd(dWq, dq, xq);
  add(dE[last], back(Wq, dq));
  return loss;
}

function mlpGradient(params, grad, ids, target) {
  const [E, W1, b1, W2, b2] = params;
  const [dE, dW1, db1, dW2, db2] = grad;
  const actualWidth = ids.length * D;
  const x = new Float64Array(actualWidth);
  ids.forEach((id, i) => x.set(E[id], i * D));
  // The omitted coordinates up to 30 * D are exactly zero.
  const h = Float64Array.from(W1, (row, i) => Math.max(0, dot(x, row) + b1[i]));
  const logits = mv(W2, h);
  add(logits, b2);
  const [loss, dy] = targetGradient(logits, target);
  outerAdd(dW2, dy, h);
  add(db2, dy);
  const dh = back(W2, dy);
  for (let i = 0; i < H; i++) if (h[i] === 0) dh[i] = 0;
  outerAdd(dW1, dh, x);
  add(db1, dh);
  for (let i = 0; i < ids.length; i++) {
    for (let j = 0; j < D; j++) {
      const col = i * D + j;
      let value = 0;
      for (let k = 0; k < H; k++) value += W1[k][col] * dh[k];
      dE[ids[i]][j] += value;
    }
  }
  return loss;
}

function update(model, grads, step) {
  const firstCorrection = 1 - 0.9 ** step;
  const secondCorrection = 1 - 0.999 ** step;
  function apply(weights, g, m, v) {
    for (let j = 0; j < weights.length; j++) {
      m[j] = 0.9 * m[j] + 0.1 * g[j];
      v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
      weights[j] -= 0.01 * (m[j] / firstCorrection) / (Math.sqrt(v[j] / secondCorrection) + 1e-8);
    }
  }
  model.params.forEach((p, i) => {
    if (ArrayBuffer.isView(p)) apply(p, grads[i], model.first[i], model.second[i]);
    else p.forEach((r, j) => apply(r, grads[i][j], model.first[i][j], model.second[i][j]));
  });
}

function shuffle(items) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

function sample() {
  const [fruit, veg] = state.pairs[Math.floor(random() * state.pairs.length)];
  const length = 7 + Math.floor(random() * 24);
  const items = shuffle([fruit, veg, ...shuffle(state.distractors.slice()).slice(0, length - 6)]);
  const askFruit = random() < 0.5;
  const q = askFruit ? state.ids.fruit : state.ids.vegetable;
  return [[state.ids.items, ...items, state.ids.name, state.ids.the, q], askFruit ? fruit : veg];
}

function trainOne() {
  const {attention, mlp} = state.models;
  const dA = moments(attention.params), dM = moments(mlp.params);
  let lossA = 0, lossM = 0;
  for (let i = 0; i < BATCH; i++) {
    const [ids, target] = sample();
    lossA += attentionGradient(attention.params, dA, ids, target);
    lossM += mlpGradient(mlp.params, dM, ids, target);
  }
  state.step++;
  update(attention, dA, state.step);
  update(mlp, dM, state.step);
  return [state.step, lossA / BATCH, lossM / BATCH];
}

function report(status, losses) {
  postMessage({
    kind: "state", status, losses, step: state.step,
    attention: state.models.attention.params, mlp: state.models.mlp.params
  });
}

function tick() {
  task = null;
  const losses = [];
  while (state.step < goal && losses.length < 5) losses.push(trainOne());
  report(state.step < goal ? "training" : "ready", losses);
  if (state.step < goal) task = setTimeout(tick, 0);
}

onmessage = ({data}) => {
  if (data.kind === "reset") {
    clearTimeout(task); task = null;
    randomState = 7;
    const ids = Object.fromEntries(data.words.map((word, i) => [word, i]));
    const pairs = [];
    data.fruits.forEach((fruit, i) => data.vegetables.forEach((veg, j) => {
      if (((j - 7 * i) % 30 + 30) % 30 >= 6) pairs.push([ids[fruit], ids[veg]]);
    }));
    state = {ids, pairs, distractors: data.distractors.map(x => ids[x]), models: initialize(data.words.length), step: 0};
    report("ready", []);
  } else if (data.kind === "train" && state && task === null) {
    goal = state.step + data.steps;
    task = setTimeout(tick, 0);
  } else if (data.kind === "pause" && state) {
    clearTimeout(task); task = null;
    report("paused", []);
  }
};
