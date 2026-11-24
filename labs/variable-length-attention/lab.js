"use strict";
const data = window.LOOKUP;
const byId = id => document.getElementById(id);
const wordIds = new Map(data.words.map((word, i) => [word, i]));
const fruit = new Set(data.fruits);
const vegetable = new Set(data.vegetables);
const distractor = new Set(data.distractors);
const promptField = byId("prompt");
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const matvec = (matrix, vector) => matrix.map(row => dot(row, vector));
const argmax = xs => xs.reduce((i, x, j) => x > xs[i] ? j : i, 0);
const rate = n => (100 * n).toFixed(1) + "%";

function inspectPrompt(text) {
  const words = text.toLowerCase().match(/[a-z]+/g) || [];
  if (words.length < 7 || words.length > 30) throw Error("Enter a prompt with 7–30 tokens.");
  const [name, the, query] = words.slice(-3);
  if (words[0] !== "items" || name !== "name" || the !== "the" || !["fruit", "vegetable"].includes(query)) {
    throw Error("Use the template “Items: … Name the fruit:” or “Items: … Name the vegetable:”");
  }
  const items = words.slice(1, -3);
  const fruits = items.filter(w => fruit.has(w));
  const vegetables = items.filter(w => vegetable.has(w));
  if (fruits.length !== 1 || vegetables.length !== 1 || items.some(w => !fruit.has(w) && !vegetable.has(w) && !distractor.has(w))) {
    throw Error("Use exactly one listed fruit, one listed vegetable, and listed distractors.");
  }
  return {words, query, target: query === "fruit" ? fruits[0] : vegetables[0], fruits, vegetables};
}

function predictAttention(ids) {
  const [E, Wq, Wk, Wv, head] = data.models.attention.params;
  const q = matvec(Wq, E[ids[ids.length - 1]]);
  const source = ids.slice(0, -1);
  const scores = source.map(id => dot(q, matvec(Wk, E[id])));
  const z = Array(q.length).fill(0);
  source.forEach((id, j) => {
    matvec(Wv, E[id]).forEach((v, k) => { z[k] += scores[j] * v; });
  });
  return {logits: matvec(head, z), scores};
}

function predictMlp(ids) {
  const [E, W1, b1, W2, b2] = data.models.mlp.params;
  const x = [...ids, ...Array(30 - ids.length).fill(0)].flatMap(id => E[id]);
  const h = matvec(W1, x).map((v, i) => Math.max(0, v + b1[i]));
  return matvec(W2, h).map((v, i) => v + b2[i]);
}

function summary(logits, target) {
  const top = argmax(logits);
  const scores = logits.map(x => Math.exp(x - logits[top]));
  const probability = scores[wordIds.get(target)] / scores.reduce((a, b) => a + b, 0);
  return {top: data.words[top], probability};
}

function element(tag, value, cls) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (cls) node.className = cls;
  return node;
}
function showCard(name, cls, logits, target) {
  const result = summary(logits, target);
  const card = element("article", "", "card " + cls);
  const guess = result.top === "<pad>" ? "padding token" : result.top;
  card.append(element("h3", name), element("strong", guess),
    element("p", "P(" + target + ") = " + rate(result.probability)));
  return card;
}
function printPrompt(words) {
  return "Items: " + words.slice(1, -3).join(", ") + ". Name the " + words[words.length - 1] + ":";
}
function updateCount() {
  byId("token-count").textContent = "· " + ((promptField.value.match(/[a-z]+/gi) || []).length) + " tokens";
}
function run() {
  updateCount();
  const cards = byId("predictions");
  const trace = byId("trace");
  cards.replaceChildren(); trace.replaceChildren();
  try {
    const inp = inspectPrompt(promptField.value);
    const ids = inp.words.map(w => wordIds.get(w));
    const raw = predictAttention(ids);
    cards.append(showCard("Attention", "", raw.logits, inp.target),
      showCard("Padded MLP", "mlp-card", predictMlp(ids), inp.target));
    byId("status").textContent = "";
    const largest = Math.max(1e-12, ...raw.scores.map(Math.abs));
    raw.scores.forEach((score, j) => {
      const box = element("div", "", "score" + (inp.words[j] === inp.target ? " target" : ""));
      const bar = element("span", "", "bar" + (score < 0 ? " neg" : ""));
      bar.style.width = (100 * Math.abs(score) / largest) + "%";
      box.append(element("strong", inp.words[j]), element("span", score.toFixed(3)), bar);
      trace.append(box);
    });
  } catch (error) {
    byId("status").textContent = error.message;
  }
}
function newExample() {
  const requested = Number(byId("length").value);
  const n = Number.isFinite(requested) ? Math.min(30, Math.max(7, Math.trunc(requested))) : 7;
  byId("length").value = n;
  const i = Math.floor(Math.random() * data.fruits.length);
  const j = (7 * i + Math.floor(Math.random() * 6)) % data.vegetables.length;
  const shuffle = a => {
    for (let k = a.length - 1; k > 0; k--) {
      const swap = Math.floor(Math.random() * (k + 1));
      [a[k], a[swap]] = [a[swap], a[k]];
    }
    return a;
  };
  const words = shuffle([data.fruits[i], data.vegetables[j],
    ...shuffle([...data.distractors]).slice(0, n - 6)]);
  promptField.value = printPrompt(["items", ...words, "name", "the", byId("query").value]);
  run();
}

function plot(canvas, series, xMin, xMax, yMin, yMax, xTicks, yTicks, labelY, labelX) {
  const g = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  const L = 62, T = 15, R = W - 15, B = H - 35;
  const sx = x => L + (R - L) * (x - xMin) / (xMax - xMin);
  const sy = y => B - (B - T) * (y - yMin) / (yMax - yMin);
  g.clearRect(0, 0, W, H);
  g.font = "15px sans-serif";
  for (const y of yTicks) {
    g.beginPath(); g.strokeStyle = "#dce6e3"; g.moveTo(L, sy(y)); g.lineTo(R, sy(y)); g.stroke();
    g.fillStyle = "#465a5e"; g.textAlign = "right"; g.fillText(labelY(y), L - 8, sy(y) + 5);
  }
  for (const x of xTicks) {
    g.fillStyle = "#465a5e"; g.textAlign = "center"; g.fillText(labelX(x), sx(x), B + 25);
  }
  series.forEach(({points, color}) => {
    g.strokeStyle = color;g.lineWidth = 3;g.beginPath();
    points.forEach(([x, y], i) => i ? g.lineTo(sx(x), sy(y)) : g.moveTo(sx(x), sy(y)));
    g.stroke();
    points.forEach(([x, y]) => {g.beginPath();g.fillStyle = color;g.arc(sx(x), sy(y), 3.5, 0, Math.PI * 2);g.fill();});
  });
}
const frames = name => data.models[name].frames;
const last = name => frames(name)[frames(name).length - 1];
const seriesFor = get => ["attention", "mlp"].map((name, i) => ({
  color: i ? "#a34c27" : "#146354", points: get(name)
}));
function charts() {
  const ce = seriesFor(name => frames(name).map(row => [row.step, Math.log10(Math.max(row.ce, 1e-12))]));
  const lo = Math.floor(Math.min(...ce.flatMap(s => s.points.map(pt => pt[1]))));
  const hi = Math.ceil(Math.max(...ce.flatMap(s => s.points.map(pt => pt[1]))));
  const stride = Math.max(1, Math.ceil((hi - lo) / 7));
  const marks = [];
  for (let j = lo; j <= hi; j += stride) marks.push(j);
  plot(byId("ce"), ce, 0, data.steps, lo, hi, [0, data.steps / 2, data.steps], marks,
    y => (10 ** y >= 1 ? String(10 ** y) : (10 ** y).toExponential(0)), String);
  const accuracy = seriesFor(name => Object.entries(last(name).by_length).map(([len, [right, all]]) => [+len, right / all]));
  plot(byId("acc"), accuracy, 7, 30, 0, 1, [7, 10, 15, 20, 25, 30],
    [0, .25, .5, .75, 1], y => Math.round(y * 100) + "%", String);
  byId("run-details").textContent = "Seed " + data.seed + " · " + data.train.toLocaleString() + " training prompts · " +
    data.test.toLocaleString() + " held-out prompts · Attention " + last("attention").correct.toLocaleString() +
    " / " + data.test.toLocaleString() + " · MLP " + last("mlp").correct.toLocaleString() + " / " + data.test.toLocaleString();
}

byId("prompt-form").addEventListener("submit", event => { event.preventDefault(); run(); });
promptField.addEventListener("input", updateCount);
byId("new-example").addEventListener("click", newExample);
byId("query").addEventListener("change", () => {
  promptField.value = promptField.value.replace(/(name the )\w+(:?)/i, "$1" + byId("query").value + ":");
  run();
});
byId("swap").addEventListener("click", () => {
  try {
    const {words, fruits, vegetables} = inspectPrompt(promptField.value);
    promptField.value = printPrompt(words.map(w => w === fruits[0] ? vegetables[0] : w === vegetables[0] ? fruits[0] : w));
    run();
  } catch (error) {
    byId("status").textContent = error.message;
  }
});
for (const [title, words] of [["Fruit", data.fruits], ["Vegetables", data.vegetables], ["Distractors", data.distractors]]) {
  byId("vocab").append(element("p", title + ": " + words.join(", ")));
}
run(); charts();
