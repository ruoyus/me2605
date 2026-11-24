"""Train the variable-length lookup lab. Only NumPy is required."""
import argparse
import json
from pathlib import Path
import numpy as np

FRUITS = "apple pear banana orange mango peach plum grape kiwi lemon lime apricot cherry papaya fig guava lychee melon raspberry blueberry".split()
VEGETABLES = "carrot cabbage celery broccoli cauliflower spinach kale lettuce radish beet turnip parsnip onion garlic leek shallot asparagus artichoke okra eggplant cucumber zucchini pumpkin potato yam cassava rutabaga watercress endive chard".split()
DISTRACTORS = "tree cat dog river stone chair book cloud train lamp bridge coat shoe clock drum road lake bird pencil paper ship window door mountain".split()
WORDS = ["<pad>", "items", "name", "the", "fruit", "vegetable"] + FRUITS + VEGETABLES + DISTRACTORS
ID = {word: i for i, word in enumerate(WORDS)}
MIN, MAX, D, H = 7, 30, 16, 11

def make_data(seed):
    """Hold out ingredient pairs, with both queries at every tested length."""
    rng = np.random.RandomState(seed)
    data = {"train": [], "test": []}
    for i, fruit in enumerate(FRUITS):
        for j, vegetable in enumerate(VEGETABLES):
            split = "test" if (j - 7 * i) % 30 < 6 else "train"
            for length in range(MIN, MAX + 1):
                items = [fruit, vegetable] + list(rng.choice(DISTRACTORS, length - 6, replace=False))
                rng.shuffle(items)
                for query, target in [("fruit", fruit), ("vegetable", vegetable)]:
                    prompt = ["items"] + items + ["name", "the", query]
                    data[split].append(([ID[w] for w in prompt] + [0] * (MAX - length), ID[target]))
    return {split: tuple(map(np.asarray, zip(*rows))) for split, rows in data.items()}

def attention(params, ids):
    E, Wq, Wk, Wv, head = params
    batch = np.arange(len(ids))
    last = (ids != 0).sum(1) - 1
    x = E[ids]
    query_x = x[batch, last]
    valid = (ids != 0) & (np.arange(ids.shape[1]) < last[:, None])
    q, k, v = query_x @ Wq.T, x @ Wk.T, x @ Wv.T
    scores = (q[:, None, :] * k).sum(-1) * valid
    z = (scores[:, :, None] * v).sum(1)
    return z @ head.T, (x, q, k, v, scores, z, batch, last, valid, query_x)

def mlp(params, ids):
    E, W1, b1, W2, b2 = params
    x = E[ids].reshape(len(ids), -1)
    pre = x @ W1.T + b1
    hidden = np.maximum(pre, 0)
    return hidden @ W2.T + b2, (x, pre, hidden)

def loss_grad(logits, target):
    rows = np.arange(len(target))
    exp_scores = np.exp(logits - logits.max(1, keepdims=True))
    probabilities = exp_scores / exp_scores.sum(1, keepdims=True)
    probabilities[rows, target] -= 1
    return probabilities / len(target)

def gradients(params, ids, target, model):
    logits, cache = model(params, ids)
    dy = loss_grad(logits, target)
    dE = np.zeros_like(params[0])
    if model is attention:
        _, Wq, Wk, Wv, head = params
        x, q, k, v, scores, z, batch, last, valid, query_x = cache
        dz = dy @ head
        ds = (dz[:, None, :] * v).sum(-1) * valid
        dq = (ds[:, :, None] * k).sum(1)
        dk = ds[:, :, None] * q[:, None, :]
        dv = scores[:, :, None] * dz[:, None, :]
        dx = dk @ Wk + dv @ Wv
        dx[batch, last] += dq @ Wq
        np.add.at(dE, ids, dx)
        result = [dE, dq.T @ query_x, dk.reshape(-1,D).T @ x.reshape(-1,D),
                  dv.reshape(-1,D).T @ x.reshape(-1,D), dy.T @ z]
    else:
        _, W1, _, W2, _ = params
        x, pre, hidden = cache
        dh = (dy @ W2) * (pre > 0)
        np.add.at(dE, ids, (dh @ W1).reshape(len(ids), MAX, D))
        result = [dE, dh.T @ x, dh.sum(0), dy.T @ hidden, dy.sum(0)]
    dE[0] = 0
    return result

def measure(params, data, model):
    ids, targets = data
    total_loss = 0.0
    correct = np.zeros(MAX + 1, dtype=int)
    count = np.zeros_like(correct)
    for start in range(0, len(ids), 512):
        x, y = ids[start:start+512], targets[start:start+512]
        scores, _ = model(params, x)
        total_loss += (np.logaddexp.reduce(scores, axis=1) - scores[np.arange(len(y)), y]).sum()
        lengths = np.count_nonzero(x, axis=1)
        np.add.at(count, lengths, 1)
        np.add.at(correct, lengths, scores.argmax(1) == y)
    return dict(ce=float(total_loss / len(ids)), correct=int(correct.sum()), total=len(ids),
                by_length={int(j): [int(correct[j]), int(count[j])] for j in np.flatnonzero(count)})

def fit(params, data, model, batches):
    train_ids, targets = data["train"]
    m = [np.zeros_like(w) for w in params]
    v = [np.zeros_like(w) for w in params]
    frames = []
    for step in range(len(batches) + 1):
        if step:
            batch = batches[step-1]
            grads = gradients(params, train_ids[batch], targets[batch], model)
            for w, g, a, b in zip(params, grads, m, v):
                a[:] = .9 * a + .1 * g
                b[:] = .999 * b + .001 * g**2
                w -= .01 * (a / (1 - .9**step)) / (np.sqrt(b / (1 - .999**step)) + 1e-8)
        if step in (0, 20, 60, 100) or step % 200 == 0 or step == len(batches):
            frames.append(dict(step=step, **measure(params, data["test"], model)))
    return dict(frames=frames, params=[w.tolist() for w in params], parameters=sum(w.size for w in params)-D)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1600)
    args = parser.parse_args()
    data = make_data(args.seed)
    rng = np.random.RandomState(args.seed)
    norm = lambda fan, shape: rng.normal(0, fan**-0.5, shape)
    emb = rng.normal(0, .3, (len(WORDS), D))
    emb[0] = 0
    a = [emb.copy()] + [norm(D, (D,D)) for _ in range(3)] + [norm(D, (len(WORDS),D))]
    m = [emb.copy(), norm(MAX*D,(H,MAX*D)), np.zeros(H), norm(H,(len(WORDS),H)), np.zeros(len(WORDS))]
    batches = rng.randint(len(data["train"][0]), size=(args.steps,128))
    results = {name: fit(params, data, model, batches)
               for name, params, model in [("attention",a,attention),("mlp",m,mlp)]}
    records = dict(seed=args.seed, steps=args.steps, train=len(data["train"][0]),
                   test=len(data["test"][0]), words=WORDS, fruits=FRUITS,
                   vegetables=VEGETABLES, distractors=DISTRACTORS, models=results)
    path = Path(__file__).with_name("results.js")
    path.write_text("window.LOOKUP = " + json.dumps(records, separators=(",",":")) + ";\n")
    for name, result in results.items():
        frame = result["frames"][-1]
        print(name, result["parameters"], frame["correct"], "/", frame["total"], "CE", frame["ce"])

if __name__ == "__main__":
    main()
