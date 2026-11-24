"""Train the single-query attention model. Run: python train.py --seed 7."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FRUITS = "apple pear banana orange mango peach plum grape kiwi lemon lime apricot cherry papaya fig guava lychee melon raspberry blueberry".split()
VEGETABLES = "carrot cabbage celery broccoli cauliflower spinach kale lettuce radish beet turnip parsnip onion garlic leek shallot asparagus artichoke okra eggplant cucumber zucchini pumpkin potato yam cassava rutabaga watercress endive chard".split()
WORDS = ["items", "name", "the", "fruit", "vegetable"] + FRUITS + VEGETABLES
ID = {word: i for i, word in enumerate(WORDS)}
QUERY = ["fruit", "vegetable"]


def make_corpus():
    splits = {"train": [], "test": []}
    for i, fruit in enumerate(FRUITS):
        for j, vegetable in enumerate(VEGETABLES):
            split = "test" if (j - 7 * i) % 30 < 6 else "train"
            for first, second in [(fruit, vegetable), (vegetable, fruit)]:
                for query, answer in zip(QUERY, [fruit, vegetable]):
                    tokens = ["items", first, second, "name", "the", query]
                    splits[split].append((tokens, answer))
    return splits


def initialize(rng, dim=16):
    shapes = [(len(WORDS), dim), (dim, dim), (dim, dim), (dim, dim), (len(WORDS), dim)]
    return [rng.normal(0, 0.3 if i == 0 else 1 / math.sqrt(dim), shape)
            for i, shape in enumerate(shapes)]


def forward(params, ids):
    E, W_Q, W_K, W_V, head = params
    x = E[ids]
    # Batch rows store vectors transposed. The math is q = W_Q x_6.
    q = x[:, 5] @ W_Q.T
    k = x[:, :5] @ W_K.T
    v = x[:, :5] @ W_V.T
    scores = (q[:, None] * k).sum(-1)
    z = (scores[:, :, None] * v).sum(1)
    return z @ head.T, (x, q, k, v, scores, z)


def logit_gradient(logits, target):
    rows = np.arange(len(target))
    logits -= logits[rows, target, None]
    shift = np.maximum(0, logits.max(axis=1))
    # dCE/dlogit = p - one_hot(target), written to retain small gradients.
    logits[rows, target] = -np.inf
    odds = np.exp(logits - shift[:, None])
    wrong_mass = odds.sum(1)
    total = np.exp(-shift) + wrong_mass
    dlogits = odds / total[:, None]
    dlogits[rows, target] = -wrong_mass / total
    return dlogits / len(target)


def gradients(params, ids, target):
    E, W_Q, W_K, W_V, head = params
    logits, (x, q, k, v, scores, z) = forward(params, ids)
    dlogits = logit_gradient(logits, target)
    dz = dlogits @ head
    dscore = (dz[:, None] * v).sum(-1)
    dq = (dscore[:, :, None] * k).sum(1)
    dk = dscore[:, :, None] * q[:, None]
    dv = scores[:, :, None] * dz[:, None]
    dx = np.empty_like(x)
    dx[:, :5] = dk @ W_K + dv @ W_V
    dx[:, 5] = dq @ W_Q
    dE = np.zeros_like(E)
    np.add.at(dE, ids, dx)  # A word can appear multiple times in a batch.
    dim = E.shape[1]
    items = x[:, :5].reshape(-1, dim)
    return [dE, dq.T @ x[:, 5], dk.reshape(-1, dim).T @ items,
            dv.reshape(-1, dim).T @ items, dlogits.T @ z]


def arrays(rows):
    return (np.array([[ID[w] for w in words] for words, _ in rows]),
            np.array([ID[answer] for _, answer in rows]))


def measure(params, data):
    x, target = data
    logits, _ = forward(params, x)
    accuracy = np.mean(logits.argmax(1) == target)
    rows = np.arange(len(x))
    logits -= logits[rows, target, None]
    logits[rows, target] = -np.inf
    # Stable even when the correct probability rounds to 1.
    loss = np.logaddexp(0, np.logaddexp.reduce(logits, axis=1))
    mean = float(loss.mean())
    return {"ce": mean, "ppl": math.exp(mean), "pplMinusOne": math.expm1(mean),
            "accuracy": float(accuracy), "worstCE": float(loss.max())}


def snapshot(params, data, step):
    return {"step": step, "train": measure(params, data["train"]),
            "test": measure(params, data["test"]),
            "parameters": [w.tolist() for w in params]}


def train(params, data, batches):
    x, target = data["train"]
    first = [np.zeros_like(w) for w in params]
    second = [np.zeros_like(w) for w in params]
    checkpoints = [snapshot(params, data, 0)]
    for step, batch in enumerate(batches, 1):
        grads = gradients(params, x[batch], target[batch])
        # Adam with bias correction. Only training rows contribute to gradients.
        for w, g, m, u in zip(params, grads, first, second):
            m[:] = 0.9 * m + 0.1 * g
            u[:] = 0.999 * u + 0.001 * g**2
            w -= 0.01 * (m / (1 - 0.9**step)) / (np.sqrt(u / (1 - 0.999**step)) + 1e-8)
        if step in (1, 5, 10, 20, 30, 40, 60, 80) or step % 40 == 0 or step == len(batches):
            checkpoints.append(snapshot(params, data, step))
    return checkpoints


def witness(scale):
    """Explicit solution inside the same 16-dimensional architecture."""
    params = [np.zeros_like(w) for w in initialize(np.random.RandomState(0))]
    E, W_Q, W_K, W_V, head = params
    E[ID["fruit"], 4] = E[ID["vegetable"], 5] = 1
    W_Q[0, 4] = W_Q[1, 5] = 1
    W_K[0, 0] = W_K[1, 1] = 1
    W_V[2, 2] = W_V[3, 3] = 1
    for i, item in enumerate(FRUITS + VEGETABLES):
        row = ID[item]
        theta = 2 * math.pi * i / 50
        E[row, int(i >= 20)] = 1
        E[row, 2:4] = [math.cos(theta), math.sin(theta)]
        head[row, 2:4] = E[row, 2:4] * scale
    return params


def project(checkpoints):
    """Fit one PCA plane per view across all checkpoints, then keep it fixed."""
    raw, projected = [], []
    for checkpoint in checkpoints:
        E, W_Q, W_K, _, _ = map(np.asarray, checkpoint["parameters"])
        raw.append(E[3:])
        projected.append(np.concatenate([E[3:5] @ W_Q.T, E[5:] @ W_K.T]))
    for name, frames in [("embedding", raw), ("queryKey", projected)]:
        points = np.asarray(frames)
        flat = points.reshape(-1, points.shape[-1])
        center = flat.mean(0)
        _, _, directions = np.linalg.svd(flat - center, full_matrices=False)
        for frame, xy in zip(checkpoints, (points - center) @ directions[:2].T):
            frame[name] = xy.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1600)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    (ROOT / "corpus").mkdir(exist_ok=True)
    (ROOT / "site").mkdir(exist_ok=True)
    splits = make_corpus()
    for split, rows in splits.items():
        with (ROOT / "corpus" / (split + ".tsv")).open("w") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(["input", "target"])
            writer.writerows((" ".join(words), answer) for words, answer in rows)
    data = {split: arrays(rows) for split, rows in splits.items()}
    params = initialize(rng)
    batches = rng.randint(len(splits["train"]), size=(args.steps, 128))
    checkpoints = train(params, data, batches)
    project(checkpoints)
    result = {
        "seed": args.seed, "lr": 0.01, "batchSize": 128,
        "words": WORDS, "fruits": FRUITS, "vegetables": VEGETABLES,
        "checkpoints": checkpoints,
        "parameterCount": sum(w.size for w in params),
        "witness": [{"scale": scale, "train": measure(witness(scale), data["train"]),
                     "test": measure(witness(scale), data["test"])}
                    for scale in [100, 1000, 3000]],
    }
    output = ROOT / "site" / ("run-" + str(args.seed) + ".js")
    output.write_text("window.RUNS ??= {};\nwindow.RUNS[" + str(args.seed) + "] = " +
                      json.dumps(result, separators=(",", ":")) + ";\n")
    print(json.dumps({key: checkpoints[-1][key] for key in ("step", "train", "test")}, indent=2))


if __name__ == "__main__":
    main()
