"""Train the two-layer ReLU baseline. Run: python train_mlp.py --seed 7."""
import argparse
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
HIDDEN_WIDTH = 11


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


def initialize(E, rng):
    width_in = 6 * E.shape[1]
    return [E, rng.normal(0, 1 / math.sqrt(width_in), (HIDDEN_WIDTH, width_in)),
            np.zeros(HIDDEN_WIDTH), rng.normal(0, 1 / math.sqrt(HIDDEN_WIDTH),
                                              (len(WORDS), HIDDEN_WIDTH)),
            np.zeros(len(WORDS))]


def forward(params, ids):
    E, W1, b1, W2, b2 = params
    x = E[ids].reshape(len(ids), -1)
    pre_activation = x @ W1.T + b1
    hidden = np.maximum(pre_activation, 0)
    return hidden @ W2.T + b2, (x, pre_activation, hidden)


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
    E, W1, _, W2, _ = params
    logits, (x, pre_activation, hidden) = forward(params, ids)
    dlogits = logit_gradient(logits, target)
    dh = (dlogits @ W2) * (pre_activation > 0)
    dE = np.zeros_like(E)
    np.add.at(dE, ids, (dh @ W1).reshape(len(ids), 6, E.shape[1]))
    return [dE, dh.T @ x, dh.sum(0), dlogits.T @ hidden, dlogits.sum(0)]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1600)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    data = {split: arrays(rows) for split, rows in make_corpus().items()}

    # Share the original experiment's embedding initialization and batch order.
    E = rng.normal(0, 0.3, (len(WORDS), 16))
    # Advance past the other setup draws in the published batch stream.
    rng.normal(size=1648)
    batches = rng.randint(len(data["train"][0]), size=(args.steps, 128))

    params = initialize(E, np.random.RandomState(args.seed + 1))
    checkpoints = train(params, data, batches)
    result = {
        "seed": args.seed, "lr": 0.01, "batchSize": 128,
        "hiddenWidth": HIDDEN_WIDTH, "parameterCount": sum(w.size for w in params),
        "checkpoints": checkpoints,
    }
    output = ROOT / "site" / ("mlp-" + str(args.seed) + ".js")
    output.parent.mkdir(exist_ok=True)
    output.write_text("window.MLP_RUNS ??= {};\nwindow.MLP_RUNS[" + str(args.seed) + "] = " +
                      json.dumps(result, separators=(",", ":")) + ";\n")
    print(json.dumps({key: checkpoints[-1][key] for key in ("step", "train", "test")}, indent=2))


if __name__ == "__main__":
    main()
