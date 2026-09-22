#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalization vs. the weight-constraining alternatives -- the Lecture 3 lab.

The lecture motivates normalization with the condition-number story and then
lists four ways of forcing a norm requirement on the *parameters* (projection,
regularizer, constraint, reparameterization), arguing that each of them costs
something and none of them is what people actually use.  This script puts the
claim under a measurement: train the same deep MLP, on the same data, with the
same optimizer, and change only the mechanism.

      none   -- He init, nothing else
      ln     -- LayerNorm before the activation
      bn     -- BatchNorm before the activation
      rms    -- RMSNorm before the activation
      proj   -- projection: after every step, rescale each weight row to ||w||=sqrt(2)
      reg    -- regularizer: add lam * sum (||w_row|| - sqrt(2))^2 to the loss
      wn     -- reparameterization: W = sqrt(2) * v/||v|| (row-wise WeightNorm)

The target sqrt(2) is not arbitrary: it is the row norm He initialization already
gives, so the four weight-side methods start from a point that already satisfies
their constraint, and are not handicapped by their own initialization.

Data: the course package's data/mnist.npz, read-only.  Nothing is downloaded.

Usage:
    python experiment.py --quick                      # screening run
    python experiment.py --layers 16 --width 128 --steps 3000 \
        --lr 0.1 --methods none ln bn rms proj reg wn --out results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = ("/Users/ruoyusun/Desktop/重要研究项目_2022起/0_临时_Teaching 教学/"
       "2026_Fall_ME2605_DDA6204_LLM_training_当前课程/Week 1/Lab Session/"
       "ME2605_Week1_Lab_Package/data/mnist.npz")
ROW_TARGET = math.sqrt(2.0)          # He row norm for ReLU
EPS = 1e-5


def load_mnist(path, n_train, n_val):
    with np.load(path) as z:
        xtr, ytr = z["x_train"], z["y_train"]
        xte, yte = z["x_test"], z["y_test"]
    xtr = xtr.reshape(len(xtr), -1).astype(np.float32) / 255.0
    xte = xte.reshape(len(xte), -1).astype(np.float32) / 255.0
    if n_train:
        xtr, ytr = xtr[:n_train], ytr[:n_train]
    if n_val:
        xte, yte = xte[:n_val], yte[:n_val]
    return (torch.from_numpy(xtr), torch.from_numpy(ytr).long(),
            torch.from_numpy(xte), torch.from_numpy(yte).long())


class Norm(nn.Module):
    """The normalization operation of the lecture, on one axis."""

    def __init__(self, mode, dim, rows=0):
        super().__init__()
        self.mode = mode
        self.g = nn.Parameter(torch.ones(rows or dim))
        self.b = nn.Parameter(torch.zeros(rows or dim)) if mode != "rms" else None
        self.running = torch.zeros(rows or dim)
        self.running_var = torch.ones(rows or dim)

    def forward(self, h):
        if self.mode == "ln":
            mu = h.mean(-1, keepdim=True)
            var = h.var(-1, unbiased=False, keepdim=True)
        elif self.mode == "rms":
            mu = 0.0
            var = (h * h).mean(-1, keepdim=True)
        else:                                            # bn
            if self.training:
                mu = h.mean(0)
                var = h.var(0, unbiased=False)
                with torch.no_grad():
                    self.running = 0.9 * self.running + 0.1 * mu.detach()
                    self.running_var = 0.9 * self.running_var + 0.1 * var.detach()
            else:
                mu, var = self.running, self.running_var
        out = (h - mu) / torch.sqrt(var + EPS) * self.g
        if self.b is not None:
            out = out + self.b
        return out


class MLP(nn.Module):
    def __init__(self, mode, layers, width, din=784, dout=10):
        super().__init__()
        self.mode = mode
        self.layers = layers
        self.L = nn.ModuleList()
        sizes = [din] + [width] * layers + [dout]
        for i in range(layers + 1):
            lin = nn.Linear(sizes[i], sizes[i + 1], bias=False)
            nn.init.normal_(lin.weight, 0.0, math.sqrt(2.0 / sizes[i]))
            self.L.append(lin)
        self.norms = nn.ModuleList()
        if mode == "bn":
            self.norms = nn.ModuleList([Norm("bn", width, rows=width)
                                        for _ in range(layers)])
        elif mode in ("ln", "rms"):
            self.norms = nn.ModuleList([Norm(mode, width) for _ in range(layers)])

        if mode == "wn":                     # row-wise reparameterization
            self.V, self.alpha = nn.ParameterList(), nn.ParameterList()
            for lin in self.L:
                w = lin.weight.data
                self.V.append(nn.Parameter(w.clone()))
                self.alpha.append(nn.Parameter(torch.full((w.shape[0], 1), ROW_TARGET)))
            for lin in self.L:
                del lin.weight                      # not used any more
        if mode == "reg":
            self.lam = 0.1

    # -- the four weight-side mechanisms -----------------------------------
    def effective_weight(self, i):
        V = self.V[i]
        alpha = self.alpha[i]
        return alpha * V / (V.norm(dim=1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def project(self):
        for lin in self.L:
            if not hasattr(lin, "weight"):
                continue
            n = lin.weight.norm(dim=1, keepdim=True)
            lin.weight.mul_(ROW_TARGET / (n + 1e-8))

    def regularizer(self):
        if self.mode != "reg":
            return 0.0
        s = 0.0
        for lin in self.L:
            n = lin.weight.norm(dim=1)
            s = s + ((n - ROW_TARGET) ** 2).mean()
        return self.lam * s

    def forward(self, x, stats=None):
        z = x
        for i in range(self.layers):
            W = self.effective_weight(i) if self.mode == "wn" else self.L[i].weight
            h = z @ W.t()
            if self.mode in ("ln", "bn", "rms"):
                h = self.norms[i](h)
            if stats is not None:
                stats["fwd"].append(float(h.detach().pow(2).mean(-1).mean().sqrt()))
            z = F.relu(h)
        h = z @ (self.effective_weight(self.layers) if self.mode == "wn"
                 else self.L[self.layers].weight).t()
        return h


def run(mode, args, data, seed=0):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = data
    net = MLP(mode, args.layers, args.width)
    params = [p for p in net.parameters() if p.requires_grad and p.numel()]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                          nesterov=args.momentum > 0)

    n = len(xtr)
    curve, diverge = [], False
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,))
        loss = F.cross_entropy(net(xtr[idx]), ytr[idx]) + net.regularizer()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if mode == "proj":
            net.project()
        if not torch.isfinite(loss):
            diverge = True
            break
        if step % args.every == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                acc = (net(xte).argmax(1) == yte).float().mean().item()
            net.train()
            curve.append({"step": step, "val_acc": round(acc, 4)})
    # end-state diagnostics: forward scale at each layer and one backward pass
    net.eval()
    stats = {"fwd": []}
    with torch.no_grad():
        net(xtr[:64], stats=stats)
    out = {"method": mode, "lr": args.lr, "momentum": args.momentum,
           "layers": args.layers,
           "width": args.width, "steps": args.steps, "seed": seed,
           "diverged": diverge, "curve": curve,
           "best_val_acc": max([c["val_acc"] for c in curve], default=None),
           "final_val_acc": curve[-1]["val_acc"] if curve else None,
           "fwd_scale_last": round(stats["fwd"][-1], 4) if stats["fwd"] else None,
           "fwd_scale_first": round(stats["fwd"][0], 4) if stats["fwd"] else None,
           "seconds": round(time.time() - t0, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=PKG)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--feature-scale", type=float, default=0.0,
                    help="0 = leave the input alone; otherwise multiply feature j by a "
                         "log-uniform factor spanning 10^+-F (the ill-conditioning of the "
                         "motivation)")
    ap.add_argument("--standardize-input", action="store_true",
                    help="the data-preprocessing answer: divide each feature by its train std")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+",
                    default=["none", "ln", "bn", "rms", "proj", "reg", "wn"])
    ap.add_argument("--quick", action="store_true",
                    help="short screening run: L=16 d=128 600 steps on 8k samples")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    if args.quick:
        args.steps, args.train, args.val, args.every = 600, 8000, 2000, 50

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    data = load_mnist(args.data, args.train, args.val)
    if args.feature_scale:
        g = torch.Generator().manual_seed(1234)
        sc = 10 ** (torch.rand(784, generator=g) * 2 * args.feature_scale - args.feature_scale)
        (data[0] * sc, data[2] * sc)
        data = (data[0] * sc, data[1], data[2] * sc, data[3])
        print("feature scales: 10^%.1f .. 10^%.1f" %
              (float(sc.log10().min()), float(sc.log10().max())))
    if args.standardize_input:
        mu, sd = data[0].mean(0), data[0].std(0) + 1e-8
        data = ((data[0] - mu) / sd, data[1], (data[2] - mu) / sd, data[3])
        print("input standardized (the preprocessing baseline)")
    print("data: %d train / %d val    model: %d x %d    lr %g  steps %d"
          % (len(data[0]), len(data[2]), args.layers, args.width, args.lr, args.steps))
    print("%-6s %10s %10s %8s %9s %9s" %
          ("method", "best", "final", "diverged", "fwd[0]", "fwd[L]"))
    rows = []
    for m in args.methods:
        r = run(m, args, data, args.seed)
        rows.append(r)
        print("%-6s %10s %10s %8s %9s %9s   (%.0fs)"
              % (m, r["best_val_acc"], r["final_val_acc"], r["diverged"],
                 r["fwd_scale_first"], r["fwd_scale_last"], r["seconds"]))
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "runs": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
