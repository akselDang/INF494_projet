"""Figures from saved runs: confusion matrices, per-class recall, learning curves, errors.

python -m src.visualize --dataset mnist --models cnn sage --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .data import load_images
from .graphs import overlay_superpixels, segment
from .summarize import LABELS

CIFAR_CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_run(results, dataset, group, seed):
    base = os.path.join(results, dataset, f"{group}_seed{seed}")
    with open(base + ".json") as f:
        return json.load(f), np.load(base + "_test_preds.npy")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--models", nargs="+", default=["cnn", "sage"],
                   help="run groups to compare, e.g. cnn sage gcn lenet gcn_3feat")
    p.add_argument("--seed", type=int, default=0, help="seed of the runs used for the per-run figures")
    p.add_argument("--results", default="results")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="figures")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    _, _, x_te, y_te = load_images(args.dataset, args.data_dir)
    names = [str(i) for i in range(10)] if args.dataset == "mnist" else CIFAR_CLASSES
    runs = {g: load_run(args.results, args.dataset, g, args.seed) for g in args.models}

    # 1. Confusion matrices
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 6))
    axes = np.atleast_1d(axes)
    for ax, (g, (r, pred)) in zip(axes, runs.items()):
        cm = confusion_matrix(y_te, pred, labels=range(10))
        ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{LABELS.get(g, g)} (test accuracy {r['test_acc']:.2%})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.dataset}_confusion.png"), dpi=150)
    plt.close(fig)

    # 2. Per-class recall: share of the test images of each class that are correctly classified
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(runs)
    lowest = 1.0
    for k, (g, (_, pred)) in enumerate(runs.items()):
        recall = [(pred[y_te == c] == c).mean() for c in range(10)]
        lowest = min(lowest, min(recall))
        ax.bar(np.arange(10) + (k - (len(runs) - 1) / 2) * width, recall, width, label=LABELS.get(g, g))
    ax.set_xticks(range(10), names)
    ax.set_ylabel("Recall (per class)")
    ax.set_ylim(max(0.0, np.floor((lowest - 0.05) * 20) / 20), 1.0)
    ax.legend(ncol=len(runs), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.dataset}_recall_per_class.png"), dpi=150)
    plt.close(fig)

    # 3. Validation accuracy per epoch, mean ± std over all available seeds
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g in args.models:
        curves = []
        for path in sorted(glob.glob(os.path.join(args.results, args.dataset, f"{g}_seed*.json"))):
            with open(path) as f:
                curves.append([h["val_acc"] for h in json.load(f)["history"]])
        n = min(len(c) for c in curves)
        c = np.array([cv[:n] for cv in curves])
        ep = np.arange(1, n + 1)
        ax.plot(ep, c.mean(0), label=LABELS.get(g, g) + (f" ({len(curves)} seeds)" if len(curves) > 1 else ""))
        if len(curves) > 1:
            ax.fill_between(ep, c.mean(0) - c.std(0), c.mean(0) + c.std(0), alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Validation accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.dataset}_learning_curves.png"), dpi=150)
    plt.close(fig)

    # 4. First misclassified test images, with superpixel boundaries for graph models
    for g, (r, pred) in runs.items():
        wrong = np.where(pred != y_te)[0][:8]
        if len(wrong) == 0:
            continue
        fig, axes = plt.subplots(1, len(wrong), figsize=(2.2 * len(wrong), 2.6))
        for ax, i in zip(np.atleast_1d(axes), wrong):
            img = x_te[i]
            show = overlay_superpixels(img, segment(img, args.dataset)) \
                if r["model"] in ("gcn", "sage") else img
            ax.imshow(show, cmap="gray" if show.ndim == 2 else None)
            ax.set_title(f"pred {names[pred[i]]}\ntrue {names[y_te[i]]}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{LABELS.get(g, g)}: first misclassified test images", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{args.dataset}_errors_{g}.png"), dpi=150)
        plt.close(fig)
    print(f"Figures written to {args.out}/")


if __name__ == "__main__":
    main()
