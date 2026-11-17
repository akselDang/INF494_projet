"""Aggregate all runs in results/ into one table (mean ± standard deviation over seeds).

python -m src.summarize            # prints the table and writes results/summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ORDER = ["gcn_3feat", "gcn", "sage_raw", "sage", "lenet", "cnn"]
LABELS = {"gcn_3feat": "GCN (3 features)", "gcn": "GCN", "sage": "GraphSAGE",
          "sage_raw": "GraphSAGE (raw features)",
          "lenet": "LeNet-5 (CNN)", "cnn": "CNN"}


def fmt(values, pct=True):
    v = np.array(values, dtype=float) * (100 if pct else 1)
    if len(v) == 1:
        return f"{v[0]:.2f}" if pct else f"{v[0]:.0f}"
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if pct else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args(argv)

    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.results, "*", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        runs[(r["dataset"], r["group"])].append(r)
    if not runs:
        print("No results found. Run python -m src.train first.")
        return

    lines = []
    for dataset in sorted({d for d, _ in runs}):
        groups = sorted((g for d, g in runs if d == dataset),
                        key=lambda g: ORDER.index(g) if g in ORDER else len(ORDER))
        lines += [f"### {dataset}", "",
                  "| Model | Input | Parameters | Seeds | Val. accuracy (%) | Test accuracy (%) "
                  "| Best epoch | Train acc., eval mode (%) | Seconds / epoch |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for g in groups:
            rs = runs[(dataset, g)]
            r0 = rs[0]
            inp = "pixels" if r0["model"] in ("cnn", "lenet") else f"graph, {r0['features']} features"
            if r0["config"].get("raw_features") and r0["model"] in ("gcn", "sage"):
                inp += ", not standardised"
            sec = [np.mean([h["seconds"] for h in r["history"]]) for r in rs]
            lines.append(
                f"| {LABELS.get(g, g)} | {inp} | {r0['n_params']:,} | {len(rs)} "
                f"| {fmt([r['val_acc'] for r in rs])} | {fmt([r['test_acc'] for r in rs])} "
                f"| {fmt([r['best_epoch'] for r in rs], pct=False)} "
                f"| {fmt([r['train_acc_eval'] for r in rs])} | {np.mean(sec):.0f} |")
        lines.append("")
    lines.append("Mean ± standard deviation over seeds. The best epoch is chosen on the validation set; "
                 "the test set is evaluated once per run.")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
