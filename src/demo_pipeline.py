"""Show the image -> graph pipeline on one image, and print graph statistics.

python -m src.demo_pipeline --index 0             # figure for one MNIST training image
python -m src.demo_pipeline --stats               # also average nodes / edges / degree over cached graphs
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import load_images
from .graphs import graph_stats, image_to_arrays, overlay_superpixels, segment


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--index", type=int, default=0, help="index of the training image")
    p.add_argument("--stats", action="store_true", help="statistics over the cached graphs (data/graphs)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="figures")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    x_tr, y_tr, _, _ = load_images(args.dataset, args.data_dir)
    img, label = x_tr[args.index], int(y_tr[args.index])
    seg = segment(img, args.dataset)
    feats, edges = image_to_arrays(img, args.dataset)
    H, W = seg.shape
    n_nodes, n_edges = feats.shape[0], edges.shape[1] // 2
    print(f"label {label}: {n_nodes} nodes, {n_edges} undirected edges, "
          f"mean degree {2 * n_edges / n_nodes:.2f}, {feats.shape[1]} features per node")

    cmap = "gray" if img.ndim == 2 else None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img, cmap=cmap)
    axes[0].set_title(f"Image (label {label})")
    axes[1].imshow(overlay_superpixels(img, seg))
    axes[1].set_title(f"SLIC superpixels ({n_nodes})")
    axes[2].imshow(img, cmap=cmap, alpha=0.35)
    xs, ys = feats[:, -4] * W, feats[:, -3] * H  # x and y are the 4th and 3rd features from the end
    for a, b in edges.T[: edges.shape[1] // 2]:
        axes[2].plot([xs[a], xs[b]], [ys[a], ys[b]], color="tab:blue", lw=0.8)
    axes[2].scatter(xs, ys, s=12, c="tab:red", zorder=3)
    axes[2].set_title(f"Graph ({n_nodes} nodes, {n_edges} edges)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(args.out, f"{args.dataset}_pipeline_demo.png")
    fig.savefig(path, dpi=150)
    print(f"figure written to {path}")

    if args.stats:
        for split in ("train", "test"):
            cache = os.path.join(args.data_dir, "graphs", f"{args.dataset}_{split}_graphs.pt")
            if not os.path.exists(cache):
                print(f"{cache} not found: run a graph model once to build it")
                continue
            s = graph_stats(torch.load(cache, weights_only=False))
            print(f"{split}: nodes {s['nodes_mean']:.1f} (min {s['nodes_min']:.0f}, max {s['nodes_max']:.0f}), "
                  f"undirected edges {s['undirected_edges_mean']:.1f}, mean degree {s['degree_mean']:.2f}")


if __name__ == "__main__":
    main()
