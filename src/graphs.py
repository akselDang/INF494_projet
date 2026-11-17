"""Image -> superpixel graph.

Each image is segmented into superpixels with SLIC. Every superpixel becomes a node
described by a few simple features, and two superpixels that touch (8-neighbourhood)
are joined by an edge in both directions. The adjacency is the same as
skimage's region adjacency graph (``rag_mean_color`` with ``connectivity=2``), computed
with numpy instead of a per-pixel Python filter, which makes graph construction much faster.
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import torch
from skimage.segmentation import relabel_sequential, slic
from torch_geometric.data import Data

SLIC_PARAMS = {
    # Intensities in [0, 1]: a low compactness favours intensity over position,
    # so superpixels follow the strokes of the digits.
    "mnist": dict(n_segments=75, compactness=0.1, channel_axis=None),
    # RGB images: scikit-image default compactness.
    "cifar10": dict(n_segments=75, compactness=10.0, channel_axis=-1),
}

FEATURE_NAMES = {
    "mnist": ["mean_intensity", "x", "y", "area", "std_intensity"],
    "cifar10": ["mean_r", "mean_g", "mean_b", "x", "y", "area", "std_intensity"],
}


def segment(img: np.ndarray, dataset: str) -> np.ndarray:
    """SLIC superpixels, labelled 0..K-1 with no gaps."""
    seg = slic(img, start_label=1, **SLIC_PARAMS[dataset])
    seg, _, _ = relabel_sequential(seg)
    return seg - 1


def node_features(img: np.ndarray, seg: np.ndarray, dataset: str) -> np.ndarray:
    """Per-superpixel features, shape (K, F).

    x and y are the centroid coordinates divided by the image width and height,
    area is the fraction of the image covered, std_intensity is the standard deviation
    of the (grey-level) intensity inside the superpixel.
    """
    H, W = seg.shape
    lab = seg.ravel()
    K = int(lab.max()) + 1
    count = np.bincount(lab, minlength=K).astype(np.float64)
    ys, xs = np.indices((H, W))
    cx = np.bincount(lab, xs.ravel(), K) / count / W
    cy = np.bincount(lab, ys.ravel(), K) / count / H
    area = count / (H * W)
    gray = img if img.ndim == 2 else img.mean(axis=2)
    g = gray.ravel().astype(np.float64)
    g_mean = np.bincount(lab, g, K) / count
    g_std = np.sqrt(np.maximum(np.bincount(lab, g * g, K) / count - g_mean**2, 0.0))
    if dataset == "mnist":
        cols = [g_mean, cx, cy, area, g_std]
    else:
        rgb = [np.bincount(lab, img[..., c].ravel().astype(np.float64), K) / count for c in range(3)]
        cols = rgb + [cx, cy, area, g_std]
    return np.stack(cols, axis=1).astype(np.float32)


def region_adjacency(seg: np.ndarray) -> np.ndarray:
    """Edges between superpixels that touch (8-neighbourhood), both directions, shape (2, E)."""
    pairs = []
    for a, b in [
        (seg[:, :-1], seg[:, 1:]),      # horizontal
        (seg[:-1, :], seg[1:, :]),      # vertical
        (seg[:-1, :-1], seg[1:, 1:]),   # diagonal
        (seg[:-1, 1:], seg[1:, :-1]),   # anti-diagonal
    ]:
        m = a != b
        pairs.append(np.stack([a[m], b[m]], axis=1))
    p = np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0)
    return np.concatenate([p, p[:, ::-1]], axis=0).T.astype(np.int64)


def image_to_arrays(img: np.ndarray, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Node features (K, F) and edge index (2, E) for one image with values in [0, 1]."""
    seg = segment(img, dataset)
    return node_features(img, seg, dataset), region_adjacency(seg)


def _convert(job):
    img, dataset = job
    return image_to_arrays(img, dataset)


def build_graphs(images: np.ndarray, labels: np.ndarray, dataset: str, workers: int = 1) -> list[Data]:
    jobs = ((img, dataset) for img in images)
    out = []
    if workers > 1:
        with Pool(workers) as pool:
            for i, res in enumerate(pool.imap(_convert, jobs, chunksize=256), 1):
                out.append(res)
                if i % 10000 == 0:
                    print(f"  {i}/{len(images)} graphs")
    else:
        for i, job in enumerate(jobs, 1):
            out.append(_convert(job))
            if i % 10000 == 0:
                print(f"  {i}/{len(images)} graphs")
    return [
        Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(e), y=torch.tensor([int(y)]))
        for (x, e), y in zip(out, labels)
    ]


def load_or_build(dataset: str, split: str, images, labels, cache_dir: str = "data/graphs",
                  workers: int = 1) -> list[Data]:
    """Load cached graphs for (dataset, split), or build and cache them."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{dataset}_{split}_graphs.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    print(f"Building {len(images)} {dataset} {split} graphs -> {path}")
    graphs = build_graphs(images, labels, dataset, workers)
    torch.save(graphs, path)
    return graphs


def overlay_superpixels(img: np.ndarray, seg: np.ndarray, scale: int = 8) -> np.ndarray:
    """Image enlarged ``scale`` times with the superpixel boundaries drawn in red (for figures)."""
    from skimage.segmentation import mark_boundaries

    big = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    big_seg = np.repeat(np.repeat(seg, scale, axis=0), scale, axis=1)
    return mark_boundaries(big, big_seg, color=(1, 0.15, 0.15), mode="inner")


def graph_stats(graphs: list[Data]) -> dict:
    """Average number of nodes, undirected edges and degree."""
    nodes = np.array([g.num_nodes for g in graphs], dtype=float)
    edges = np.array([g.edge_index.size(1) / 2 for g in graphs], dtype=float)
    return {
        "nodes_mean": nodes.mean(),
        "nodes_min": nodes.min(),
        "nodes_max": nodes.max(),
        "undirected_edges_mean": edges.mean(),
        "degree_mean": (2 * edges / nodes).mean(),
    }
