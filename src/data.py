"""Raw images and the fixed train / validation split."""
from __future__ import annotations

import numpy as np
import torchvision


def load_images(dataset: str, root: str = "data"):
    """Return (x_train, y_train, x_test, y_test) as numpy arrays with values in [0, 1].

    MNIST images have shape (N, 28, 28); CIFAR-10 images have shape (N, 32, 32, 3).
    """
    if dataset == "mnist":
        tr = torchvision.datasets.MNIST(root, train=True, download=True)
        te = torchvision.datasets.MNIST(root, train=False, download=True)
        return (tr.data.numpy().astype(np.float32) / 255.0, tr.targets.numpy(),
                te.data.numpy().astype(np.float32) / 255.0, te.targets.numpy())
    if dataset == "cifar10":
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        return (tr.data.astype(np.float32) / 255.0, np.array(tr.targets),
                te.data.astype(np.float32) / 255.0, np.array(te.targets))
    raise ValueError(f"unknown dataset: {dataset}")


def train_val_split(n: int, val_size: int, split_seed: int = 0):
    """Indices of the training and validation images.

    The split depends only on ``split_seed``, so every model and every training seed
    uses the same validation set. The official test set is never used for any choice.
    """
    perm = np.random.default_rng(split_seed).permutation(n)
    return np.sort(perm[val_size:]), np.sort(perm[:val_size])
