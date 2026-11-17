"""Models: two GNNs (GCN, GraphSAGE) and two CNN baselines (a small LeNet-5 and a larger CNN)."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

GRAPH_MODELS = {"gcn", "sage"}
IMAGE_MODELS = {"cnn", "lenet"}


class GNN(nn.Module):
    """Three graph convolutions, global mean pooling over the nodes, then a 2-layer MLP.

    ``conv`` is GCNConv (degree-normalised mean over the neighbours, self included)
    or SAGEConv (separate weights for the node itself and for the mean of its neighbours).
    """

    def __init__(self, conv, in_channels: int, hidden: int = 64, num_classes: int = 10,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = conv(in_channels, hidden)
        self.conv2 = conv(hidden, 2 * hidden)
        self.conv3 = conv(2 * hidden, 2 * hidden)
        self.lin1 = nn.Linear(2 * hidden, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.drop(F.relu(self.lin1(x)))
        return self.lin2(x)


class MnistCNN(nn.Module):
    """Two 3x3 conv blocks and a dense layer (421,642 parameters)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(64 * 7 * 7, 128)
        self.lin2 = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.lin1(x.flatten(1))))
        return self.lin2(x)


class CifarCNN(nn.Module):
    """Three conv + batch-norm + pooling blocks and a dense layer (2,474,506 parameters)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv2, self.bn2 = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.conv3, self.bn3 = nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(256 * 4 * 4, 512)
        self.lin2 = nn.Linear(512, num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(F.relu(self.lin1(x.flatten(1))))
        return self.lin2(x)


class LeNet5(nn.Module):
    """Classic LeNet-5 (about 62k parameters): a CNN with a parameter budget close to GraphSAGE."""

    def __init__(self, in_channels: int = 1, padding: int = 2, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=padding)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(16 * 5 * 5, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.lin1(x.flatten(1)))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


def build_model(name: str, dataset: str, in_features: int) -> nn.Module:
    hidden = 64 if dataset == "mnist" else 128
    if name == "gcn":
        return GNN(GCNConv, in_features, hidden)
    if name == "sage":
        return GNN(SAGEConv, in_features, hidden)
    if name == "cnn":
        return MnistCNN() if dataset == "mnist" else CifarCNN()
    if name == "lenet":
        return LeNet5(in_channels=1, padding=2) if dataset == "mnist" else LeNet5(in_channels=3, padding=0)
    raise ValueError(f"unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
