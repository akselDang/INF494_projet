"""Train one model with a train / validation / test protocol.

The best epoch is chosen on the validation set (5,000 training images held out).
The test set is used once, at the end, with the weights of that best epoch.

Examples
--------
python -m src.train --dataset mnist --model sage --seed 0
python -m src.train --dataset mnist --model gcn --features 3 --seed 1
python -m src.train --dataset mnist --model lenet --seed 2
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import DataLoader as ImageLoader
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader

from .data import load_images, train_val_split
from .graphs import FEATURE_NAMES, load_or_build
from .models import GRAPH_MODELS, build_model, count_parameters


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--model", choices=["gcn", "sage", "cnn", "lenet"], required=True)
    p.add_argument("--seed", type=int, default=0, help="seed for weight initialisation and shuffling")
    p.add_argument("--features", type=int, default=None,
                   help="MNIST only: 3 keeps (mean_intensity, x, y), default keeps all 5")
    p.add_argument("--raw-features", action="store_true",
                   help="graph models: do not standardise the node features (first version of the project)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--step-size", type=int, default=8, help="StepLR: halve the learning rate every N epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--max-train", type=int, default=None, help="use only N training images (quick tests)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="processes used to build the graphs the first time")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--ckpt-dir", default="checkpoints")
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, is_graph: bool, device, optimizer=None):
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates.

    Returns (mean loss, accuracy, predictions); predictions are only kept in evaluation.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total, preds = 0.0, 0, 0, []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if is_graph:
                batch = batch.to(device)
                out, y = model(batch.x, batch.edge_index, batch.batch), batch.y
            else:
                xb, y = batch[0].to(device), batch[1].to(device)
                out = model(xb)
            loss = F.cross_entropy(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pred = out.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if not training:
                preds.append(pred.cpu())
    return total_loss / total, correct / total, (torch.cat(preds).numpy() if preds else None)


def keep_features(graphs, k: int):
    return [Data(x=g.x[:, :k], edge_index=g.edge_index, y=g.y) for g in graphs]


def standardize(train, *others):
    """Z-score the node features with the mean and std of the training nodes only.

    The raw features have very different scales (area is about 0.01, intensity up to 1),
    so without this the small-scale features barely influence the first layer.
    """
    x = torch.cat([g.x for g in train])
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-8)
    scale = lambda gs: [Data(x=(g.x - mu) / sd, edge_index=g.edge_index, y=g.y) for g in gs]
    return [scale(train)] + [scale(o) for o in others]


def make_datasets(args):
    """Train, validation and test sets for the chosen model type."""
    x_tr, y_tr, x_te, y_te = load_images(args.dataset, args.data_dir)
    tr_idx, va_idx = train_val_split(len(y_tr), args.val_size)
    if args.max_train:
        tr_idx = tr_idx[: args.max_train]
    if args.model in GRAPH_MODELS:
        cache = os.path.join(args.data_dir, "graphs")
        g_tr = load_or_build(args.dataset, "train", x_tr, y_tr, cache, args.workers)
        g_te = load_or_build(args.dataset, "test", x_te, y_te, cache, args.workers)
        n_feat = len(FEATURE_NAMES[args.dataset])
        k = args.features or n_feat
        if k != n_feat:
            if args.dataset != "mnist" or k != 3:
                raise ValueError("--features only supports 3 on MNIST")
            g_tr, g_te = keep_features(g_tr, k), keep_features(g_te, k)
        train, val, test = [g_tr[i] for i in tr_idx], [g_tr[i] for i in va_idx], g_te
        if not args.raw_features:
            train, val, test = standardize(train, val, test)
        return train, val, test, k
    to_tensor = (lambda a: torch.from_numpy(a).unsqueeze(1)) if args.dataset == "mnist" \
        else (lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous())
    xt, yt = to_tensor(x_tr), torch.from_numpy(y_tr).long()
    return (TensorDataset(xt[tr_idx], yt[tr_idx]), TensorDataset(xt[va_idx], yt[va_idx]),
            TensorDataset(to_tensor(x_te), torch.from_numpy(y_te).long()), None)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_graph = args.model in GRAPH_MODELS
    batch_size = args.batch_size
    Loader = GraphLoader if is_graph else ImageLoader

    train_set, val_set, test_set, n_feat = make_datasets(args)
    train_loader = Loader(train_set, batch_size=batch_size, shuffle=True)
    eval_loaders = {name: Loader(ds, batch_size=256, shuffle=False)
                    for name, ds in [("train", train_set), ("val", val_set), ("test", test_set)]}

    model = build_model(args.model, args.dataset, n_feat or 0).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    run_name = (args.model + (f"_{args.features}feat" if args.features else "")
                + ("_raw" if args.raw_features and is_graph else "") + f"_seed{args.seed}")
    print(f"{args.dataset} | {run_name} | {n_params:,} parameters | {device} | "
          f"train {len(train_set)} / val {len(val_set)} / test {len(test_set)}")
    print(f"{'epoch':>5} | {'train loss':>10} | {'train acc':>9} | {'val loss':>8} | {'val acc':>7} | {'time':>6}")

    history, best_val, best_epoch, best_state = [], -1.0, 0, None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, is_graph, device, optimizer)
        va_loss, va_acc, _ = run_epoch(model, eval_loaders["val"], is_graph, device)
        scheduler.step()
        dt = time.time() - t0
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc_dropout=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, seconds=dt))
        flag = ""
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            flag = " *"
        print(f"{epoch:>5} | {tr_loss:>10.4f} | {tr_acc:>8.2%} | {va_loss:>8.4f} | {va_acc:>6.2%} | {dt:>5.1f}s{flag}")
    train_seconds = time.time() - t_start

    # Final evaluation, once, with the weights of the best validation epoch.
    model.load_state_dict(best_state)
    _, train_acc, _ = run_epoch(model, eval_loaders["train"], is_graph, device)
    test_loss, test_acc, test_preds = run_epoch(model, eval_loaders["test"], is_graph, device)
    print(f"\nbest epoch {best_epoch}: val {best_val:.2%} | train (eval mode) {train_acc:.2%} | test {test_acc:.2%}")

    out_dir = os.path.join(args.out_dir, args.dataset)
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(best_state, os.path.join(ckpt_dir, f"{run_name}.pt"))
    np.save(os.path.join(out_dir, f"{run_name}_test_preds.npy"), test_preds)
    result = dict(
        dataset=args.dataset, model=args.model, features=n_feat, seed=args.seed,
        run_name=run_name, group=run_name.rsplit("_seed", 1)[0],
        n_params=n_params, best_epoch=best_epoch, val_acc=best_val,
        train_acc_eval=train_acc, test_acc=test_acc, test_loss=test_loss,
        train_seconds=train_seconds, config=vars(args), batch_size=batch_size, device=str(device),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      torch_geometric=torch_geometric.__version__),
        history=history,
    )
    with open(os.path.join(out_dir, f"{run_name}.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
