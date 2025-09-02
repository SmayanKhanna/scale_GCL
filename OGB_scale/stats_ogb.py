# stats_probe_ogb.py
import os
import json
import pickle
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset

from eval_ogb import linear_probe


# ------------------------- utils -------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bins(bins_str: str) -> List[float]:
    tokens = [t.strip() for t in bins_str.split(",")]
    out = []
    for t in tokens:
        out.append(np.inf if t.lower() in ("inf", "+inf", "infty", "infinite") else float(t))
    if not np.all(np.diff(out) > 0):
        raise ValueError(f"Bins must be strictly increasing: {out}")
    return out


def choose_train_subset(idxs: torch.Tensor, pct: float, seed: int) -> np.ndarray:
    arr = np.asarray(idxs, dtype=np.int64).copy()
    if pct is None or pct >= 1.0:
        return arr
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    keep_n = max(1, int(len(arr) * pct))
    return arr[:keep_n]


def _label_to_int(y) -> int:
    if isinstance(y, torch.Tensor):
        y = y.view(-1)[0].item()
    return int(y)


def graph_stats_vector(data, bins: List[float]) -> np.ndarray:
    """
    Per-graph feature = [num_nodes, mean_degree, degree_histogram(bins)]
    """
    n = int(data.num_nodes)
    if n <= 0:
        avg_deg = 0.0
        hist = np.zeros(len(bins) - 1, dtype=float)
    else:
        degs = degree(data.edge_index[0], num_nodes=n)
        degs = degs.detach().cpu().numpy()
        avg_deg = float(degs.mean()) if degs.size > 0 else 0.0
        hist, _ = np.histogram(degs, bins=bins)
        hist = hist.astype(float)
    return np.concatenate(([float(n), avg_deg], hist))


def build_feature_matrix(dataset, indices: np.ndarray, bins: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in indices:
        d = dataset[int(idx)]
        X.append(graph_stats_vector(d, bins))
        y.append(_label_to_int(d.y))
    X = np.vstack(X).astype(np.float64)
    y = np.asarray(y, dtype=int)
    return X, y


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser(description="Handcrafted stats baseline using existing linear_probe (OGB).")
    # data
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--root", type=str, default="dataset")

    # subset + seed
    parser.add_argument("--max_train_pct", type=float, default=1.0, help="fraction of TRAIN split used")
    parser.add_argument("--seed", type=int, default=42)

    # features
    parser.add_argument("--bins", type=str, default="0,1,2,3,4,inf",
                        help="Comma-separated degree bin edges, e.g. '0,1,2,3,4,inf'")

    # output
    parser.add_argument("--result_dir", type=str, default="./results_stats")
    parser.add_argument("--save_pkl", action="store_true", help="also save a pickle of the results dict")

    args = parser.parse_args()
    set_seed(args.seed)

    # parse knobs
    bins = parse_bins(args.bins)

    # load dataset + splits
    dataset = PygGraphPropPredDataset(name=args.dataset, root=args.root)
    split = dataset.get_idx_split()
    train_idx = split["train"]
    valid_idx = split["valid"]
    test_idx  = split["test"]

    # subsample train
    chosen_train = choose_train_subset(train_idx, args.max_train_pct, seed=args.seed)

    # features
    X_train, y_train = build_feature_matrix(dataset, chosen_train, bins)
    X_val,   y_val   = build_feature_matrix(dataset, np.asarray(valid_idx), bins)
    X_test,  y_test  = build_feature_matrix(dataset, np.asarray(test_idx),  bins)

    # probe
    best_val_auc, test_auc = linear_probe(
        X_train, y_train, X_val, y_val, X_test, y_test,
        dataset=args.dataset, random_state=args.seed
    )

    # save
    os.makedirs(args.result_dir, exist_ok=True)
    tag = f"{args.dataset}_pct{args.max_train_pct}_seed{args.seed}"
    out_json = os.path.join(args.result_dir, f"stats_{tag}.json")

    results = {
        "dataset": args.dataset,
        "subset": {
            "seed": args.seed,
            "max_train_pct": float(args.max_train_pct),
            "chosen_train_size": int(len(chosen_train)),
            "no_train_graphs_total": int(len(train_idx)),
        },
        "features": {
            "bins": bins,
            "dim": int(2 + (len(bins) - 1)),  # [num_nodes, mean_degree] + histogram bins
        },
        "metrics": {
            "val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
        }
    }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    if args.save_pkl:
        out_pkl = os.path.join(args.result_dir, f"stats_{tag}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(results, f)

    print(json.dumps(results, indent=2))
    print(f"Saved: {out_json}" + (f" and {out_pkl}" if args.save_pkl else ""))


if __name__ == "__main__":
    main()
