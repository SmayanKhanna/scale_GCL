# run_stats_baseline.py — simple handcrafted-statistics baseline for graph-level classification.

import os, argparse, json, pickle, random, numpy as np, torch
from typing import List
from torch_geometric.utils import degree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_datalist(path: str):
    if os.path.isdir(path):
        path = os.path.join(path, "data_list.pt")
    data_list = torch.load(path)
    assert isinstance(data_list, list) and len(data_list) > 0, "data_list.pt must be a non-empty list[Data]"
    return data_list

def graph_stats_vector(data, bins: List[float]):
    """
    Build a simple per-graph feature vector:
      [num_nodes, mean_degree, degree_histogram(bins)]
    """
    n = int(data.num_nodes)

    degs = degree(data.edge_index[0], num_nodes=n).cpu().numpy() if n > 0 else np.array([0.0])
    avg_deg = float(degs.mean()) if n > 0 else 0.0
    hist, _ = np.histogram(degs, bins=bins)
    return np.concatenate([[n, avg_deg], hist.astype(float)])

def build_feature_matrix(data_list, bins: List[float]):
    X = []
    y = []
    for d in data_list:
        X.append(graph_stats_vector(d, bins))
        lbl = d.y
        if isinstance(lbl, torch.Tensor):
            lbl = lbl.item() if lbl.numel() == 1 else lbl.squeeze().item()
        y.append(int(lbl))
    X = np.vstack(X).astype(np.float64)
    y = np.array(y, dtype=int)
    return X, y

def eval_stats_baseline(X, y, seed: int, n_splits: int = 10):
    set_seed(seed)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    param_grid = {"svc__C": [1e-3,1e-2,1e-1,1,10,100,1_000]}
    accs = []
    for tr, te in kf.split(X, y):
        gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
        gs.fit(X[tr], y[tr])
        yhat = gs.predict(X[te])
        accs.append(accuracy_score(y[te], yhat))
    return float(np.mean(accs)), float(np.std(accs))

def main():
    parser = argparse.ArgumentParser(description="Handcrafted statistics baseline for graph-level classification.")
    parser.add_argument("--synth_path", type=str, required=True,
                        help="Path to data_list.pt or a directory containing it.")
    parser.add_argument("--dataset_tag", type=str, default="synth",
                        help="Used for naming saved result files.")
    parser.add_argument("--result_dir", type=str, default="./results_stats")
    parser.add_argument("--bins", type=float, nargs="+",
                        default=[0,1,2,3,4,5, np.inf],
                        help="Degree histogram bin edges (space-separated). Include np.inf at the end.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds for CV shuffles (reports mean/std across seeds).")
    parser.add_argument("--n_splits", type=int, default=10, help="StratifiedKFold splits.")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # Load and featurize
    data_list = load_datalist(args.synth_path)
    X, y = build_feature_matrix(data_list, bins=args.bins)

    # Evaluate across seeds
    per_seed = []
    for s in args.seeds:
        mean_acc, std_acc = eval_stats_baseline(X, y, seed=s, n_splits=args.n_splits)
        per_seed.append({"seed": int(s), "cv_mean_acc": mean_acc, "cv_std_acc": std_acc})
        print(f"[seed {s}] 10-fold CV accuracy = {mean_acc:.4f} (±{std_acc:.4f})")

    # Aggregate
    means = np.array([p["cv_mean_acc"] for p in per_seed])
    agg = {
        "dataset": args.dataset_tag,
        "num_graphs": int(len(data_list)),
        "feature_dim": int(X.shape[1]),
        "bins": [float(b) if np.isfinite(b) else "inf" for b in args.bins],
        "n_splits": int(args.n_splits),
        "seeds": [int(s) for s in args.seeds],
        "per_seed": per_seed,
        "aggregate": {
            "mean_of_means": float(means.mean()),
            "std_of_means": float(means.std(ddof=0)),
        },
    }

    tag = f"{args.dataset_tag}_stats_seed{args.seeds[0]}{'_multi' if len(args.seeds)>1 else ''}"
    json_path = os.path.join(args.result_dir, f"{tag}.json")
    pkl_path = os.path.join(args.result_dir, f"{tag}.pkl")
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    with open(pkl_path, "wb") as f:
        pickle.dump(agg, f)

    print("=== RESULTS (aggregate) ===")
    print(f"Mean 10-fold accuracy across seeds: {agg['aggregate']['mean_of_means']:.4f} "
          f"(std across seeds {agg['aggregate']['std_of_means']:.4f})")
    print(f"Saved to {json_path} and {pkl_path}")

if __name__ == "__main__":
    main()
