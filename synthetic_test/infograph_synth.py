# infograph_synth.py - infograph

import os
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from evaluate_embeddings import evaluate_embedding
from model import Encoder

# ───────────────────────── utils ─────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_node_features(batch):
    """Ensure batch.x exists; if None, create constant 1-D features on correct device."""
    if getattr(batch, "x", None) is None:
        n = int(batch.num_nodes) if getattr(batch, "num_nodes", None) is not None else int(batch.edge_index.max().item()) + 1
        batch.x = torch.ones((n, 1), dtype=torch.float32, device=batch.edge_index.device)
    return batch

def infer_node_vs_graph(encoder, one_batch, device):
    """
    Robustly infer which output is node-level vs graph-level.
    Returns (node_emb_dim, graph_emb_dim, node_first: bool)
    """
    encoder.eval()
    with torch.no_grad():
        b = one_batch.to(device)
        b = ensure_node_features(b)
        outs = encoder(b.x, b.edge_index, b.batch)
    if not (isinstance(outs, (list, tuple)) and len(outs) == 2):
        raise RuntimeError("Encoder must return a tuple of (graph_emb, node_emb) OR (node_emb, graph_emb).")
    a, c = outs
    n_nodes = b.x.size(0)
    if a.size(0) == n_nodes:
        node_first = True
        node_dim, graph_dim = a.size(1), c.size(1)
    elif c.size(0) == n_nodes:
        node_first = False
        node_dim, graph_dim = c.size(1), a.size(1)
    else:
        # fallback: assume smaller batch = graphs
        node_first = (a.size(0) > c.size(0))
        node_dim, graph_dim = (a.size(1), c.size(1)) if node_first else (c.size(1), a.size(1))
    return node_dim, graph_dim, node_first

# InfoGraph core

class IdentityMLP(nn.Module):
    """Small projection head that preserves dimensionality (d→d)."""
    def __init__(self, dim: int, hidden: int = None, depth: int = 2):
        super().__init__()
        h = dim if hidden is None else hidden
        layers = []
        in_d = dim
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(in_d, h), nn.ReLU(inplace=True)]
            in_d = h
        layers += [nn.Linear(in_d, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BilinearScorer(nn.Module):
    """Score(h, g) = (h W) · g (no bias)."""
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # h: [N, d], g: [B, d] -> scores: [N, B]
        return self.W(h) @ g.t()

class InfoGraphWrapper(nn.Module):
    """
    Wrap your Encoder with local/global projection heads and a bilinear discriminator.
    """
    def __init__(self, encoder: Encoder, node_dim: int, graph_dim: int, proj_depth: int = 2, temperature: float = 0.2):
        super().__init__()

        if node_dim != graph_dim:
            D = min(node_dim, graph_dim)
            self.adapt_local  = nn.Linear(node_dim, D, bias=False)
            self.adapt_global = nn.Linear(graph_dim, D, bias=False)
            proj_in = D
        else:
            self.adapt_local = self.adapt_global = nn.Identity()
            proj_in = node_dim

        self.encoder = encoder
        self.local_proj  = IdentityMLP(proj_in, depth=proj_depth)
        self.global_proj = IdentityMLP(proj_in, depth=proj_depth)
        self.scorer = BilinearScorer(proj_in)
        self.temp = temperature

    def forward_node_graph(self, batch: Batch, device):
        batch = batch.to(device)
        batch = ensure_node_features(batch)
        outs = self.encoder(batch.x, batch.edge_index, batch.batch)
        if not (isinstance(outs, (list, tuple)) and len(outs) == 2):
            raise RuntimeError("Encoder must return a tuple of (graph_emb, node_emb) OR (node_emb, graph_emb).")
        a, c = outs
        n_nodes = batch.x.size(0)
        if a.size(0) == n_nodes:
            node_emb, graph_emb = a, c
        elif c.size(0) == n_nodes:
            node_emb, graph_emb = c, a
        else:
            node_emb, graph_emb = (a, c) if a.size(0) > c.size(0) else (c, a)

        z_local  = self.local_proj(self.adapt_local(node_emb))
        z_global = self.global_proj(self.adapt_global(graph_emb))
        # L2 norm
        z_local  = F.normalize(z_local,  dim=1)
        z_global = F.normalize(z_global, dim=1)
        return z_local, z_global, batch.batch

def jsd_loss(scores_pos: torch.Tensor, scores_neg: torch.Tensor):
    """
    Jensen–Shannon MI estimator (as in Deep InfoMax/InfoGraph):
      L = E_pos[softplus(-s)] + E_neg[softplus(s)]
    where s are discriminator logits.
    """
    loss_pos = F.softplus(-scores_pos).mean()
    loss_neg = F.softplus(scores_neg).mean()
    return loss_pos + loss_neg

def infograph_jsd_objective(z_local: torch.Tensor, z_global: torch.Tensor, batch_vec: torch.Tensor, scorer: BilinearScorer, temperature: float):
    """
    Build positive/negative logits from all node↔graph pairs in the batch.
      - positives: node i with its own graph summary g_{batch[i]}
      - negatives: node i with all other graphs' summaries
    """
    # scores: [N, B]
    scores = scorer(z_local, z_global) / max(temperature, 1e-6)
    N, B = scores.size()

    # Positives: gather per-node
    device = scores.device
    node_indices = torch.arange(N, device=device)
    pos_scores = scores[node_indices, batch_vec]  # [N]

    # Negatives: mask out positives, keep others
    mask = F.one_hot(batch_vec, num_classes=B).bool()  # [N, B]
    neg_scores = scores.masked_fill(mask, float('-inf'))
    # Keep finite entries as negatives:
    neg_scores = neg_scores[~mask]
    # Compute JSD loss
    return jsd_loss(pos_scores, neg_scores)

# ───────────────────────── main ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth_path", type=str, required=True,
                        help="Path to data_list.pt or a directory containing it.")
    parser.add_argument("--dataset_tag", type=str, default="synth",
                        help="Used only for naming saved result files.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # scaling: fraction of ALL graphs used for contrastive pretraining
    parser.add_argument("--train_frac", type=float, default=1.0)

    # encoder / training
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--proj_depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--result_dir", type=str, default="./results_synth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load dataset
    path = args.synth_path
    if os.path.isdir(path):
        path = os.path.join(path, "data_list.pt")
    data_list = torch.load(path)
    assert isinstance(data_list, list) and len(data_list) > 0, "data_list.pt must be a non-empty list[Data]"

    # infer input feature dim; if None, we will synthesize features in-batch
    in_dim = data_list[0].x.size(1) if getattr(data_list[0], "x", None) is not None else 1

    # loaders
    full_loader  = DataLoader(data_list, batch_size=args.batch_size, shuffle=False)
    N = len(data_list)
    rng = np.random.default_rng(args.seed)
    train_size = max(1, int(N * float(args.train_frac)))
    train_idx  = rng.choice(np.arange(N), size=train_size, replace=False)
    train_loader = DataLoader(Subset(data_list, train_idx), batch_size=args.batch_size, shuffle=True)

    # model
    base_encoder = Encoder(in_dim, args.dim, args.num_layers).to(device)

    # Infer dims and ordering for heads
    tiny = next(iter(DataLoader([data_list[train_idx[0]]], batch_size=1)))
    node_dim, graph_dim, _ = infer_node_vs_graph(base_encoder, tiny, device)

    model = InfoGraphWrapper(
        encoder=base_encoder,
        node_dim=node_dim,
        graph_dim=graph_dim,
        proj_depth=args.proj_depth,
        temperature=args.temperature,
    ).to(device)

    # untrained probe
    emb_u, y_all = model.encoder.get_embeddings(full_loader, device=device)
    val_u, test_u = evaluate_embedding(emb_u, y_all, device=device, search=True)

    # pretrain
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss, ngraphs = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            z_local, z_global, bvec = model.forward_node_graph(batch, device)
            loss = infograph_jsd_objective(z_local, z_global, bvec, model.scorer, model.temp)
            loss.backward()
            opt.step()
            ngraphs += int(bvec.max().item()) + 1
            total_loss += loss.item() * (int(bvec.max().item()) + 1)
        print(f"[{epoch:03d}] loss={total_loss / max(ngraphs, 1):.4f}")

    # Trained embeddings + probe
    emb_t, _ = model.encoder.get_embeddings(full_loader, device=device)
    val_t, test_t = evaluate_embedding(emb_t, y_all, device=device, search=True)

    # save
    os.makedirs(args.result_dir, exist_ok=True)
    tag = f"{args.dataset_tag}_frac{args.train_frac}_seed{args.seed}_methodinfograph_nopygcl"
    results = {
        "dataset": args.dataset_tag,
        "num_graphs": int(N),
        "train_frac": float(args.train_frac),
        "method": "infograph_nopygcl",
        "hparams": {
            "dim": args.dim, "num_layers": args.num_layers,
            "proj_depth": args.proj_depth, "temperature": args.temperature,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "loss": "JSD", "mode": "G2L"
        },
        "untrained": {"val": float(val_u), "test": float(test_u)},
        "trained":   {"val": float(val_t), "test": float(test_t)}
    }
    with open(os.path.join(args.result_dir, f"{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.result_dir, f"{tag}.pkl"), "wb") as f:
        pickle.dump(results, f)

    print("=== RESULTS (InfoGraph, no PyGCL) ===")
    print(f"Untrained  | val={val_u:.4f}  test={test_u:.4f}")
    print(f"Trained    | val={val_t:.4f}  test={test_t:.4f}")
    print(f"Saved to {args.result_dir}/{tag}.json (and .pkl)")

if __name__ == "__main__":
    main()
