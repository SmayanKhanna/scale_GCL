# gcl.py
import os
import json
import pickle
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Subset

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

from aug import drop_node, drop_edge, subgraph_khop, drop_feature
from models import GINE_Encoder
from losses import info_nce_loss
from eval_ogb import linear_probe


#utils
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_two_views(batch, transform):
    graphs = batch.to_data_list()
    v1, v2 = [], []
    for g in graphs:
        a, b = transform(g)
        v1.append(a)
        v2.append(b)
    return Batch.from_data_list(v1), Batch.from_data_list(v2)


def get_embeddings(encoder, loader, device='cuda'):
    encoder.eval()
    ret, y = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            if x is None:
                x = torch.ones((batch.shape[0], 1), device=device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(torch.float)
            g_z, _ = encoder(x, edge_index, batch, edge_attr)
            ret.append(g_z.cpu().numpy())
            y.append(data.y.view(-1).cpu().numpy())
    return np.concatenate(ret, 0), np.concatenate(y, 0)

#transform

class GraphCLTransform:
    """
    Simple recipes with a single probability p:
      - 'nd'     : node-drop only (keep_prob = 1 - p)
      - 'ed'     : edge-drop only (drop p)
      - 'nd+ed'  : node-drop then edge-drop (both use p)
      - 'rws'    : k-hop/ego subgraph (ignores p)
    Each view is sampled independently.
    """
    def __init__(self, recipe='nd', p=0.2, k_hop=2):
        self.recipe = recipe
        self.p = float(p)
        self.k = int(k_hop)

    #ive realised that having drop prob and keep drop is not ideal but in this case, they are implemented correctly (thankfully). 
    #Keep prob for drop nodes and drop prob for drop edges.
    def _nd(self, g):
        out = deepcopy(g)
        ei, ea = drop_node(out.edge_index, getattr(out, 'edge_attr', None),
                           keep_prob=max(0.0, 1.0 - self.p))
        out.edge_index, out.edge_attr = ei, ea
        return out

    def _ed(self, g):
        out = deepcopy(g)
        ei, ea = drop_edge(out.edge_index, self.p, edge_weights=getattr(out, 'edge_attr', None))
        out.edge_index, out.edge_attr = ei, ea
        return out

    def _rws(self, g):
        return subgraph_khop(g, self.k)
        
    def _apply(self, g):
        if self.recipe == 'nd':
            return self._nd(g)
        if self.recipe == 'ed':
            return self._ed(g)
        if self.recipe == 'nd+ed':
            return self._ed(self._nd(g))
        if self.recipe == 'rws':
            return self._rws(g)

        # default: nd
        return self._nd(g)

    def __call__(self, data):
        return self._apply(data), self._apply(data)


#model wrapper

class GraphCLModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_layers=3, proj_dim=32, edge_dim=None):
        super().__init__()
        self.encoder = GINE_Encoder(in_dim, hidden_dim, num_layers, edge_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def encode_graph(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        edge_attr = edge_attr.to(torch.float) if edge_attr is not None else None
        graph_z, _node_z = self.encoder(x, edge_index, batch, edge_attr)
        return graph_z

    def forward(self, data):
        return self.encode_graph(data)

    def project(self, data):
        with torch.set_grad_enabled(self.training):
            g = self.encode_graph(data)
            z = self.proj(g)
        return z


# helpers (data)

def choose_train_subset(train_idx, pct, seed):
    idx = np.array(train_idx, dtype=np.int64, copy=True)
    if pct is None or pct >= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)                       # in-place on the local copy, not the original
    keep_n = max(1, int(len(idx) * pct))
    return idx[:keep_n]

def build_loaders(dataset, idxs, batch_size, seed, num_workers=0):
    train_idx, valid_idx, test_idx = idxs
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    test_set  = Subset(dataset, test_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=g, num_workers=num_workers, worker_init_fn=_worker_init)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


# main
def main():
    parser = argparse.ArgumentParser()
    # data / device
    parser.add_argument('--dataset', type=str, default='ogbg-molhiv')
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--device', type=str, default='cuda')

    # result directory
    parser.add_argument('--result_dir', type=str, default='./results')

    # seed (single knob)
    parser.add_argument('--seed', type=int, default=42)

    # scaling
    parser.add_argument('--max_train_pct', type=float, default=0.1, help='fraction of train split used')

    # training budget
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # encoder / projector
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--proj_dim', type=int, default=32)

    # augmentations
    parser.add_argument('--recipe', type=str, default='nd', choices=['nd', 'ed', 'nd+ed', 'rws'])
    parser.add_argument('--p', type=float, default=0.2)
    parser.add_argument('--k_hop', type=int, default=2)

    # contrastive
    parser.add_argument('--temp', type=float, default=0.2)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # ---- data
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=args.root)
    split = dataset.get_idx_split()
    in_dim = dataset.num_features
    edge_dim = dataset.num_edge_features

    train_idx, valid_idx, test_idx = split['train'], split['valid'], split['test']
    no_train_graphs = len(train_idx)

    # choose subset with the SAME seed (simple, consistent)
    chosen_train = choose_train_subset(train_idx, args.max_train_pct, seed=args.seed)
    idxs = (chosen_train, valid_idx, test_idx)

    # loaders (deterministic w.r.t. seed)
    train_loader, valid_loader, test_loader = build_loaders(
        dataset, idxs, args.batch_size, seed=args.seed, num_workers=0
    )

    # model + opt
    model = GraphCLModel(in_dim, hidden_dim=args.hidden_dim,
                         num_layers=args.num_layers, proj_dim=args.proj_dim, edge_dim=edge_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # transform (single p)
    transform = GraphCLTransform(recipe=args.recipe, p=args.p, k_hop=args.k_hop)

    # untrained probe
    Z_train, y_train = get_embeddings(model.encoder, train_loader, device)
    Z_valid, y_valid = get_embeddings(model.encoder, valid_loader, device)
    Z_test,  y_test  = get_embeddings(model.encoder, test_loader,  device)
    best_val_auc_untrained, test_auc_untrained = linear_probe(
        Z_train, y_train, Z_valid, y_valid, Z_test, y_test, dataset=args.dataset
    )

    # pretrain?
    model.train()
    for epoch in range(1, args.epochs + 1):
        # reseed ONLY aug randomness so masks differ each epoch
        epoch_aug_seed = args.seed + epoch * 100003
        torch.manual_seed(epoch_aug_seed)
        np.random.seed(epoch_aug_seed)

        total_loss, total_graphs = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            v1, v2 = make_two_views(batch, transform)
            z1 = model.project(v1)
            z2 = model.project(v2)
            loss = info_nce_loss(z1, z2, temp=args.temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        avg = total_loss / max(total_graphs, 1)
        print(f"[Epoch {epoch:03d}] pretrain loss: {avg:.4f}")

    # trained probe
    Z_train, y_train = get_embeddings(model.encoder, train_loader, device)
    Z_valid, y_valid = get_embeddings(model.encoder, valid_loader, device)
    Z_test,  y_test  = get_embeddings(model.encoder, test_loader,  device)
    best_val_auc, test_auc = linear_probe(
        Z_train, y_train, Z_valid, y_valid, Z_test, y_test, dataset=args.dataset, random_state=args.seed
    )

    # save
    # save
    os.makedirs(args.result_dir, exist_ok=True)

    results = {
        "dataset": args.dataset,
        "subset": {
            "seed": args.seed,
            "max_train_pct": float(args.max_train_pct),
            "chosen_train_size": int(len(chosen_train)),
            "no_train_graphs_total": int(no_train_graphs)
        },
        "hparams": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "proj_dim": args.proj_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "temp": args.temp,
            "recipe": args.recipe,
            "p": args.p,
            "k_hop": args.k_hop
        },
        "untrained": {
            "val_auc": float(best_val_auc_untrained),
            "test_auc": float(test_auc_untrained)
        },
        "trained": {
            "val_auc": float(best_val_auc),
            "test_auc": float(test_auc)
        }
    }

    tag = f"{args.dataset}_pct{args.max_train_pct}_seed{args.seed}"
    with open(os.path.join(args.result_dir, f"{tag}.pkl"), "wb") as f:
        pickle.dump(results, f)
    with open(os.path.join(args.result_dir, f"{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.result_dir}/{tag}.pkl and .json")

    print(f"Untrained: Val AUC: {best_val_auc_untrained:.4f}, Test AUC: {test_auc_untrained:.4f}")
    print(f"Trained:   Val AUC: {best_val_auc:.4f},   Test AUC: {test_auc:.4f}")
    print(f"Saved to {args.result_dir}/{tag}.pkl and .json")


if __name__ == "__main__":
    main()
