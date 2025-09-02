#run graph_cl on synthetic dataset
import os, argparse, json, pickle, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from copy import deepcopy
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, GINConv
from torch_geometric.utils import dropout_edge, subgraph, k_hop_subgraph, dropout_node
from torch.distributions import Bernoulli
from torch.utils.data import Subset
from evaluate_embeddings import evaluate_embedding
from model import Encoder

# ───────────────────────── utils ─────────────────────────

def set_seed(seed: int):
	import random
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def info_nce_loss(z1: Tensor, z2: Tensor, temp: float = 0.2):
    """
    Graph-level NT-Xent loss (SimCLR style).
    z1, z2: [B, D]
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)

    reps = torch.cat([z1, z2], dim=0)                 # [2B, D]
    sim = reps @ reps.t() / temp                      # cosine sim since normalized
    mask = torch.eye(2*B, dtype=torch.bool, device=sim.device)
    sim.masked_fill_(mask, float('-inf'))

    # positives: (i, i+B) and (i+B, i)
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(sim.device)
    logits = sim
    labels = pos

    loss = F.cross_entropy(logits, labels)
    return loss

# ─────────────────────── augmentations ───────────────────

def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x

#pyGCL's implementation
# def drop_node(edge_index: torch.Tensor, edge_weight: torch.Tensor = None, keep_prob: float = 0.5):
#     num_nodes = edge_index.max().item() + 1
#     probs = torch.tensor([keep_prob for _ in range(num_nodes)])
#     dist = Bernoulli(probs)

#     subset = dist.sample().to(torch.bool).to(edge_index.device)
#     ei, ew = subgraph(subset, edge_index, edge_weight)

#     return ei, ew

# or we can use PyG's node dropout funciton

def drop_node(edge_index: torch.Tensor, drop_prob: float = 0.5):
	ei, e_mask, n_mask = dropout_node(edge_index, p=drop_prob, training=True, relabel_nodes=True)
	return ei, e_mask, n_mask

def drop_edge(edge_index: torch.LongTensor,
              drop_prob: float,
              force_undirected: bool = True,
              edge_weights: torch.Tensor = None
             ) :
    # remove a fraction drop_prob of edges
    ei, mask = dropout_edge(edge_index,
                             p=drop_prob,
                             force_undirected=force_undirected)
    ew = None
    if edge_weights is not None:
        ew = edge_weights[mask]
    return ei, ew

def subgraph_khop(data, num_hops: int):
    """
    Sample a random k-hop subgraph out of `data`:
      - pick one random node,
      - keep its k-hop neighbors,
      - re-index everything.
    """
    root = torch.randint(0, data.num_nodes, (1,)).item()
    node_idx, edge_index, _, batch_idx = k_hop_subgraph(
        root,
        num_hops,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        batch=data.batch if hasattr(data, 'batch') else None
    )
    out = deepcopy(data)
    out.x = data.x[node_idx]
    out.edge_index = edge_index
    if hasattr(data, 'batch'):
        out.batch = data.batch[node_idx]
    return out

class GraphCLTransform:
	def __init__(self, recipe='nd', p=0.2, k=2):
		self.recipe = recipe
		self.p = float(p)
		self.k = int(k)

	def _nd(self, g):
		out = deepcopy(g)
		ei, e_mask, n_mask = drop_node(out.edge_index, drop_prob=self.p)
		out.edge_index = ei
		out.edge_attr = out.edge_attr[e_mask] if out.edge_attr is not None else None
		out.x = out.x[n_mask] if out.x is not None else None

		return out

	def _ed(self, g):
		out = deepcopy(g)
		out.edge_index, out.edge_attr = drop_edge(out.edge_index, drop_prob=self.p)
		return out

	def _rws(self, g):
		return subgraph_khop(g, self.k)

	def _apply(self, g):
		if self.recipe == 'nd': return self._nd(g)
		if self.recipe == 'ed': return self._ed(g)
		if self.recipe == 'nd+ed': return self._ed(self._nd(g))
		if self.recipe == 'rws': return self._rws(g)
		return self._nd(g)

	def __call__(self, data):
		return self._apply(data), self._apply(data)

def make_two_views(batch, transform):

	# this is so we don't have to deal with batching logic but it might be slightly inefficient
	graphs = batch.to_data_list()
	v1, v2 = [], []
	for g in graphs:
		a, b = transform(g)
		v1.append(a); v2.append(b)
	return Batch.from_data_list(v1), Batch.from_data_list(v2)


class GraphCLModel(nn.Module):
	def __init__(self, in_dim, dim=32, num_layers=3, proj_dim=32):
		super().__init__()
		self.encoder = Encoder(in_dim, dim, num_layers)
		self.proj = nn.Sequential(
			nn.Linear(dim * num_layers, proj_dim),
			nn.ReLU(inplace=True),
			nn.Linear(proj_dim, proj_dim),
		)
	def project(self, batch):
		x, ei, bt = batch.x, batch.edge_index, batch.batch
		g, _ = self.encoder(x, ei, bt)
		return self.proj(g)

# ───────────────────────── main ─────────────────────────

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--synth_path", type=str, required=True,
	                    help="Path to data_list.pt or a directory containing it.")
	parser.add_argument("--dataset_tag", type=str, default="synth6",
	                    help="Used only for naming saved result files.")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--seed", type=int, default=42)

	# scaling: fraction of ALL graphs used for contrastive pretraining
	parser.add_argument("--train_frac", type=float, default=0.1)

	# encoder / training
	parser.add_argument("--dim", type=int, default=32)
	parser.add_argument("--num_layers", type=int, default=3)
	parser.add_argument("--proj_dim", type=int, default=32)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--lr", type=float, default=1e-2)
	parser.add_argument("--weight_decay", type=float, default=1e-5)

	# augs + loss
	parser.add_argument("--recipe", type=str, default="nd", choices=["nd","ed","nd+ed","rws"])
	parser.add_argument("--p", type=float, default=0.2)
	parser.add_argument("--k_hop", type=int, default=2)
	parser.add_argument("--temp", type=float, default=0.2)

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
	num_features = data_list[0].x.size(1) if data_list[0].x is not None else 1

	# loaders
	full_loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=False)
	N = len(data_list)
	rng = np.random.default_rng(args.seed)
	train_size = max(1, int(N * args.train_frac))
	train_idx = rng.choice(np.arange(N), size=train_size, replace=False)
	train_loader = DataLoader(Subset(data_list, train_idx), batch_size=args.batch_size, shuffle=True)

	# model
	model = GraphCLModel(num_features, dim=args.dim, num_layers=args.num_layers, proj_dim=args.proj_dim).to(device)
	opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	transform = GraphCLTransform(args.recipe, args.p, args.k_hop)

	# untrained embeddings
	emb_u, y_all = model.encoder.get_embeddings(full_loader, device=device)
	val_u, test_u = evaluate_embedding(emb_u, y_all, device=device, search=True)

	# contrastive pretraining
	model.train()
	for epoch in range(1, args.epochs + 1):
		total, ngraphs = 0.0, 0
		for batch in train_loader:
			batch = batch.to(device)
			v1, v2 = make_two_views(batch, transform)
			z1 = model.project(v1)
			z2 = model.project(v2)
			loss = info_nce_loss(z1, z2, temp=args.temp)
			opt.zero_grad(); loss.backward(); opt.step()
			total += loss.item() * batch.num_graphs
			ngraphs += batch.num_graphs
		print(f"[{epoch:03d}] loss={total/max(1, ngraphs):.4f}")

	# probe trained embeddings
	emb_t, _ = model.encoder.get_embeddings(full_loader, device=device)
	val_t, test_t = evaluate_embedding(emb_t, y_all, device=device, search=True)

	# save
	os.makedirs(args.result_dir, exist_ok=True)
	tag = f"{args.dataset_tag}_frac{args.train_frac}_seed{args.seed}_aug{args.recipe}"
	results = {
		"dataset": args.dataset_tag,
		"num_graphs": int(N),
		"train_frac": float(args.train_frac),
		"hparams": {
			"dim": args.dim, "num_layers": args.num_layers, "proj_dim": args.proj_dim,
			"batch_size": args.batch_size, "epochs": args.epochs,
			"lr": args.lr, "weight_decay": args.weight_decay,
			"recipe": args.recipe, "p": args.p, "k_hop": args.k_hop, "temp": args.temp
		},
		"untrained": {"val": float(val_u), "test": float(test_u)},
		"trained":   {"val": float(val_t), "test": float(test_t)}
	}
	with open(os.path.join(args.result_dir, f"{tag}.json"), "w") as f:
		json.dump(results, f, indent=2)
	with open(os.path.join(args.result_dir, f"{tag}.pkl"), "wb") as f:
		pickle.dump(results, f)
	print("=== RESULTS ===")
	print(f"Untrained  | val={val_u:.4f}  test={test_u:.4f}")
	print(f"Trained    | val={val_t:.4f}  test={test_t:.4f}")
	print(f"Saved to {args.result_dir}/{tag}.json (and .pkl)")

if __name__ == "__main__":
	main()
