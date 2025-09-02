# build_multiclass_synth_TUstyle.py
import os, argparse, random, numpy as np, torch, networkx as nx
from scipy import sparse
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from motif_dict import MOTIFDICT
from create_dataset import make_class_dataset, get_correct_edges

def nx_to_pyg(G, class_id, feature_dim=10, feature_noise_p=0.0):
	# mark motif vs background edges
	correct = set(get_correct_edges(G))
	for u,v,d in G.edges(data=True):
		d["weight"] = 1 if ((int(u),int(v)) in correct or (int(v),int(u)) in correct) else -1

	A_w = nx.to_numpy_array(G, weight="weight", dtype=float, nonedge=0.0)
	edge_index, edge_attr = from_scipy_sparse_matrix(sparse.csr_matrix(A_w))
	A_unw = nx.to_numpy_array(G, dtype=float)
	_, edge_motif = from_scipy_sparse_matrix(sparse.csr_matrix(A_unw))

	x = torch.ones(G.number_of_nodes(), feature_dim, dtype=torch.float32)
	if feature_noise_p > 0:
		noise = torch.bernoulli(torch.full_like(x, feature_noise_p))
		x = torch.remainder(x + noise, 2.0)

	role_id = torch.tensor([G.nodes[n]["label"] for n in G.nodes()], dtype=torch.long)
	y = torch.tensor([class_id], dtype=torch.long)

	return Data(x=x, edge_index=edge_index, edge_attr=edge_attr.float(),
	            edge_motif=edge_motif.float(), role_id=role_id, y=y)

def main():
	p = argparse.ArgumentParser()
	p.add_argument("--motifs", type=str, default="auto6")
	p.add_argument("--samples_per_class", type=int, default=1000)
	p.add_argument("--background_graph", choices=["tree","ba"], default="tree")
	p.add_argument("--multiplier", type=float, default=4.0)
	p.add_argument("--feature_dim", type=int, default=10)
	p.add_argument("--feature_noise_p", type=float, default=0.0)
	p.add_argument("--out_dir", type=str, default="data/synth6_TU")
	p.add_argument("--seed", type=int, default=42)
	args = p.parse_args()

	random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

	if args.motifs == "auto6":
		keys = list(MOTIFDICT.keys())
		assert len(keys) >= 6, "Need at least 6 motifs in MOTIFDICT"
		motif_names = keys[:6]
	else:
		motif_names = [m.strip() for m in args.motifs.split(",")]
		for m in motif_names:
			if m not in MOTIFDICT: raise KeyError(f"Unknown motif '{m}'")

	os.makedirs(args.out_dir, exist_ok=True)

	data_list = []
	for class_id, name in enumerate(motif_names):
		graphs = make_class_dataset(
			sample_size=args.samples_per_class,
			motif_edge_list=MOTIFDICT[name],
			background_graph=args.background_graph,
			multiplier=args.multiplier,
		)
		for G,_ in graphs:
			data_list.append(nx_to_pyg(G, class_id,
				feature_dim=args.feature_dim,
				feature_noise_p=args.feature_noise_p))

	# save like TU-style: a single tensor file with just Data objects
	torch.save(data_list, os.path.join(args.out_dir, "data_list.pt"))
	torch.save(dict(
		motifs=motif_names,
		samples_per_class=args.samples_per_class,
		background_graph=args.background_graph,
		multiplier=args.multiplier,
		feature_dim=args.feature_dim,
		feature_noise_p=args.feature_noise_p,
		seed=args.seed,
		total_graphs=len(data_list),
	), os.path.join(args.out_dir, "meta.pt"))

	print("saved:", os.path.join(args.out_dir, "data_list.pt"))
	print("classes:", {i:n for i,n in enumerate(motif_names)})
	print("total graphs:", len(data_list))

if __name__ == "__main__":
	main()
