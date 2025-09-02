# initialize the necessary imports for this code
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, GINConv
import torch
import numpy as np



# ───────────────────── simple GIN encoder ─────────────────────

class GINBlock(nn.Module):
	def __init__(self, in_features, hidden_dim):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(in_features, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, hidden_dim),
		)
		self.gin = GINConv(self.mlp)
		self.bn  = nn.BatchNorm1d(hidden_dim)
	def forward(self, x, edge_index):
		x = self.gin(x, edge_index)
		x = F.relu(x)
		x = self.bn(x)
		return x

class Encoder(nn.Module):
	def __init__(self, num_features, dim, num_layers):
		super().__init__()
		self.blocks = nn.ModuleList()
		for i in range(num_layers):
			in_dim = num_features if i == 0 else dim
			self.blocks.append(GINBlock(in_dim, dim))
	def forward(self, x, edge_index, batch):
		if x is None:
			x = torch.ones((batch.shape[0], 1), device=batch.device)
		xs = []
		for block in self.blocks:
			x = block(x, edge_index)
			xs.append(x)
		pooled = [global_add_pool(h, batch) for h in xs]
		return torch.cat(pooled, dim=1), x
	
	def get_embeddings(self, loader, device):
		self.eval()
		ret, y = [], []
		with torch.no_grad():
			for data in loader:
				data = data.to(device)
				x, edge_index, batch = data.x, data.edge_index, data.batch
				if x is None:
					x = torch.ones((data.num_nodes, 1), device=device)
				g_z, _ = self.forward(x, edge_index, batch)
				ret.append(g_z.cpu().numpy())
				y.append(data.y.view(-1).cpu().numpy())
		return np.concatenate(ret, 0), np.concatenate(y, 0)