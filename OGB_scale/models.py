# models.py
import numpy as np
# PyTorch Geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.loader import DataLoader 
from torch_geometric.datasets import TUDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# GNN Layers
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, GINEConv, HeteroConv,
    global_mean_pool, global_max_pool, global_add_pool,
    DMoNPooling
)

# Graph Utilities
from torch_geometric.utils import (
    k_hop_subgraph, subgraph, from_networkx,
    to_networkx, to_dense_adj, to_dense_batch,
    degree
)

# This is the architecture that https://github.com/sunfanyunn/InfoGraph/blob/master/unsupervised/gin.py uses (and 99% of graph-classification-based GCL methods)
class GINEBlock(nn.Module):
    """
    GINE block: LazyLinear→ReLU→Linear inside GINEConv, then BN+ReLU.
    """
    def __init__(self, in_features, hidden_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            # nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gine = GINEConv(self.mlp, edge_dim = edge_dim)
        self.bn  = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr = None):

        x = self.gine(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn(x)
        # return F.relu(x)
        return x

#encoder
class GINE_Encoder(nn.Module):
    # GINE ENCODER
    def __init__(self, num_features, dim, num_layers, edge_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_features if i==0 else dim
            self.blocks.append(GINEBlock(in_dim, dim, edge_dim))

    def forward(self, x, edge_index, batch, edge_attr = None):

        xs = []
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)       # GINE → BN → ReLU
            xs.append(x)

        pooled = [global_add_pool(h, batch) for h in xs]
        return torch.cat(pooled, dim=1), x

    def get_embeddings(self, loader):
        #not necessarily needed but ill keep it why not
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)

                if edge_attr is not None:
                    #cuz of some data mismatching error
                    edge_attr = edge_attr.to(torch.float)
                    
                x, _ = self.forward(x, edge_index, batch, edge_attr)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


