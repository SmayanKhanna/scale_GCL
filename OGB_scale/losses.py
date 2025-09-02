
#losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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

# PyGCL implementation of InfoNCE

# def _similarity(h1: torch.Tensor, h2: torch.Tensor):
#     h1 = F.normalize(h1)
#     h2 = F.normalize(h2)
#     return h1 @ h2.t()


# class InfoNCE(nn.Module):
#     def __init__(self, tau):
#         super(InfoNCE, self).__init__()
#         self.tau = tau

#     def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
#         sim = _similarity(anchor, sample) / self.tau
#         exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
#         log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
#         loss = log_prob * pos_mask
#         loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
#         return -loss.mean()

