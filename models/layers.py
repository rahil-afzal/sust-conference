
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class AdvancedEdgeConvLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        msg = torch.cat([x[row], x[col], edge_attr], dim=-1)
        msg = self.mlp(msg)
        return scatter(msg, row, dim=0, dim_size=x.size(0), reduce="sum")
