
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        w = torch.softmax(self.att(x), dim=0)
        return scatter(x * w, batch, dim=0, reduce="sum")
