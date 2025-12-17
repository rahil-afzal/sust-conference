
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from .layers import AdvancedEdgeConvLayer
from .pooling import AttentionPooling

class AdvancedGNNModelWithEdge(nn.Module):
    def __init__(self, node_fea_len, edge_fea_len, hidden_fea_len,
                 num_layers, edge_hidden_dim, h_fea_len,
                 n_h, dropout_rate, use_attention_pooling=False, **kwargs):
        super().__init__()

        self.embed = nn.Linear(node_fea_len, hidden_fea_len)
        self.convs = nn.ModuleList([
            AdvancedEdgeConvLayer(hidden_fea_len, edge_fea_len, edge_hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.att_pool = AttentionPooling(hidden_fea_len)

        pool_dim = hidden_fea_len * (3 if use_attention_pooling else 2)
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, h_fea_len),
            nn.ReLU(),
            nn.Linear(h_fea_len, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        if self.use_attention_pooling:
            x_att = self.att_pool(x, batch)
            x = nn.functional.concat([x_mean, x_max, x_att], dim=1)
        else:
            x = nn.functional.concat([x_mean, x_max], dim=1)

        return self.mlp(x)
