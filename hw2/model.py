import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GATv2Conv, GCNConv


class GCN_ori(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class GCN_geo(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GCNConv(in_size, hid_size))
        self.layers.append(GCNConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        edge_index = torch.stack(g.edges())
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, edge_index)
        return h


class GCN_GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GATv2Conv(in_size, hid_size, heads=8, dropout=dropout))
        self.layers.append(
            GATv2Conv(hid_size * 8, out_size, heads=1, concat=False, dropout=dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        edge_index = torch.stack(g.edges())
        h = features
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(h, edge_index)
            if i != len(self.layers) - 1:
                h = F.relu(h)

        return h


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.crd = CRD(num_features, hidden, dropout)
        self.cls = CLS(hidden, num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    # def forward(self, data):
    #     x, edge_index = data.x, data.edge_index
    #     x = self.crd(x, edge_index, data.train_mask)
    #     x = self.cls(x, edge_index, data.train_mask)
    #     return x
    def forward(self, data, features):
        h = features
        edge_index = torch.stack(data.edges())
        train_mask = data.ndata["train_mask"]
        h = self.crd(h, edge_index, train_mask)
        h = self.cls(h, edge_index, train_mask)
        return h


class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
