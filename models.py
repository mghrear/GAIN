import pandas as pd
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Optional, Tuple

# Basic relational models used by IN and GAIN
class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        return self.layers(m)

# Basic object models used by IN and GAIN
class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)

# IN layer adapted from https://github.com/GageDeZoort/interaction_network_paper
class INLayer(MessagePassing):
    def __init__(self, hidden_size=40, node_feat_in = 3, edge_feat_in = 3, node_feat_out=3, edge_feat_out=3):
        super(INLayer, self).__init__(aggr='add', 
                                                 flow='source_to_target')
        self.R1 = RelationalModel(2*node_feat_in+edge_feat_in, edge_feat_out, hidden_size)
        self.O = ObjectModel(node_feat_in+edge_feat_out, node_feat_out, hidden_size)
        self.E: Tensor = Tensor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return x_tilde, self.E

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c)

# Graph Attention Interaction Network (GAIN) layer
class GAINLayer(MessagePassing):
    def __init__(self, dropout = 0.1, hidden_size=40, node_feat_in=3, edge_feat_in=3,
                 node_feat_out=3, edge_feat_out=3, heads=1, negative_slope=0.2):
        super(GAINLayer, self).__init__(aggr='add',
                                       flow='source_to_target',
                                       node_dim=0)
        
        #Check that desired output edge feature dimension is compatible with number of heads
        assert edge_feat_out % heads == 0, \
            f'edge_feat_out ({edge_feat_out}) must be divisible by heads ({heads})'
        self.heads = heads
        self.head_edge_dim = edge_feat_out // heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # An independent relational model for each head, mapping to head_edge_dim
        self.R1s = nn.ModuleList([
            RelationalModel(2*node_feat_in + edge_feat_in, self.head_edge_dim, hidden_size)
            for _ in range(heads)
        ])
        # A single object model, mapping to node_feat_out
        self.O  = ObjectModel(node_feat_in + edge_feat_out, node_feat_out, hidden_size)

        # One attention vector per head: [1, heads, head_edge_dim]
        self.a = nn.Parameter(torch.empty(1, heads, self.head_edge_dim))
        nn.init.xavier_uniform_(self.a)

        # Edge features after attention-weighted message passing; will be set in message() and used in forward()
        self.E: Tensor = Tensor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return x_tilde, self.E

    def message(self, x_i, x_j, edge_attr, index, ptr: OptTensor, size_i):
        # x_i --> incoming (target), x_j --> outgoing (source)
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)            # [E, 2*node_feat_in + edge_feat_in]

        # Each head independently maps m1 → head_edge_dim
        raw_mh = torch.stack([r(m1) for r in self.R1s], dim=1)  # [E, heads, head_edge_dim]

        # Attention score: dot with learned vector, LeakyReLU, softmax over neighbours
        e     = (raw_mh * self.a).sum(dim=-1)                   # [E, heads]
        e     = F.leaky_relu(e, self.negative_slope)
        alpha = softmax(e, index, ptr, size_i)                  # [E, heads], sums to 1 per node
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weight messages by attention: [E, heads, head_edge_dim]
        weighted_mh = raw_mh * alpha.unsqueeze(-1)
        self.E = weighted_mh.view(-1, self.heads * self.head_edge_dim)  # [E, edge_feat_out]
        return weighted_mh

    def update(self, aggr_out, x):
        # aggr_out: [N, heads, head_edge_dim] → flatten to [N, edge_feat_out]
        aggr_flat = aggr_out.view(aggr_out.shape[0], self.heads * self.head_edge_dim)
        c = torch.cat([x, aggr_flat], dim=1)
        return self.O(c)


# IN model used for tagger studies     
class MyIN(nn.Module):
    def __init__(self, hidden_size):
        super(MyIN, self).__init__()

        self.IN1 = INLayer(hidden_size=hidden_size, node_feat_in = 3, edge_feat_in = 3, node_feat_out=64, edge_feat_out=64)
        self.IN2 = INLayer(hidden_size=hidden_size, node_feat_in = 64, edge_feat_in = 64, node_feat_out=128, edge_feat_out=128)
        self.IN3 = INLayer(hidden_size=hidden_size, node_feat_in = 128, edge_feat_in = 128, node_feat_out=128, edge_feat_out=128)
        self.R2 = RelationalModel(2*128+128, 1, hidden_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):

        x, edge_attr = self.IN1(x, edge_index, edge_attr)
        x, edge_attr = self.IN2(x, edge_index, edge_attr)
        x, edge_attr = self.IN3(x, edge_index, edge_attr)

        edge_final = torch.cat([x[edge_index[1]],
                        x[edge_index[0]],
                        edge_attr], dim=1)

        return torch.sigmoid(self.R2(edge_final))
        
# IN model used for recoil (signal) studies
class MyIN_small(nn.Module):
    def __init__(self, hidden_size):
        super(MyIN_small, self).__init__()

        self.IN1 = INLayer(hidden_size=hidden_size, node_feat_in = 3, edge_feat_in = 3, node_feat_out=12, edge_feat_out=12)
        self.IN2 = INLayer(hidden_size=hidden_size, node_feat_in = 12, edge_feat_in = 12, node_feat_out=24, edge_feat_out=24)
        self.IN3 = INLayer(hidden_size=hidden_size, node_feat_in = 24, edge_feat_in = 24, node_feat_out=24, edge_feat_out=24)
        self.R2 = RelationalModel(2*24+24, 1, hidden_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):

        x, edge_attr = self.IN1(x, edge_index, edge_attr)
        x, edge_attr = self.IN2(x, edge_index, edge_attr)
        x, edge_attr = self.IN3(x, edge_index, edge_attr)

        edge_final = torch.cat([x[edge_index[1]],
                        x[edge_index[0]],
                        edge_attr], dim=1)

        return torch.sigmoid(self.R2(edge_final))

# 5 message-passing layer IN model used in tagger study    
class MyIN_5L(nn.Module):
    def __init__(self, hidden_size):
        super(MyIN_5L, self).__init__()

        self.IN1 = INLayer(hidden_size=hidden_size, node_feat_in = 3, edge_feat_in = 3, node_feat_out=64, edge_feat_out=64)
        self.IN2 = INLayer(hidden_size=hidden_size, node_feat_in = 64, edge_feat_in = 64, node_feat_out=128, edge_feat_out=128)
        self.IN3 = INLayer(hidden_size=hidden_size, node_feat_in = 128, edge_feat_in = 128, node_feat_out=128, edge_feat_out=128)
        self.IN4 = INLayer(hidden_size=hidden_size, node_feat_in = 128, edge_feat_in = 128, node_feat_out=128, edge_feat_out=128)
        self.IN5 = INLayer(hidden_size=hidden_size, node_feat_in = 128, edge_feat_in = 128, node_feat_out=128, edge_feat_out=128)
        self.R2 = RelationalModel(2*128+128, 1, hidden_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):

        x, edge_attr = self.IN1(x, edge_index, edge_attr)
        x, edge_attr = self.IN2(x, edge_index, edge_attr)
        x, edge_attr = self.IN3(x, edge_index, edge_attr)
        x, edge_attr = self.IN4(x, edge_index, edge_attr)
        x, edge_attr = self.IN5(x, edge_index, edge_attr)

        edge_final = torch.cat([x[edge_index[1]],
                        x[edge_index[0]],
                        edge_attr], dim=1)

        return torch.sigmoid(self.R2(edge_final))

# GAIN model used in recoil (signal) and tagger studies
# for recoil (signal): hidden_size=40, heads = 3, base_dim=4
# for tagger: hidden_size=50, heads = 8, base_dim=8
class MyGAIN(nn.Module):
    #3-layer Graph Attention Interaction Network.

    def __init__(self, hidden_size, heads=2, base_dim=4):
        super(MyGAIN, self).__init__()
        d1 = heads * base_dim
        d2 = heads * base_dim * 2
        self.IN1 = GAINLayer(hidden_size=hidden_size, node_feat_in=3,  edge_feat_in=3,  node_feat_out=d1, edge_feat_out=d1, heads=heads)
        self.IN2 = GAINLayer(hidden_size=hidden_size, node_feat_in=d1, edge_feat_in=d1, node_feat_out=d2, edge_feat_out=d2, heads=heads)
        self.IN3 = GAINLayer(hidden_size=hidden_size, node_feat_in=d2, edge_feat_in=d2, node_feat_out=d2, edge_feat_out=d2, heads=heads)
        self.R2  = RelationalModel(2*d2 + d2, 1, hidden_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x, edge_attr = self.IN1(x, edge_index, edge_attr)
        x, edge_attr = self.IN2(x, edge_index, edge_attr)
        x, edge_attr = self.IN3(x, edge_index, edge_attr)

        edge_final = torch.cat([x[edge_index[1]],
                                x[edge_index[0]],
                                edge_attr], dim=1)
        return torch.sigmoid(self.R2(edge_final))
        
# 5 message-passing layer GAIN model used in tagger study (hidden_size=50, heads = 8, base_dim=8) 
class MyGAIN_5L(nn.Module):
    #5-layer Graph Attention Interaction Network.

    def __init__(self, hidden_size, heads=2, base_dim=4):
        super(MyGAIN_5L, self).__init__()
        d1 = heads * base_dim
        d2 = heads * base_dim * 2
        self.IN1 = GAINLayer(hidden_size=hidden_size, node_feat_in=3,  edge_feat_in=3,  node_feat_out=d1, edge_feat_out=d1, heads=heads)
        self.IN2 = GAINLayer(hidden_size=hidden_size, node_feat_in=d1, edge_feat_in=d1, node_feat_out=d2, edge_feat_out=d2, heads=heads)
        self.IN3 = GAINLayer(hidden_size=hidden_size, node_feat_in=d2, edge_feat_in=d2, node_feat_out=d2, edge_feat_out=d2, heads=heads)
        self.IN4 = GAINLayer(hidden_size=hidden_size, node_feat_in=d2, edge_feat_in=d2, node_feat_out=d2, edge_feat_out=d2, heads=heads)
        self.IN5 = GAINLayer(hidden_size=hidden_size, node_feat_in=d2, edge_feat_in=d2, node_feat_out=d2, edge_feat_out=d2, heads=heads)
        self.R2  = RelationalModel(2*d2 + d2, 1, hidden_size)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x, edge_attr = self.IN1(x, edge_index, edge_attr)
        x, edge_attr = self.IN2(x, edge_index, edge_attr)
        x, edge_attr = self.IN3(x, edge_index, edge_attr)
        x, edge_attr = self.IN4(x, edge_index, edge_attr)
        x, edge_attr = self.IN5(x, edge_index, edge_attr)

        edge_final = torch.cat([x[edge_index[1]],
                                x[edge_index[0]],
                                edge_attr], dim=1)
        return torch.sigmoid(self.R2(edge_final))
    

