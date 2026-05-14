import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.autograd import grad
import math
import time

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
        

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
        zero_init=False
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None
        self.zero_init = zero_init

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        if self.zero_init:
            nn.init.zeros_(self.update_net[2].weight)
        else:
            nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v

        
class EquivariantScalarHead(torch.nn.Module):
    def __init__(self, 
                hidden_dim=128,
                aggr='sum', 
                activation="silu",
                level='graph',
                graph_dec=False,
                mean=None,
                std=None,
                zero_init=False):
        super(EquivariantScalarHead, self).__init__()
        self.level = level
        self.aggr = aggr
        
        mean = torch.scalar_tensor(0 if mean is None else mean)
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1 if std is None else std)
        self.register_buffer("std", std)
        
        self.graph_dec = None
        self.zero_init = zero_init
        if level == 'node':
            self.node_dec = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_dim,
                    hidden_dim // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_dim // 2, 1, activation=activation),
            ]
        ) 
        if level == 'graph':
            self.node_dec = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_dim,
                    hidden_dim // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_dim // 2, 1 if not graph_dec else hidden_dim, activation=activation, zero_init=(self.zero_init and (not graph_dec))),
            ]
        )      
            if graph_dec:
                self.graph_dec = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       act_class_mapping[activation](),
                                       nn.Linear(hidden_dim, 1))
                                   
    def reset_parameters(self):
        for layer in self.node_dec:
            layer.reset_parameters()
        if self.graph_dec is not None:
            for layer in self.graph_dec:
                layer.reset_parameters()
            nn.init.zeros_(self.graph_dec[2].bias)   
            if self.zero_init:
                nn.init.zeros_(self.graph_dec[2].weight)                                                              

    def forward(self, x, v, batch, z=None, pos=None):
        for layer in self.node_dec:
            x, v = layer(x, v)
        x = x + v.sum() * 0
        xg = scatter(x, batch, dim=0, reduce=self.aggr)
        if self.graph_dec is not None:
            xg = self.graph_dec(xg)
        xg = xg * self.std + self.mean
        return xg
