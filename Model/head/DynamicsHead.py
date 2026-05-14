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
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
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


class DirectDynamicHead(nn.Module):
    def __init__(
        self,
        hidden_dim = 128,
        activation="silu"
    ):
        super(DirectDynamicHead, self).__init__()
        # self.node_dec1 = nn.ModuleList(
        # [
        #     GatedEquivariantBlock(
        #         hidden_dim,
        #         hidden_dim,
        #         activation=activation,
        #         scalar_activation=True,
        #     ),
        #     GatedEquivariantBlock(hidden_dim, hidden_dim, activation=activation),
        # ])
        self.node_dec2 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                    act_class_mapping[activation](),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.output_dec = nn.ModuleList(
        [
            GatedEquivariantBlock(
                hidden_dim,
                hidden_dim,
                activation=activation,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(hidden_dim, 1, activation=activation),
        ])
        self.signed_func = nn.Sequential(nn.Linear(hidden_dim, 1, bias=False),
                                    act_class_mapping['tanh']())
                                   
    def reset_parameters(self):
        for layer in self.node_dec1:
            layer.reset_parameters()
        for layer in self.node_dec2:
            layer.reset_parameters()
        for layer in self.output_dec:
            layer.reset_parameters()                                                         

    def forward(self, rep0, rept, repT):
        x0, v0 = rep0
        xt, vt = rept
        xT, vT = repT

        # for layer in self.node_dec1:
        #     x0, v0 = layer(x0, v0)
        #     xt, vt = layer(xt, vt)
        #     xT, vT = layer(xT, vT)
        x0 = x0 + v0.sum() * 0
        xT = xT + vT.sum() * 0
        temp1 = self.node_dec2(torch.concat([x0, xt], dim=-1))
        temp2 = self.node_dec2(torch.concat([xT, xt], dim=-1))
        x = temp1 * temp2
        sign = self.signed_func(temp1 - temp2)
        for layer in self.output_dec:
            x, vt = layer(x, vt)
        return vt.squeeze(-1) * sign
