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

class VectorHead(torch.nn.Module):
    def __init__(self, 
                hidden_dim=128,
                aggr='sum', 
                activation="silu",
                level='graph',
                graph_dec=False,
                mean=None,
                std=None,
                output_dim=3):
        super(VectorHead, self).__init__()
        self.level = level
        self.aggr = aggr
        
        mean = torch.scalar_tensor(0 if mean is None else mean)
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1 if std is None else std)
        self.register_buffer("std", std)
        
        self.graph_dec = None
        if level == 'node':   
            self.node_dec = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      act_class_mapping[activation](),
                                      nn.Linear(hidden_dim, output_dim)) 
        if level == 'graph':
            self.node_dec = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      act_class_mapping[activation](),
                                      nn.Linear(hidden_dim, output_dim if not graph_dec else hidden_dim))         
            if graph_dec:
                self.graph_dec = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       act_class_mapping[activation](),
                                       nn.Linear(hidden_dim, output_dim))
                                   
    def reset_parameters(self):
        for layer in self.node_dec:
            layer.reset_parameters()
        if self.graph_dec is not None:
            for layer in self.graph_dec:
                layer.reset_parameters()                                                               

    def forward(self, x, batch): 
        x = self.node_dec(x)
        if self.level == 'graph':
            xg = scatter(x, batch, dim=0, reduce=self.aggr)
        if self.graph_dec is not None:
            xg = self.graph_dec(xg)
        xg = xg * self.std + self.mean
        return xg
