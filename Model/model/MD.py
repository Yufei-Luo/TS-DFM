import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch.autograd import grad
import math
import time
from ..head import *

class MDNet(torch.nn.Module):
    def __init__(self, backbone, head, reference_energy=None):
        super(MDNet, self).__init__()
        self.backbone = backbone
        self.head = head
        if reference_energy is not None:
            self.reference_energy = True
            self.reference_energy_list = torch.nn.Parameter(torch.zeros(100, dtype=torch.float32), requires_grad=False)
            for key, val in reference_energy.items():
                self.reference_energy_list[key] = val
        else:
            self.reference_energy = False              

    def get_energy(self, x, pos, edge_index, edge_attr, batch):
        temp = self.backbone(x, pos, edge_index, edge_attr, batch)   
        if isinstance(self.head, EquivariantScalarHead):
            xl, vl = temp
            e = self.head(xl, vl, batch)
        else:
            xl = temp[0] if isinstance(temp, tuple) else temp
            e = self.head(xl, batch)
        e = e.squeeze(-1)
        if self.reference_energy:
            ref_energy = self.reference_energy_list[x]
            ref_energy = scatter_add(ref_energy, batch)
            e += ref_energy
        return e
    
    def get_force(self, pred_energy, pos):
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(pred_energy)]
        dy = grad(
                    [pred_energy],
                    [pos],
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
            
        pred_force = (-dy).view(-1, 3)
        return pred_force
    
    def get_hessian(self, pred_force, pos):
        hessian = torch.vmap(
            lambda vec: grad(
                -pred_force.flatten(),
                pos,
                grad_outputs=vec,
                create_graph=True,
                retain_graph=True
            )[0],
        )(torch.eye(pred_force.numel(), device=pred_force.device))
        hessian = hessian.reshape(*pred_force.shape, *pos.shape)
        return hessian
  
    def get_energy_and_force(self, x, pos, edge_index, edge_attr, batch):
        try:
            pos.requires_grad = True
        except:
            pass
        pred_energy = self.get_energy(x, pos, edge_index, edge_attr, batch)
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(pred_energy)]
        dy = grad(
                    [pred_energy],
                    [pos],
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
            
        pred_force = (-dy).view(-1, 3)
        return pred_energy, pred_force

    def langevin_dynamics(self, x, pos, edge_index, edge_attr, batch, steps, alpha = 0.5, add_noise = True):
        pos_steps = []
        new_pos = pos
        for t in range(1, steps + 1):
            new_pos = new_pos.detach()
            _, force = self.get_energy_and_force(x, new_pos, edge_index, edge_attr, batch)
            new_pos = new_pos + alpha * force
            if add_noise:
                new_pos = new_pos + torch.randn_like(new_pos) * math.sqrt(2 * alpha)
            pos_steps.append(new_pos.clone())
        return pos_steps
             
    def forward(self, x, pos, edge_index, edge_attr, batch):
        energy, force = self.get_energy_and_force(x, pos, edge_index, edge_attr, batch)
        return energy, force
