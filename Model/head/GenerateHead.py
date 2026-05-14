import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch.autograd import grad
import math
import time

from .Scalar import ScalarHead
from .Vector import VectorHead
from .EquivariantScalar import EquivariantScalarHead
from .EquivariantVector import EquivariantVectorHead

def generate_head(config):
    if config.name == 'Scalar':
        return ScalarHead(**config.parameters) if config.parameters is not None else ScalarHead()
        
    elif config.name == 'Vector':
        return VectorHead(**config.parameters) if config.parameters is not None else VectorHead()
                      
    elif config.name == 'EquivariantScalar':
        return EquivariantScalarHead(**config.parameters) if config.parameters is not None else EquivariantScalarHead()
    
    elif config.name == 'EquivariantVector':
        return EquivariantVectorHead(**config.parameters) if config.parameters is not None else EquivariantVectorHead()
    
    else:
        print('Head', config.name, 'Not implemented.')
        return None
