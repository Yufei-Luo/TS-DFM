from torch_scatter import scatter
import torch
from torch import nn
from .TorchMD import *
from .PaiNN import *
from .EGNN import *

def generate_backbone(config):
    
    if config.name == 'TorchMD':
        return TorchMD_ET(**config.parameters) if config.parameters is not None else TorchMD_ET()
        
    elif config.name == 'PaiNN':
        return PaiNN(**config.parameters) if config.parameters is not None else PaiNN()
        
    elif config.name == 'EGNN':
        return EGNN(**config.parameters) if config.parameters is not None else EGNN()
    
    else:
        print('Backbone', config.name, 'Not implemented.')
        return None
