import os
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random

import sys
sys.path.append('.')
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor

from collections.abc import Mapping
from typing import List, Optional, Sequence, Union

import h5py
import pickle

from Utils.alignment import Kabsch_alignment
from Utils.utils import generate_fully_connected



class Dataset_dynamics(Dataset):
    def __init__(self, datafile, datasplit):
        super(Dataset_dynamics, self).__init__()
        self.datasplit = datasplit
        assert datasplit in [
            'id_train',
            'id_val',
            'id_test',
            'ood_test_type',
            'ood_test_size'
        ]
        self.data = pickle.load(open(datafile, 'rb'))[self.datasplit]

    def __getitem__(self, index):
        x = torch.tensor(self.data[index]['atom_types'], dtype=torch.long)

        reactant_pos = torch.tensor(self.data[index]['R_G'], dtype=torch.float32)
        product_pos = torch.tensor(self.data[index]['P_G'], dtype=torch.float32)
        transition_state_pos = torch.tensor(self.data[index]['TS_G'], dtype=torch.float32)
        product_pos = Kabsch_alignment(product_pos, reactant_pos, torch.zeros_like(x))
        transition_state_pos = Kabsch_alignment(transition_state_pos, reactant_pos, torch.zeros_like(x))

        energies = list()
        energies.append(torch.tensor(self.data[index]['R_E'], dtype=torch.float32))
        energies.append(torch.tensor(self.data[index]['P_E'], dtype=torch.float32))
        energies.append(torch.tensor(self.data[index]['TS_E'], dtype=torch.float32))
        
        return Data(x=x, reactant_pos=reactant_pos, product_pos=product_pos, transition_state_pos=transition_state_pos, energies=torch.stack(energies))
                    
    def __len__(self):
        return len(self.data)


def generate_dataloader_dynamics(data_file, batch_size):
    dataloaders = {}
    dataloaders['id_train'] = DataLoader(dataset=Dataset_dynamics(data_file, 'id_train'), batch_size = batch_size, shuffle=True)
    dataloaders['id_val'] = DataLoader(dataset=Dataset_dynamics(data_file, 'id_val'), batch_size = batch_size)
    dataloaders['id_test'] = DataLoader(dataset=Dataset_dynamics(data_file, 'id_test'), batch_size = batch_size)
    dataloaders['ood_test_type'] = DataLoader(dataset=Dataset_dynamics(data_file, 'ood_test_type'), batch_size = batch_size)
    dataloaders['ood_test_size'] = DataLoader(dataset=Dataset_dynamics(data_file, 'ood_test_size'), batch_size = batch_size)
    return dataloaders