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

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}


def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def generator(formula, rxn, grp):
    """ Iterates through a h5 group """

    energies = grp["wB97x_6-31G(d).energy"]
    forces = grp["wB97x_6-31G(d).forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["positions"]
    molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)

    for energy, force, position in zip(energies, forces, positions):
        d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": position,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }

        yield d


def transform_to_pyg_data_potential(mol):
    x = torch.tensor(mol['atomic_numbers'], dtype=torch.long)
    pos = torch.tensor(mol['positions'], dtype=torch.float32)
    energy = torch.tensor(mol['wB97x_6-31G(d).energy'], dtype=torch.float32)
    atomization_energy = torch.tensor(mol['wB97x_6-31G(d).atomization_energy'], dtype=torch.float32)
    force = torch.tensor(mol['wB97x_6-31G(d).forces'], dtype=torch.float32)
    return Data(x=x, pos=pos, energy=energy, atomization_energy=atomization_energy, force=force)

def get_dynamics_data(formula, rxn, data):
    reactant = next(generator(formula, rxn, data[formula][rxn]["reactant"]))
    product = next(generator(formula, rxn, data[formula][rxn]["product"]))
    transition_state = next(generator(formula, rxn, data[formula][rxn]["transition_state"]))
    x = torch.tensor(reactant['atomic_numbers'], dtype=torch.long)

    reactant_pos = torch.tensor(reactant['positions'], dtype=torch.float32)
    product_pos = torch.tensor(product['positions'], dtype=torch.float32)
    transition_state_pos = torch.tensor(transition_state['positions'], dtype=torch.float32)
    product_pos = Kabsch_alignment(product_pos, reactant_pos, torch.zeros_like(x))
    transition_state_pos = Kabsch_alignment(transition_state_pos, reactant_pos, torch.zeros_like(x))

    energies = list()
    energies.append(torch.tensor(reactant['wB97x_6-31G(d).energy'], dtype=torch.float32))
    energies.append(torch.tensor(product['wB97x_6-31G(d).energy'], dtype=torch.float32))
    energies.append(torch.tensor(transition_state['wB97x_6-31G(d).energy'], dtype=torch.float32))
    
    return Data(x=x, reactant_pos=reactant_pos, product_pos=product_pos, transition_state_pos=transition_state_pos, energies=torch.stack(energies))

def get_dynamics_data_neb(formula, rxn, data):
    reactant = next(generator(formula, rxn, data[formula][rxn]["reactant"]))
    product = next(generator(formula, rxn, data[formula][rxn]["product"]))
    x = torch.tensor(reactant['atomic_numbers'], dtype=torch.long)

    states = list()
    states.append(torch.tensor(reactant['positions'], dtype=torch.float32))
    energies = list()
    energies.append(torch.tensor(reactant['wB97x_6-31G(d).energy'], dtype=torch.float32))
    forces = list()
    forces.append(torch.tensor(reactant['wB97x_6-31G(d).forces'], dtype=torch.float32))

    transition_states = [molecule for molecule in generator(formula, rxn, data[formula][rxn])]
    transition_states = transition_states[-8:]
    for mol in transition_states:
        temp = torch.tensor(mol['positions'], dtype=torch.float32)
        # temp_aligned = Kabsch_alignment(temp, states[0], batch)
        # states.append(temp_aligned)
        states.append(temp)
        energies.append(torch.tensor(mol['wB97x_6-31G(d).energy'], dtype=torch.float32))
        forces.append(torch.tensor(mol['wB97x_6-31G(d).forces'], dtype=torch.float32))

    temp = torch.tensor(product['positions'], dtype=torch.float32)
    # temp_aligned = Kabsch_alignment(temp, states[0], batch)
    # states.append(temp_aligned)
    states.append(temp)
    energies.append(torch.tensor(product['wB97x_6-31G(d).energy'], dtype=torch.float32))
    forces.append(torch.tensor(product['wB97x_6-31G(d).forces'], dtype=torch.float32))
    
    return (x, states, energies, forces)

def get_dynamics_data_neb_2(formula, rxn, data, neg_path_num):
    reactant = next(generator(formula, rxn, data[formula][rxn]["reactant"]))
    product = next(generator(formula, rxn, data[formula][rxn]["product"]))
    x = torch.tensor(reactant['atomic_numbers'], dtype=torch.long)

    pos_ref = torch.tensor(reactant['positions'], dtype=torch.float32)
    batch_ref = torch.zeros(pos_ref.shape[0], dtype=torch.long)

    neb_states = list()
    neb_states.append(torch.tensor(reactant['positions'], dtype=torch.float32))
    neb_energies = list()
    neb_energies.append(torch.tensor(reactant['wB97x_6-31G(d).energy'], dtype=torch.float32))

    transition_states = [molecule for molecule in generator(formula, rxn, data[formula][rxn])]
    neb_path_states = transition_states[-8:]
    for mol in neb_path_states:
        temp = torch.tensor(mol['positions'], dtype=torch.float32)
        temp_aligned = Kabsch_alignment(temp, pos_ref, batch_ref)
        neb_states.append(temp_aligned)
        # neb_states.append(temp)
        neb_energies.append(torch.tensor(mol['wB97x_6-31G(d).energy'], dtype=torch.float32))

    temp = torch.tensor(product['positions'], dtype=torch.float32)
    temp_aligned = Kabsch_alignment(temp, pos_ref, batch_ref)
    neb_states.append(temp_aligned)
    # neb_states.append(temp)
    neb_energies.append(torch.tensor(product['wB97x_6-31G(d).energy'], dtype=torch.float32))
    
    neg_states = []
    neg_energies = []
    neg_forces = []
    num_paths = (len(transition_states) - 2) // 8 - 1
    path = random.sample(list(range(num_paths)), neg_path_num)
    for path_num in path:
        beg_pos = 1 if path_num == 0 else 10 + (path_num - 1) * 8
        temp_path_states = transition_states[beg_pos:beg_pos+8]
        for mol in temp_path_states:
            temp = torch.tensor(mol['positions'], dtype=torch.float32)
            temp_aligned = Kabsch_alignment(temp, pos_ref, batch_ref)
            neg_states.append(temp_aligned)
            neg_states.append(temp)
            neg_energies.append(torch.tensor(mol['wB97x_6-31G(d).energy'], dtype=torch.float32))

    return x, neb_states + neg_states, neb_energies + neg_energies

class Dataset_dynamics(IterableDataset):
    def __init__(self, hdf5_file, datasplit):
        super(Dataset_dynamics, self).__init__()
        self.hdf5_file = hdf5_file
        self.datasplit = datasplit
        assert datasplit in [
            "train",
            "valid",
            "test",
        ]
        with open('Data/reactions_'+self.datasplit+'.pickle', 'rb') as f:
            self.datalist = pickle.load(f)

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            data = f['data']
            random.shuffle(self.datalist)
            for formula, rxn in self.datalist:
                yield get_dynamics_data(formula, rxn, data)
                    
    def __len__(self):
        pass
        

class Dataset_potential(IterableDataset):
    def __init__(self, hdf5_file, datasplit):
        super(Dataset_potential, self).__init__()
        self.hdf5_file = hdf5_file
        self.datasplit = datasplit
        assert datasplit in [
            "train",
            "valid",
            "test",
        ]
        with open('Data/reactions_'+self.datasplit+'.pickle', 'rb') as f:
            self.datalist = pickle.load(f)
    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            data = f['data']
            if self.datasplit == 'train' or self.datasplit == 'valid':
                random.shuffle(self.datalist)

            for formula, rxn in self.datalist:
                states = [molecule for molecule in generator(formula, rxn, data[formula][rxn])]
                for state in states:
                    yield transform_to_pyg_data_potential(state)

class Dataloader_dynamics_neb(IterableDataset):
    def __init__(self, hdf5_file, datasplit):
        super(Dataloader_dynamics_neb, self).__init__()
        self.hdf5_file = hdf5_file
        self.datasplit = datasplit
        assert datasplit in [
            "train",
            "valid",
            "test",
        ]
        with open('Data/reactions_'+self.datasplit+'.pickle', 'rb') as f:
            self.datalist = pickle.load(f)

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            data = f['data']
            if self.datasplit == 'train' or self.datasplit == 'valid':
                random.shuffle(self.datalist)
            
            for formula, rxn in self.datalist:
                x, states, energies, forces = get_dynamics_data_neb(formula, rxn, data)
                x_list = []
                batch = []
                for j in range(len(states)):
                    x_list.append(deepcopy(x))
                    batch.append(torch.ones_like(x) * j)
                x = torch.concat(x_list, dim=0)
                states = torch.concat(states, dim=0)
                energies = torch.tensor(energies)
                forces = torch.concat(forces, dim=0)
                batch = torch.concat(batch, dim=0)
                src, dst = generate_fully_connected(batch)
                yield (x, states, energies, forces, batch, src, dst)
                    
    def __len__(self):
        pass

class Dataloader_dynamics_neb_2(IterableDataset):
    def __init__(self, hdf5_file, datasplit, negative_path_num):
        super(Dataloader_dynamics_neb_2, self).__init__()
        self.hdf5_file = hdf5_file
        self.datasplit = datasplit
        self.negative_path_num = negative_path_num
        assert datasplit in [
            "train",
            "valid",
            "test",
        ]
        with open('Data/reactions_'+self.datasplit+'.pickle', 'rb') as f:
            self.datalist = pickle.load(f)

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            data = f['data']
            if self.datasplit == 'train' or self.datasplit == 'valid':
                random.shuffle(self.datalist)
            
            for formula, rxn in self.datalist:
                x, states, energies = get_dynamics_data_neb_2(formula, rxn, data, self.negative_path_num)
                x_list = []
                batch = []
                for j in range(len(states)):
                    x_list.append(deepcopy(x))
                    batch.append(torch.ones_like(x) * j)
                x = torch.concat(x_list, dim=0)
                states = torch.concat(states, dim=0)
                energies = torch.tensor(energies)
                batch = torch.concat(batch, dim=0)
                yield (x, states, energies, batch)
                    
    def __len__(self):
        pass

class Dataset_dynamics_neb_3(IterableDataset):
    def __init__(self, hdf5_file, datasplit):
        super(Dataset_dynamics_neb_3, self).__init__()
        self.hdf5_file = hdf5_file
        self.datasplit = datasplit
        assert datasplit in [
            "train",
            "valid",
            "test",
        ]
        with open('Data/reactions_'+self.datasplit+'.pickle', 'rb') as f:
            self.datalist = pickle.load(f)

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            data = f['data']
            if self.datasplit == 'train' or self.datasplit == 'valid':
                random.shuffle(self.datalist)
            
            for formula, rxn in self.datalist:
                x, states, energies = get_dynamics_data_neb_2(formula, rxn, data, 0)
                states_temp = torch.concat(states, dim=0)
                size = x.shape[0]
                pos_diff = torch.norm(states_temp[size:] - states_temp[:-size], p=2, dim=-1)
                batch_temp = torch.arange(9).repeat_interleave(size)
                pos_diff_sum = scatter_add(pos_diff, batch_temp)
                timestamps = torch.cumsum(pos_diff_sum, dim=0) / torch.sum(pos_diff_sum)
                timestamps[-1] += 1e-10
                timestamps = torch.concat([torch.tensor([0.0]), timestamps])
                time_rand = random.random()
                ind = torch.argmin(torch.abs(timestamps - time_rand))
                if time_rand > timestamps[ind]:
                    time_beg = timestamps[ind]
                    time_end = timestamps[ind + 1]
                    pos_interp = (time_end - time_rand) / (time_end - time_beg) * states[ind] + (time_rand - time_beg) / (time_end - time_beg) * states[ind+1]
                elif time_rand == timestamps[ind]:
                    pos_interp = states[ind]
                else:
                    time_beg = timestamps[ind - 1]
                    time_end = timestamps[ind]
                    pos_interp = (time_end - time_rand) / (time_end - time_beg) * states[ind-1] + (time_rand - time_beg) / (time_end - time_beg) * states[ind]
                yield Data(pos=pos_interp, x=x, time=torch.tensor([time_rand]), pos_0=states[0], pos_T=states[-1])
                    
    def __len__(self):
        pass


def generate_dataloader_dynamics(hdf5_file, batch_size):
    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset=Dataset_dynamics(hdf5_file, 'train'), batch_size = batch_size)
    dataloaders['val'] = DataLoader(dataset=Dataset_dynamics(hdf5_file, 'valid'), batch_size = batch_size)
    dataloaders['test'] = DataLoader(dataset=Dataset_dynamics(hdf5_file, 'test'), batch_size = batch_size)
    return dataloaders


def generate_dataloader_potential(hdf5_file, size):
    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset = Dataset_potential(hdf5_file, 'train'), batch_size = size)
    dataloaders['val'] = DataLoader(dataset = Dataset_potential(hdf5_file, 'valid'), batch_size = size)
    dataloaders['test'] = DataLoader(dataset = Dataset_potential(hdf5_file, 'test'), batch_size = size)
    return dataloaders

def generate_dataloader_dynamics_neb(hdf5_file):
    dataloaders = {}
    dataloaders['train'] = Dataloader_dynamics_neb(hdf5_file, 'train')
    dataloaders['val'] = Dataloader_dynamics_neb(hdf5_file, 'valid')
    dataloaders['test'] = Dataloader_dynamics_neb(hdf5_file, 'test')
    return dataloaders

def generate_dataloader_dynamics_neb_2(hdf5_file, negative_path_num):
    dataloaders = {}
    dataloaders['train'] = Dataloader_dynamics_neb_2(hdf5_file, 'train', negative_path_num)
    dataloaders['val'] = Dataloader_dynamics_neb_2(hdf5_file, 'valid', negative_path_num)
    dataloaders['test'] = Dataloader_dynamics_neb_2(hdf5_file, 'test', negative_path_num)
    return dataloaders

def generate_dataloader_dynamics_neb_3(hdf5_file, size):
    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset = Dataset_dynamics_neb_3(hdf5_file, 'train'), batch_size = size)
    dataloaders['val'] = DataLoader(dataset = Dataset_dynamics_neb_3(hdf5_file, 'valid'), batch_size = size)
    dataloaders['test'] = DataLoader(dataset = Dataset_dynamics_neb_3(hdf5_file, 'test'), batch_size = size)
    return dataloaders