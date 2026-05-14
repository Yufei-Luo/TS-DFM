import os
import re
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from ase import Atoms, Atom
from ase.data import atomic_masses, atomic_numbers
import numpy as np
from functools import lru_cache
from tqdm import tqdm
import sys
import pickle
import h5py
import random
from torch_scatter import scatter_mean, scatter_add
num_images = 1

def Kabsch_alignment(pos_to_fit, pos, batch):
    pos_to_fit = torch.tensor(pos_to_fit)
    pos = torch.tensor(pos)
    v = pos.shape[-1]
    center = scatter_mean(pos, batch, dim = -2) # B * 3
    pos_to_fit_center = scatter_mean(pos_to_fit, batch, dim = -2) # B * 3
    pos_c = pos - center[batch]
    pos_to_fit_c = pos_to_fit - pos_to_fit_center[batch]
    pos_c = pos_c.repeat([1,v])
    pos_to_fit_c = pos_to_fit.repeat_interleave(v,dim=-1)
    H = scatter_add(pos_c * pos_to_fit_c, batch, dim = -2).reshape(-1,v,v) # B * 3 * 3
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V @ U.transpose(2,1)
    t = center - (pos_to_fit_center.unsqueeze(1) @ R.transpose(2,1)).squeeze(1)
    R = R[batch]
    t = t[batch]
    p_aligned = (pos_to_fit.unsqueeze(1) @ R.transpose(2,1)).squeeze(1) + t
    return p_aligned.numpy()


def get_B(x, pos):
    from .normal_mode_analysis import vibrational_basis
    masses = list(map(lambda s: atomic_masses[s], x))
    return vibrational_basis(pos, masses)

def get_weight(x):
    masses = list(map(lambda s: atomic_masses[s], x))
    m = np.ravel(np.outer(masses,[1.0]*3))
    return np.sqrt(np.outer(m,m))

def calculate_edm(positions):
    out = []
    for position in positions:
        N = len(position)
        X = np.transpose(position)
        one=np.ones((N,1))
        G = np.transpose(X)@X
        diag_G = np.diag(G).reshape((N,1))
        out.append( one@diag_G.T - 2* G  + diag_G@one.T )
    return np.array(out)

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

    for energy, force, positions in zip(energies, forces, positions):
        d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": positions,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }

        yield d

def get_dynamics_data(formula, rxn, data, test=False):
    reactant = next(generator(formula, rxn, data[formula][rxn]["reactant"]))
    product = next(generator(formula, rxn, data[formula][rxn]["product"]))
    transition_state = next(generator(formula, rxn, data[formula][rxn]["transition_state"]))

    reactant_pos = reactant['positions']
    transition_state_pos = Kabsch_alignment(transition_state['positions'], reactant_pos, torch.zeros((reactant_pos.shape[0]), dtype=torch.long))
    product_pos = Kabsch_alignment(product['positions'], reactant_pos, torch.zeros((reactant_pos.shape[0]), dtype=torch.long))
    
    if test:
        return  reactant['atomic_numbers'], \
                calculate_edm([reactant_pos, transition_state_pos, product_pos]), \
                calculate_edm([reactant_pos, (reactant_pos + product_pos)/ 2, product_pos]), \
                transition_state['wB97x_6-31G(d).energy'] - reactant['wB97x_6-31G(d).energy'], \
                transition_state['wB97x_6-31G(d).energy'] - product['wB97x_6-31G(d).energy'], \
                np.stack([get_B(reactant['atomic_numbers'], reactant['positions']), get_B(transition_state['atomic_numbers'], transition_state['positions']), get_B(product['atomic_numbers'], product['positions']),]), \
                get_weight(reactant['atomic_numbers']), \
                reactant['positions'], product['positions'], transition_state['positions'], transition_state['wB97x_6-31G(d).energy']
    # return: atomic number, true pairwise dist (re, tr, pr), linear interp pairwise dist, energy barrier reactant, energy barrier product, vibrational basis (re, tr, pr)
    return reactant['atomic_numbers'], \
           calculate_edm([reactant_pos, transition_state_pos, product_pos]), \
                calculate_edm([reactant_pos, (reactant_pos + product_pos)/ 2, product_pos]), \
           transition_state['wB97x_6-31G(d).energy'] - reactant['wB97x_6-31G(d).energy'], \
           transition_state['wB97x_6-31G(d).energy'] - product['wB97x_6-31G(d).energy'], \
           np.stack([get_B(reactant['atomic_numbers'], reactant['positions']), get_B(transition_state['atomic_numbers'], transition_state['positions']), get_B(product['atomic_numbers'], product['positions']),]), \
           get_weight(reactant['atomic_numbers'])


class DB(IterableDataset):
    def __init__(self, hdf5_file, datasplit):
        super(DB, self).__init__()
        self.hdf5_file = hdf5_file
        self.datasplit = datasplit
        assert datasplit in [
            "train",
            "valid",
            "test",
        ]
        with open('learnts/reactions_'+self.datasplit+'.pickle', 'rb') as f:
            self.datalist = pickle.load(f)

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            data = f['data']
            if self.datasplit == 'train':
                random.shuffle(self.datalist)
            for formula, rxn in self.datalist:
                yield get_dynamics_data(formula, rxn, data, self.datasplit=='test')
                    
    def __len__(self):
        pass

def my_collate_fn(batch, num_images, grid, alpha, duplicate=False, test=False):
    factor = 2 if (duplicate) else 1
    batch_size = len(batch)
    max_atoms  = max( [ len(item[0]) for item in batch ] )

    atomic_numbers = np.zeros((factor*batch_size, max_atoms ), dtype=np.int64)
    true_edm  = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms ), dtype=np.float32)
    linear_edm  = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms ), dtype=np.float32)
    energy_r = np.zeros((factor*batch_size ) )
    energy_p = np.zeros((factor*batch_size ) )

    B_matrices = np.zeros((factor*batch_size, 3, 3*max_atoms, 3*max_atoms-6), dtype=np.float32)
    inv_weights  = np.zeros((factor*batch_size, 3*max_atoms, 3*max_atoms ), dtype=np.float32)

    edm_mask = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms), dtype=bool)
    edm_diag_mask = np.ones((factor*batch_size, 2*num_images+1, max_atoms, max_atoms), dtype=bool)
    edm_diag_mask[:,:, np.arange(max_atoms), np.arange(max_atoms)] =False


    for i, temp in enumerate(batch):
        if not test:
            atomic_numbers_, edm_, linear_edm_, e_r_, e_p_, B_, weights_ = temp
        else:
            atomic_numbers_, edm_, linear_edm_, e_r_, e_p_, B_, weights_, reactant_pos, product_pos, transition_state_pos, trans_energy = temp
        num_atoms = len(atomic_numbers_)
        atomic_numbers [i,:num_atoms] = atomic_numbers_
        true_edm [i,:, :num_atoms, :num_atoms] = edm_
        linear_edm [i,:, :num_atoms, :num_atoms] = linear_edm_
        energy_r[i] = e_r_
        energy_p[i] = e_p_

        B_matrices[i,:,:3*num_atoms, :3*num_atoms-6] = B_
        inv_weights[i,:3*num_atoms, :3*num_atoms] = weights_

        edm_mask[i,:, :num_atoms, :num_atoms] = True
    if (duplicate):
        for i, temp in enumerate(batch):
            if not test:
                atomic_numbers_, edm_, linear_edm_, e_r_, e_p_, B_, weights_ = temp
            else:
                atomic_numbers_, edm_, linear_edm_, e_r_, e_p_, B_, weights_, reactant_pos, product_pos, transition_state_pos, trans_energy = temp
            num_atoms = len(atomic_numbers_)
            atomic_numbers [batch_size+i,:num_atoms] = atomic_numbers_
            true_edm [batch_size+i,:, :num_atoms, :num_atoms] = edm_[::-1]
            linear_edm [batch_size+i,:, :num_atoms, :num_atoms] = linear_edm_[::-1]
            energy_r[batch_size+i] = e_p_
            energy_p[batch_size+i] = e_r_

            B_matrices[batch_size+i,:,:3*num_atoms, :3*num_atoms-6] = B_
            inv_weights[i,:3*num_atoms, :3*num_atoms] = weights_

            edm_mask[batch_size+i,:, :num_atoms, :num_atoms] = True

    grid_ = np.reshape(grid, (1,1,1,-1) )
    triu_indices = np.triu_indices(max_atoms )

    true_edm        = np.sqrt(true_edm[:,:,triu_indices[0], triu_indices[1] ] )
    linear_edm = np.sqrt(linear_edm[:,:,triu_indices[0], triu_indices[1] ] )

    edm_mask   = edm_mask[:,:,triu_indices[0], triu_indices[1]]
    edm_diag_mask   = edm_diag_mask[:,:,triu_indices[0], triu_indices[1]]


    edge_f = np.exp( -alpha* (grid_-np.expand_dims(linear_edm,-1))**2 )
    edge_f[edm_mask==False] = 0.0

    if test:
        return torch.from_numpy(atomic_numbers).type(torch.long),\
           torch.from_numpy(true_edm).type(torch.float32),\
           torch.from_numpy(linear_edm).type(torch.float32),\
           torch.from_numpy(edge_f).type(torch.float32),\
           torch.from_numpy(energy_r).type(torch.float32),\
           torch.from_numpy(energy_p).type(torch.float32),\
           torch.from_numpy(B_matrices).type(torch.float32),\
           torch.from_numpy(inv_weights).type(torch.float32),\
           torch.from_numpy(edm_mask).type(torch.bool),\
           torch.from_numpy(edm_diag_mask).type(torch.bool), \
           torch.from_numpy(reactant_pos).type(torch.float32), \
           torch.from_numpy(product_pos).type(torch.float32), \
           torch.from_numpy(transition_state_pos).type(torch.float32), \
           torch.from_numpy(np.array(trans_energy)).type(torch.float32),\

    return torch.from_numpy(atomic_numbers).type(torch.long),\
           torch.from_numpy(true_edm).type(torch.float32),\
           torch.from_numpy(linear_edm).type(torch.float32),\
           torch.from_numpy(edge_f).type(torch.float32),\
           torch.from_numpy(energy_r).type(torch.float32),\
           torch.from_numpy(energy_p).type(torch.float32),\
           torch.from_numpy(B_matrices).type(torch.float32),\
           torch.from_numpy(inv_weights).type(torch.float32),\
           torch.from_numpy(edm_mask).type(torch.bool),\
           torch.from_numpy(edm_diag_mask).type(torch.bool)