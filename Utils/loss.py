import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from Utils.alignment import Kabsch_alignment
import itertools
from itertools import combinations

import numpy as np

def rmsd_loss(pred, target, batch, ignore_chriality=False):
    pred = Kabsch_alignment(pred, target, batch)
    temp = torch.nn.functional.mse_loss(pred, target, reduction='none')
    temp = torch.sum(temp, dim=-1)
    temp = scatter_mean(temp, batch)
    temp = torch.sqrt(temp)
    if ignore_chriality:
        pred[:, -1] = pred[:, -1] * -1
        pred = Kabsch_alignment(pred, target, batch)
        temp2 = torch.nn.functional.mse_loss(pred, target, reduction='none')
        temp2 = torch.sum(temp2, dim=-1)
        temp2 = scatter_mean(temp2, batch)
        temp2 = torch.sqrt(temp2)
        if temp2 < temp:
            temp = temp2
    return temp

def d_mae_loss(pred, target, batch):
    device = batch.device
    src = torch.arange(0, batch.shape[0])
    dst = torch.arange(0, batch.shape[0])
    src = torch.repeat_interleave(src, batch.shape[0])
    dst = dst.repeat(batch.shape[0])
    mask = (batch[src] == batch[dst])
    src = src.to(device)
    dst = dst.to(device)
    src = src[mask]
    dst = dst[mask]
    mask = (src != dst)
    src = src[mask]
    dst = dst[mask]
    diff_pred = torch.norm(pred[src] - pred[dst], p=2, dim=-1)
    diff_true = torch.norm(target[src] - target[dst], p=2, dim=-1)
    loss = scatter_mean(torch.abs(diff_pred - diff_true), batch[src])
    return loss

def rmsd_loss_reorder(pred, target, batch, atom_type):
    # The speed could be slow if batch_size of input is greater than 1
    best_rmsd = 1000000
    idx = torch.arange(0, atom_type.shape[0]).tolist()
    all_permutations = list(itertools.permutations(idx))
    for perm in all_permutations:
        perm = list(perm)
        if torch.all(atom_type == atom_type[perm]):
            rmsd = rmsd_loss(pred, target[perm], batch)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
    return best_rmsd

def bond_error_cnt(bond_pred, bond_true):
    bond_breaking_error_cnt = np.sum((bond_true == 1) & (bond_pred == 0)) // 2
    bond_formation_error_cnt = np.sum((bond_true == 0) & (bond_pred == 1)) // 2
    return bond_breaking_error_cnt, bond_formation_error_cnt

def reaction_center_error_cnt(reactant_bond, product_bond, true_ts_bond, pred_ts_bond):
    def find_reaction_center(bond1, bond2):
        bond_diff = (bond1 != bond2).astype(np.int32)
        n_atoms = bond1.shape[0]
        src = np.arange(n_atoms)
        dst = np.arange(n_atoms)
        src = np.repeat(src, n_atoms)
        dst = np.tile(dst, n_atoms)
        mask = (bond_diff[src, dst] == 1) & (src < dst)
        src = src[mask]
        dst = dst[mask]

        react_centers = set()
        for i in range(src.shape[0]):
            react_centers.add(src[i])
            react_centers.add(dst[i])
        return react_centers
    
    r_ts_center = find_reaction_center(reactant_bond, true_ts_bond)
    ts_p_center = find_reaction_center(true_ts_bond, product_bond)

    true_centers = r_ts_center.union(ts_p_center)

    r_ts_pred_center = find_reaction_center(reactant_bond, pred_ts_bond)
    ts_pred_p_center = find_reaction_center(pred_ts_bond, product_bond)

    pred_centers = r_ts_pred_center.union(ts_pred_p_center)

    return len(pred_centers - true_centers) + len(true_centers - pred_centers)

def calculate_angle(angle_ind, pos):
    n1_indices, centers, n2_indices = angle_ind[:, 0], angle_ind[:, 1], angle_ind[:, 2]

    p1 = pos[n1_indices]
    p2 = pos[centers] 
    p3 = pos[n2_indices] 

    v1 = p1 - p2 
    v2 = p3 - p2 

    temp = v1 * v2
    
    dot_products = np.sum(temp, axis=1)
    norms_v1 = np.linalg.norm(v1, axis=1)
    norms_v2 = np.linalg.norm(v2, axis=1)
    
    cos_theta = dot_products / (norms_v1 * norms_v2 + 1e-10)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angles = np.degrees(np.arccos(cos_theta))

    return angles

def calculate_angle_error(pos_true, pos_pred, bond_true):
    if not isinstance(pos_true, np.ndarray):
        pos_true = pos_true.cpu().numpy()
    if not isinstance(pos_pred, np.ndarray):
        pos_pred = pos_pred.cpu().numpy()

    n_atoms = pos_true.shape[0]
    src = np.arange(n_atoms)
    dst = np.arange(n_atoms)
    src = np.repeat(src, n_atoms)
    dst = np.tile(dst, n_atoms)
    mask = bond_true[src, dst] > 0
    src = src[mask]
    dst = dst[mask]

    neighbors_dict = dict()
    for idx in range(n_atoms):
        neighbors_dict[idx] = []
    for idx in range(src.shape[0]):
        neighbors_dict[src[idx].item()].append(dst[idx].item())

    centers = []
    neighbor_pairs = []
    
    for center in range(n_atoms):
        neighbors = neighbors_dict[center]
        if len(neighbors) >= 2:
            for n1, n2 in combinations(neighbors, 2):
                centers.append(center)
                neighbor_pairs.append((n1, n2))

    if len(centers) == 0:
        return np.array([0.0])

    centers = np.array(centers)
    neighbor_pairs = np.array(neighbor_pairs)
    n1_indices = neighbor_pairs[:, 0]
    n2_indices = neighbor_pairs[:, 1]
    angle_triplets = np.column_stack([n1_indices, centers, n2_indices])

    angle_true = calculate_angle(angle_triplets, pos_true)
    angle_pred = calculate_angle(angle_triplets, pos_pred)
    
    return np.abs(angle_true - angle_pred)

def calculate_proper_dihedrals(proper_quads, pos):
    p0 = pos[proper_quads[:, 0]]  # A
    p1 = pos[proper_quads[:, 1]]  # B
    p2 = pos[proper_quads[:, 2]]  # C
    p3 = pos[proper_quads[:, 3]]  # D

    b0 = p1 - p0  # A->B
    b1 = p2 - p1  # B->C
    b2 = p3 - p2  # C->D
    
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2) 
    
    n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)
    n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)
    n1_unit = n1 / (n1_norm + 1e-10)
    n2_unit = n2 / (n2_norm + 1e-10)
    
    b1_norm = np.linalg.norm(b1, axis=1, keepdims=True)
    b1_unit = b1 / (b1_norm + 1e-10)
    
    cos_phi = np.sum(n1_unit * n2_unit, axis=1)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    
    n1_cross_n2 = np.cross(n1_unit, n2_unit)
    sign = np.sum(n1_cross_n2 * b1_unit, axis=1)
    phi = np.where(sign < 0, -phi, phi)
    
    dihedrals = np.degrees(phi)
    dihedrals = ((dihedrals + 180) % 360) - 180
    
    return dihedrals


def calculate_torsional_error(pos_true, pos_pred, bond_true):
    if not isinstance(pos_true, np.ndarray):
        pos_true = pos_true.cpu().numpy()
    if not isinstance(pos_pred, np.ndarray):
        pos_pred = pos_pred.cpu().numpy()
    n_atoms = pos_true.shape[0]

    src = np.arange(n_atoms)
    dst = np.arange(n_atoms)
    src = np.repeat(src, n_atoms)
    dst = np.tile(dst, n_atoms)
    mask = (bond_true[src, dst] > 0) & (src < dst)
    src = src[mask]
    dst = dst[mask]

    neighbors_dict = dict()
    for idx in range(n_atoms):
        neighbors_dict[idx] = []
    for idx in range(src.shape[0]):
        neighbors_dict[src[idx].item()].append(dst[idx].item())
        neighbors_dict[dst[idx].item()].append(src[idx].item())
    
    neighbors_array = [np.array(neighbors_dict[i], dtype=np.int32) for i in range(n_atoms)]
    
    proper_quads = []
    
    for B in range(n_atoms):
        neighbors_B = neighbors_array[B]
        
        for C in neighbors_B:
            if C <= B: 
                continue
                
            A_candidates = neighbors_B[neighbors_B != C]
            
            neighbors_C = neighbors_array[C]
            D_candidates = neighbors_C[neighbors_C != B]
            
            if len(A_candidates) == 0 or len(D_candidates) == 0:
                continue
            
            for A in A_candidates:
                for D in D_candidates:
                    if A != D:
                        proper_quads.append([A, B, C, D])

    if len(proper_quads) == 0:
        return np.array([0.0])

    proper_quads = np.array(proper_quads)
    dihedral_true = calculate_proper_dihedrals(proper_quads, pos_true)
    dihedral_pred = calculate_proper_dihedrals(proper_quads, pos_pred)
    dihedral_diff = np.abs(dihedral_true - dihedral_pred)
    dihedral_diff = np.where(dihedral_diff > 180, 360 - dihedral_diff, dihedral_diff)
    return dihedral_diff

def calculate_improper_dihedrals(improper_quads, pos):
    pA = pos[improper_quads[:, 0]]  # A
    pB = pos[improper_quads[:, 1]]  # B
    pC = pos[improper_quads[:, 2]]  # C
    pD = pos[improper_quads[:, 3]]  # D

    vAC = pC - pA
    vAD = pD - pA
    normal = np.cross(vAC, vAD)
    
    normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal_unit = normal / (normal_norm + 1e-10)
    
    vAB = pB - pA
    
    distance = np.sum(vAB * normal_unit, axis=1)
    
    AB_length = np.linalg.norm(vAB, axis=1)
    
    sin_theta = np.abs(distance) / (AB_length + 1e-10)
    sin_theta = np.clip(sin_theta, 0.0, 1.0)
    theta = np.degrees(np.arcsin(sin_theta))
    
    return theta

def calculate_improper_error(pos_true, pos_pred, bond_true):
    if not isinstance(pos_true, np.ndarray):
        pos_true = pos_true.cpu().numpy()
    if not isinstance(pos_pred, np.ndarray):
        pos_pred = pos_pred.cpu().numpy()
    improper_quads = []
    n_atoms = pos_true.shape[0]

    improper_quads = []

    src = np.arange(n_atoms)
    dst = np.arange(n_atoms)
    src = np.repeat(src, n_atoms)
    dst = np.tile(dst, n_atoms)
    mask = (bond_true[src, dst] > 0) & (src < dst)
    src = src[mask]
    dst = dst[mask]

    neighbors_dict = dict()
    for idx in range(n_atoms):
        neighbors_dict[idx] = []
    for idx in range(src.shape[0]):
        neighbors_dict[src[idx].item()].append(dst[idx].item())
        neighbors_dict[dst[idx].item()].append(src[idx].item())
    
    neighbors_array = [np.array(neighbors_dict[i], dtype=np.int32) for i in range(n_atoms)]
    
    for B in range(n_atoms):
        neighbors = neighbors_array[B]
        
        if len(neighbors) >= 3:
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    for k in range(j+1, len(neighbors)):
                        A = neighbors[i]
                        C = neighbors[j]
                        D = neighbors[k]
                        improper_quads.append([A, B, C, D])

    if len(improper_quads) == 0:
        return np.array([0.0])
    
    improper_quads = np.array(improper_quads)
    improper_true = calculate_improper_dihedrals(improper_quads, pos_true)
    improper_pred = calculate_improper_dihedrals(improper_quads, pos_pred)
    improper_diff = np.abs(improper_true - improper_pred)
    improper_diff = np.where(improper_diff > 180, 360 - improper_diff, improper_diff)
    return improper_diff