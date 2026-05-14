import torch
import numpy as np
import time
import os
import math
import logging
import torch.nn.functional as F
import argparse
import yaml
from easydict import EasyDict
from collections import OrderedDict
import random
from torch.optim import LBFGS
from torch_sparse import SparseTensor

import torch_geometric

from ase import Atoms
from ase.mep.neb import NEB, NEBTools, NEBOptimizer

from Utils.ase_utils import construct_atoms

def calculate_loss_coord(adj_matrix, coord, edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    diff = coord[src] - coord[dst]
    dists = torch.norm(diff, p=2, dim=-1)
    loss = torch.sum(torch.square(dists - adj_matrix) * torch.pow(1 / (adj_matrix + 1e-6), 2))
    return loss

def pairwise_dist_to_coord(x, reactant_pos, product_pos, pairwise_distance_ts_pred):
    reactant = construct_atoms(x, reactant_pos)
    product = construct_atoms(x, product_pos)

    src, dst = generate_fully_connected(torch.zeros_like(x))
    edge_index = torch.concat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

    i = 0
    error_opt = 1000000.0

    while i < 10 and error_opt > 0.01:
        coords_temp = (reactant_pos + product_pos) / 2
        coords_temp = coords_temp + torch.randn_like(coords_temp) * 0.1
        coords_temp.requires_grad = True
        optimizer_temp = LBFGS([coords_temp], max_iter=100, lr=0.1)
        def closure_temp():
            optimizer_temp.zero_grad()
            loss = calculate_loss_coord(pairwise_distance_ts_pred, coords_temp, edge_index)
            loss.backward()
            return loss
        optimizer_temp.step(closure_temp)
        error_temp = calculate_loss_coord(pairwise_distance_ts_pred, coords_temp, edge_index)
        if error_temp < error_opt:
            coords = coords_temp
            error_opt = error_temp
        i += 1

    return coords, error_opt

def generate_fully_connected(batch):
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
    return (src, dst)

def create_angular_index(src, dst, dist=None):
    i = src
    j = dst
    
    num_nodes = torch.max(j) + 1

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t.t()[j] # get adjecent nodes of node j
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long) #num of adjecent nodes of node j

    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    if dist is None:
        return idx_i, idx_j, idx_k

    else:
        dist = SparseTensor(row=i, col=j, value=dist).to_dense()
        dist_ij = dist[idx_i, idx_j]
        dist_jk = dist[idx_j, idx_k]
        return idx_i, idx_j, idx_k, dist_ij, dist_jk
    
def calculate_angle(pos, idx_i, idx_j, idx_k):
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle


def load_model(model, model_path, model_name='model'):
    state = torch.load(model_path)
    params = state[model_name].items()
    model_dict = model.state_dict()
    new_dict_filtered = OrderedDict()
    unmatched_keys = 0
    for k, v in params:
        if k in model_dict and v.size() == model_dict[k].size():
            new_dict_filtered[k] = v
        else:
            new_dict_filtered[k] = model_dict[k]
            print('unmatched: ', k)
            unmatched_keys += 1

    model.load_state_dict(new_dict_filtered, strict=False)
    print('Unmatched keys:', unmatched_keys)
    return new_dict_filtered
    
def load_model_from_dict(model, params_dict):
    model_dict = model.state_dict()
    new_dict_filtered = OrderedDict()
    unmatched_keys = 0
    for k, v in params_dict.items():
        if k in model_dict and v.size() == model_dict[k].size():
            new_dict_filtered[k] = v
        else:
            new_dict_filtered[k] = model_dict[k]
            unmatched_keys += 1

    model.load_state_dict(new_dict_filtered, strict=False)
    print('Unmatched keys:', unmatched_keys)
    return new_dict_filtered
    
def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
    

def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir

def get_mean_std(loader):
    data_list = []
    for i, data in enumerate(loader):
        x, states, batch, timesteps = data
        pos_0 = states[0]
        pos_T = states[-1]
        data_list.append(pos_T - pos_0)
    data = torch.concat(data_list)
    return torch.mean(data), torch.std(data)
    
    
def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        
        torch.cuda.manual_seed_all(seed)

def calc_norm(vec_list):
    norm = 0.0
    for g in vec_list:
        if g is not None:
            norm += (g**2).sum()
    return np.sqrt(norm.item())
