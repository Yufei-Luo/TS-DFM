import torch
from torch import nn, optim
import argparse
import sys
sys.path.append('.')
import os
import yaml
from easydict import EasyDict
from collections import OrderedDict
import random
import numpy as np
import pickle as pkl

from copy import deepcopy

import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from Data.RGD1 import generate_dataloader_dynamics
from Model.model import DistFlowMatchingNetwork, ODEWrapper2
from Model.backbone import generate_backbone
from Model.head import generate_head
from Model.model import MDNet
from Utils import get_logger, get_new_log_dir, seed_all, Kabsch_alignment, generate_fully_connected, calc_norm, pairwise_dist_to_coord
import math
import gc

parser = argparse.ArgumentParser(description='Training rgd1 dynamics')
parser.add_argument('--config_file', required=True)
parser.add_argument('--log_prefix', default='logs')
parser.add_argument('--notes', default=' ')
parser.add_argument('--device', default='cuda')
parser.add_argument('--resume_status', default=' ')
parser.add_argument('--random', action='store_true')
parser.add_argument('--sigma', default=1.0, type=float)
args = parser.parse_args()

dtype = torch.float32

config_path=args.config_file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)
config.notes = args.notes

device = args.device

seed_all(config.train.seed)
torch.backends.cudnn.benchmark = True

dataloaders = generate_dataloader_dynamics(config.data.path, config.data.batch_size)
dynamic_model = DistFlowMatchingNetwork(**config.dynamic_model.parameters)
ode = ODEWrapper2(dynamic_model)


print(sum(p.numel() for p in dynamic_model.parameters()))

# create logger and log folder
prefix = args.log_prefix
log_dir = get_new_log_dir(config.train.save_path + '/' , prefix=prefix)
ckpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
logger = get_logger('train', log_dir)

#logger.info(args)
logger.info(config)

best_val_loss = None

dynamic_model.load_state_dict(torch.load('/home/yufeiluo/research/transition_state_pred/logs/dynamics_flow/fm_dist_rgd1_equalweights_2025_09_16__18_53_35/checkpoints/checkpoint_best.pth')['model'])

optimizer = optim.Adam([param for name, param in dynamic_model.named_parameters()], lr=config.optimizer.lr, weight_decay=float(config.optimizer.weight_decay))
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             config.train.epochs,
#             config.scheduler.min_lr
#         )
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config.scheduler.patience,
            factor=config.scheduler.factor,
            min_lr=config.scheduler.min_lr
        )

dynamic_model = dynamic_model.to(device)

param_cnt_dynamic = len(list(dynamic_model.parameters()))

if args.resume_status != " ":
    temp = torch.load(args.resume_status)
    dynamic_model.load_state_dict(temp['model'])
    optimizer.load_state_dict(temp['optimizer'])
    lr_scheduler.load_state_dict(temp['scheduler'])
    print("Status loaded")

def get_loss_train(model, data):
    x = data.x.to(device)
    reactant_pos = data.reactant_pos.to(device)
    product_pos = data.product_pos.to(device)
    transition_state_pos = data.transition_state_pos.to(device)
    batch = data.batch.to(device)

    time = torch.rand((torch.max(batch) + 1, 1), dtype=torch.float32, device=device)[batch]
    # time = torch.ones((torch.max(batch) + 1, 1), dtype=torch.float32, device=device)[batch]

    src, dst = generate_fully_connected(batch)

    edge_index = torch.concat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

    dist_reactant = torch.norm(reactant_pos[src] - reactant_pos[dst], p=2, dim=-1)
    dist_product = torch.norm(product_pos[src] - product_pos[dst], p=2, dim=-1)

    interp_t = torch.ones_like(batch, dtype=torch.float32) * 0.5

    dist_lin_interp = (1 - interp_t[batch[src]]) * torch.norm(reactant_pos[src] - reactant_pos[dst], p=2, dim=-1) + interp_t[batch[src]] * torch.norm(product_pos[src] - product_pos[dst], p=2, dim=-1)

    dist_transition_state = torch.norm(transition_state_pos[src] - transition_state_pos[dst], p=2, dim=-1)

    dist_t = (1 - time[src].squeeze()) * dist_lin_interp + time[src].squeeze() * dist_transition_state

    if args.random:
        temp = torch.ones_like(time)
        sigma_t = args.sigma * temp
        
        rand_t = torch.randn_like(dist_t)

        dist_t = dist_t + rand_t * sigma_t[src].squeeze()

    pred_flow = model(time, x, edge_index, dist_reactant, dist_product, dist_t, batch)

    true_flow = dist_transition_state - dist_t

    loss_temp = F.mse_loss(pred_flow, true_flow, reduction='none')  / (dist_transition_state + 1e-6)
    loss_temp = scatter_mean(loss_temp, batch[src])
    loss_flow = loss_temp.mean()

    loss = loss_flow

    return loss

def get_loss_val(model, data):
    x = data.x.to(device)
    reactant_pos = data.reactant_pos.to(device)
    product_pos = data.product_pos.to(device)
    transition_state_pos = data.transition_state_pos.to(device)
    batch = data.batch.to(device)

    src, dst = generate_fully_connected(batch)

    edge_index = torch.concat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

    dist_reactant = torch.norm(reactant_pos[src] - reactant_pos[dst], p=2, dim=-1)
    dist_product = torch.norm(product_pos[src] - product_pos[dst], p=2, dim=-1)

    interp_t = torch.ones_like(batch, dtype=torch.float32) * 0.5

    dist_lin_interp = (1 - interp_t[batch[src]]) * torch.norm(reactant_pos[src] - reactant_pos[dst], p=2, dim=-1) + interp_t[batch[src]] * torch.norm(product_pos[src] - product_pos[dst], p=2, dim=-1)

    dist_transition_state = torch.norm(transition_state_pos[src] - transition_state_pos[dst], p=2, dim=-1)

    dist_transition_state_pred = model(x, edge_index, dist_reactant, dist_product, dist_lin_interp, batch)

    loss_temp = F.l1_loss(dist_transition_state_pred, dist_transition_state, reduction='none')  / (dist_transition_state + 1e-6)
    loss_temp = scatter_mean(loss_temp, batch[src])
    loss = loss_temp.mean()

    return loss

def train(epoch, dataloader_train, config):
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    dynamic_model.train()

    for i, data in enumerate(dataloader_train):
        optimizer.zero_grad()
 
        loss = get_loss_train(dynamic_model, data)
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(dynamic_model.parameters(), 1.0)
        if (not norm.isinf() and not norm.isnan()):
            optimizer.step()

        batch_size = config.data.batch_size
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        prefix = 'Train >> '

        if (i % config.train.log_interval) == 0:
            logger.info(prefix + 'Epoch: {:4d} | Iter: {:4d} | loss: {:.6f} | lr: {:.6f} | grad norm: {:.6f}'.format(epoch, i, loss.item(), optimizer.state_dict()['param_groups'][0]['lr'], norm))

    return res['loss'] / res['counter']

def valid(epoch, loader, config):
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    dynamic_model.eval()
    for i, data in enumerate(loader):
        # if args.random:
        #     loss = get_loss_train(dynamic_model, data)
        # else:
        #     loss = get_loss_val(ode, data)
        loss = get_loss_val(ode, data)
        batch_size = config.data.batch_size
        
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

    return res['loss'] / res['counter']    

if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    all_train_loss, all_val_loss, all_test_loss = [], [], []

    for epoch in range(0, config.train.epochs):
        train_loss = train(epoch, dataloaders['id_train'], config)
        all_train_loss.append(train_loss)

        if epoch % config.train.val_interval == 0:
            val_loss = valid(epoch, dataloaders['id_val'], config)
            res['epochs'].append(epoch)
            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_epoch'] = epoch
                state = {
                    "model": dynamic_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch
                }
                torch.save(state, ckpt_dir + "/checkpoint_best.pth")
            logger.info("Val loss: %.6f \t epoch %d" % (val_loss, epoch))
            logger.info("Best: val loss: %.6f \t epoch %d"% (res['best_val'], res['best_epoch']))
            all_val_loss.append(val_loss) # save current loss

            # lr_scheduler.step()
            lr_scheduler.step(val_loss)

            if res['best_epoch'] + config.scheduler.stop_tolerance < epoch:
                break

    best_state = torch.load(ckpt_dir + "/checkpoint_best.pth", map_location=device)
    dynamic_model.load_state_dict(best_state['model'])

    test_loss = valid(epoch, dataloaders['id_test'], config)
    logger.info("Test loss id: %.6f " % (test_loss))

    test_loss = valid(epoch, dataloaders['ood_test_type'], config)
    logger.info("Test loss ood type: %.6f " % (test_loss))

    test_loss = valid(epoch, dataloaders['ood_test_size'], config)
    logger.info("Test loss ood size: %.6f " % (test_loss))
