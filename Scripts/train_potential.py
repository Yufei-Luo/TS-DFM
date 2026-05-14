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

from Data.Transition1x import generate_dataloader_potential
from Model.backbone import generate_backbone
from Model.head import generate_head
from Model.model import MDNet
from Utils import get_logger, get_new_log_dir, seed_all, load_model

parser = argparse.ArgumentParser(description='Training Transition1x potential')
parser.add_argument('--config_file', required=True)
parser.add_argument('--log_prefix', default='logs')
parser.add_argument('--notes', default=' ')
parser.add_argument('--device', default='cuda')
parser.add_argument('--resume_status', default=' ')
parser.add_argument('--pretrained_model', default=' ')
args = parser.parse_args()

dtype = torch.float32

config_path=args.config_file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)
config.notes = args.notes

device = args.device

config.model.pretrained_model = args.pretrained_model if args.pretrained_model != ' ' else config.model.pretrained_model

seed_all(config.train.seed)
torch.backends.cudnn.benchmark = True

dataloaders = generate_dataloader_potential(config.data.path, config.data.batch_size)

def get_mean_std(dataloader):
    ys = torch.cat([batch.atomization_energy for batch in dataloader])
    return ys.mean(), ys.std()

# compute mean and mean absolute deviation
mean, std = get_mean_std(dataloaders['val'])

backbone = generate_backbone(config.model.backbone)
head = generate_head(config.model.head)

head.mean = mean
head.std = std

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}

model = MDNet(backbone, head, REFERENCE_ENERGIES)
print(model)
print(sum(p.numel() for p in model.parameters()))

if config.model.pretrained_model is not None:
    encoder_param = load_model(model.backbone, config.model.pretrained_model, 'pretrained_backbone')
    print('load model from', config.model.pretrained_model)

# create logger and log folder
prefix = args.log_prefix
log_dir = get_new_log_dir(config.train.save_path + '/', prefix=prefix)
ckpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
logger = get_logger('train', log_dir)

#logger.info(args)
logger.info(config)

best_val_loss = None

optimizer = optim.Adam([param for name, param in model.named_parameters()], lr=config.optimizer.lr, weight_decay=float(config.optimizer.weight_decay))
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=float(config.scheduler.min_lr),
        )

start_step = 0

model = model.to(device)

if args.resume_status != ' ':
    print('Loading training status ...')
    state = torch.load(args.resume_status, map_location=device)
    model.load_state_dict(state['model'])
    start_step = state['step']
    optimizer.load_state_dict(state['optimizer'])
    lr_scheduler.load_state_dict(state['scheduler'])

loss_l1 = nn.L1Loss()
loss_l2 = nn.MSELoss()
if config.train.loss == 'mse':
    loss_func = loss_l2
else:
    loss_func = loss_l1

def train_batch(data, config, partition='train'):
    if partition == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    x = data.x.to(device)
    pos = data.pos.to(device)
    edge_index = None
    edge_attr = None
    energy = data.energy.to(device)
    force = data.force.to(device)
    batch = data.batch.to(device)
    
    pred_energy, pred_force = model.get_energy_and_force(x, pos, edge_index, edge_attr, batch)
    batch_size = len(pred_energy)
    if partition == "train":
        loss_energy = loss_func(pred_energy, energy)
        loss_force = loss_func(pred_force, force)
    else:
        loss_energy = loss_l1(pred_energy, energy)
        loss_force = loss_l1(pred_force, force)   
    loss = loss_energy * config.train.loss_energy + loss_force * config.train.loss_force
    
    if partition == 'train':
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not norm.isinf() and not norm.isnan():
            optimizer.step()

    return loss.item(), loss_energy.item(), loss_force.item()

if __name__ == "__main__":
    res = {'steps': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_step': 0}
    
    all_train_loss, all_val_loss, all_test_loss = [], [], []
    for step in range(start_step, config.train.steps):
        data = next(iter(dataloaders['train']))
        train_loss, train_energy, train_force = train_batch(data, config, partition='train')

        all_train_loss.append(train_loss)

        if step % config.train.log_interval == 0:
            logger.info('Train: ' + 'Step: {:4d} | loss: {:.6f} | energy: {:.6f} | force: {:.6f} | lr: {:.6f}'
                  .format(step, train_loss, train_energy, train_force, optimizer.state_dict()['param_groups'][0]['lr']))
        if step % config.train.val_interval == config.train.val_interval - 1:
            res_val = {'loss': 0, 'counter': 0, 'loss_arr':[], 'energy': 0, 'energy_arr':[], 'force': 0, 'force_arr':[]}
            for val_step in range(0, config.train.val_steps):
                data = next(iter(dataloaders['val']))
                val_loss, val_energy, val_force = train_batch(data, config, partition='valid')
                res_val['loss_arr'].append(val_loss * config.data.batch_size)
                res_val['energy_arr'].append(val_energy * config.data.batch_size)
                res_val['force_arr'].append(val_force * config.data.batch_size)
                res_val['counter'] += config.data.batch_size

            val_loss = np.sum(res_val['loss_arr']) / res_val['counter']

            lr_scheduler.step(val_loss)

            # save current loss
            logger.info("Val loss: %.6f \t step %d" % (val_loss, step))
            all_val_loss.append(val_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_step'] = step
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "step": step
                }
                torch.save(state, ckpt_dir + "/checkpoint_best.pth")
            else:
                if abs(config.scheduler.min_lr - optimizer.state_dict()['param_groups'][0]['lr']) < 1e-7 and step - res['best_step'] > config.train.stop_tolarance:
                    logger.info("Exceed stop tolarance.")
                    break
                
            logger.info("Best: val loss: %.6f \t step %d"% (res['best_val'], res['best_step']))
    
    best_state = torch.load(ckpt_dir + "/checkpoint_best.pth", map_location=device)
    model.load_state_dict(best_state['model'])

    res_test = {'loss': 0, 'counter': 0, 'loss_arr':[], 'energy': 0, 'energy_arr':[], 'force': 0, 'force_arr':[]}
    for data in dataloaders['test']:
        test_loss, test_energy, test_force = train_batch(data, config, partition='test')
        res_test['loss_arr'].append(test_loss * config.data.batch_size)
        res_test['energy_arr'].append(test_energy * config.data.batch_size)
        res_test['force_arr'].append(test_force * config.data.batch_size)
        res_test['counter'] += config.data.batch_size

    test_loss = np.sum(res_test['loss_arr']) / res_test['counter']
    test_energy = np.sum(res_test['energy_arr']) / res_test['counter']
    test_force = np.sum(res_test['force_arr']) / res_test['counter']
    logger.info("Test loss: %.6f \t energy: %.6f \t force %.6f" % (test_loss, test_energy, test_force))
    
    loss_file = ckpt_dir + '/loss.pkl'
    
    with open(loss_file, 'wb') as f:
        pkl.dump((all_train_loss, all_val_loss, test_loss, test_energy, test_force), f)
   
