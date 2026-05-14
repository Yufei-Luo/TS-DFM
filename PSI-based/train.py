import os
import torch
from   torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from   pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from   pytorch_lightning.loggers import CSVLogger
from   learnts.db import num_images
from learnts.model import MyModel

if __name__=="__main__":
    import os
    import pickle
    import numpy as np
    from learnts.db import DB, my_collate_fn
    from torch.utils.data import DataLoader
    from functools import partial
    from argparse import ArgumentParser
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    import socket
    from itertools import product
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)


    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_pair_embedding", type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_gru_layers', type=int, default=2)
    parser.add_argument('--gru_dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--factor1', type=float, default=100.0,help="This value correspond c in Equation 5 in the reference paper")
    parser.add_argument('--factor2', type=float, default=1.0, help="This value correspond c' in Equation 5 in the reference paper")
    parser.add_argument('--factor3', type=float, default=0.0)
    parser.add_argument('--factor4', type=float, default=0.0)
    parser.add_argument('--factor5', type=float, default=1.0, help="This value correspond c' in Equation 5 in the reference paper")
    parser.add_argument('--hdf5_file', type=str)
    parser.add_argument('--precision', type=str, default=64)
    parser.add_argument('--log_every_n_steps', type=int, default=10)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='cuda')
    parser.add_argument('--max_epoch', type=int, default=3000)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    args = parser.parse_args()

    grid = np.mgrid[0.0:30.0:301j]

    db = DB(args.hdf5_file, 'train')

    energy_list = []
    for temp in iter(db):
        _, __, ___, energy_r, energy_p, ______, _______ = temp
        energy_list.append(energy_r)
        energy_list.append(energy_p)
    mean = np.mean(energy_list)
    std = np.std(energy_list)
    print(mean, std)

    loader1 = DataLoader(db, batch_size = args.batch_size,  collate_fn=partial(my_collate_fn, num_images=num_images, grid = grid, alpha=10.0, duplicate=True))
    
    db = DB(args.hdf5_file, 'valid')
    loader2 = DataLoader(db, batch_size = 2*args.batch_size, collate_fn=partial(my_collate_fn, num_images=num_images, grid = grid, alpha=10.0, duplicate=False))
    
    print(args.num_pair_embedding, args.num_layers, args.gru_dropout, args.learning_rate, args.precision)
    checkpoint_callback = ModelCheckpoint(  monitor='ts_edm_loss_v', 
                                            save_top_k = 5, save_last=True)
    earlystopping_callback = EarlyStopping(monitor='ts_edm_loss_v', mode='min', patience=20)
    args.callbacks=[checkpoint_callback, earlystopping_callback]

    csv_logger = CSVLogger('lightning_csv_logs', name=f'{args.num_pair_embedding}-{args.num_layers}-{args.num_gru_layers}-{args.gru_dropout}-{args.factor1}-{args.factor2}-{args.factor3}-{args.factor4}-{args.factor5}-{args.learning_rate:.5E}')

    model = MyModel(args.num_pair_embedding, len(grid), e_mean=mean, e_std=std, num_layers=args.num_layers, num_gru_layers=args.num_gru_layers, gru_dropout = args.gru_dropout, learning_rate=args.learning_rate, factor=[args.factor1, args.factor2, args.factor3, args.factor4, args.factor5 ])

    trainer = pl.Trainer(max_epochs=args.max_epoch, gradient_clip_val=args.gradient_clip_val, log_every_n_steps=args.log_every_n_steps, num_nodes=args.num_nodes, precision=args.precision, accelerator=args.accelerator, callbacks=args.callbacks)

    trainer.logger=csv_logger
    trainer.fit(model, loader1, loader2)
    
