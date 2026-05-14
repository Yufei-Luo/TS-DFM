import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../OAReactDiff-main')
from typing import List, Optional, Tuple
from uuid import uuid4
import os
import shutil
import torch


from reactot.trainer.pl_trainer import SBModule, DDPMModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from reactot.trainer.ema import EMACallback
from reactot.model import LEFTNet

from ipdb import set_trace as debug
import colored_traceback.always


class OPT:
    def __init__(
        self,
        solver,
        method,
    ):
        self.solver = solver
        self.method = method
        self.atol = 1e-2
        self.rtol = 1e-2
    
opt = OPT(solver="ddpm", method="midpoint")

model_type = "leftnet"
version = "ts_guess_NEBCI-xtb-ema"
project = "RPSB-FT-Schedule"
# ---EGNNDynamics---
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
)

if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    lr=1e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

T_0 = 10
T_mult = 1
training_config = dict(
    datadir="reactot/data/RGD1/",
    remove_h=False,
    bz=14,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=True,
    ema_decay=0.999,
    swapping_react_prod=False,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=False,
    # react_type="xTB-IRC",
    # position_key="xtb_positions",
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=10,
    ),  # step
    # lr_schedule_config=dict(
    #     T_0=T_0,
    #     T_mult=T_mult,
    #     eta_min=0,
    # ),  #cos
    use_sampler=False,
    sampler_config=dict(
        max_num=2800,  # This is for 16GB GPU; Scale linearly with memory
        mode="node^2",
        shuffle=True,
        ddp=False,
    )
)
training_data_frac = 1.0 if not training_config["reflection"] else 0.5


node_nfs: List[int] = [9] * 3  # 3 (pos) + 5 (cat) + 1 (charge)
edge_nf: int = 0  # edge type
condition_nf: int = 1
fragment_names: List[str] = ["R", "TS", "P"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None
loss_type = "l2"
pos_only = True
process_type = "TS1x"
enforce_same_encoding = None
scales = [1., 2., 1.]
fixed_idx = [0, 2]
eval_epochs = 1
save_epochs = 1

# ----Normalizer---
norm_values: Tuple = (1., 1., 1.)
norm_biases: Tuple = (0., 0., 0.)

# ---Schedule---
timesteps: int = 3000
beta_max: float = 0.3
power: float = 0.5
inv_power: float = 1
precision: float = 1e-5  # not used
noise_schedule: str = "cosine"  # not used

# ---SB---
mapping: str = "R+P->TS"
mapping_initial: str = "RP"  # RP for (r+p)/2, GUESS for guessing 
nfe: int = 25
ot_ode: bool = True
sigma: float = 0.
ts_guess: bool = None  #"ts_guess_NEBCI-xtb"  # "ts_guess_sbv1"  # "ts_guess_linear"

norms = "_".join([str(x) for x in norm_values])
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

## === Fine tuning from a FF ---
# tspath = "/home/ubuntu/efs/reactot/reactot/trainer/ckpt"
# checkpoint_path=f"{tspath}/TSDiffusion-TS1x-All/leftnet-8-70b75beeaac1/ddpm-epoch=2074-val-totloss=531.18.ckpt"  # All diffuse/denoise
# checkpoint_path=f"{tspath}/leftnet-10-d13a2c2bace6_wo_oa_align/ddpm-epoch=719-val-totloss=680.64.ckpt"
# checkpoint_path=f"{tspath}/TSDiffusion-TS1x-All/RGD1xtb-pretrained-leftnet-0-7962cf1208dc/ddpm-epoch=1279-val-error_t_1=0.237.ckpt"
# checkpoint_path = "/home/ubuntu/efs/reactot/reactot/trainer/checkpoint/TSDiff/leftnet-xtb-from-dftckpt-cd01d85c5152/ddpm-epoch=189-val-totloss=619.05.ckpt"
# checkpoint_path = "/home/ubuntu/efs/reactot/reactot/trainer/checkpoint/RPSB-FT-Schedule/leftnet-xtb-c79fcfe0518d/sb-epoch=349-val_ep_scaled_err=0.0483.ckpt"
use_pretrain: bool = False

source = None
if use_pretrain:
    ddpm_trainer = DDPMModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )
    source = {
        "model": ddpm_trainer.ddpm.dynamics.model.state_dict(),
        "encoders": ddpm_trainer.ddpm.dynamics.encoders.state_dict(),
        "decoders": ddpm_trainer.ddpm.dynamics.decoders.state_dict(),
    }
    training_config.update(
        {
            "checkpoint_path": checkpoint_path,
            "use_pretrain": use_pretrain,
        }
    )

seed_everything(42, workers=True)
ddpm = SBModule(
    model_config,
    optimizer_config,
    training_config,
    node_nfs,
    edge_nf,
    condition_nf,
    fragment_names,
    pos_dim,
    update_pocket_coords,
    condition_time,
    edge_cutoff,
    norm_values,
    norm_biases,
    noise_schedule,
    timesteps,
    precision,
    loss_type,
    pos_only,
    process_type,
    model,
    enforce_same_encoding,
    scales,
    source=source,
    fixed_idx=fixed_idx,
    eval_epochs=eval_epochs,
    mapping=mapping,
    mapping_initial=mapping_initial,
    nfe=nfe,
    beta_max=beta_max,
    ot_ode=ot_ode,
    power=power,
    inv_power=inv_power,
    sigma=sigma,
    ts_guess=ts_guess,
)
ddpm.ddpm.opt = opt  # heck for the new optimizer

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None
# if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
#     wandb_logger = WandbLogger(
#         project=project,
#         log_model=False,
#         name=run_name,
#     )
#     try:  # Avoid errors for creating wandb instances multiple times
#         wandb_logger.experiment.config.update(config)
#         wandb_logger.watch(
#             ddpm.ddpm.dynamics, log="all", log_freq=100, log_graph=False
#         )
#     except:
#         pass

ckpt_path = f"checkpoint/{project}/1"
earlystopping = EarlyStopping(
    monitor="val_ep_scaled_err",
    patience=200,
    verbose=True,
    log_rank_zero_only=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_ep_scaled_err",
    dirpath=ckpt_path,
    filename="sb-{epoch:03d}-{val_ep_scaled_err:.4f}",
    every_n_epochs=save_epochs,
    save_top_k=10,
)
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]

strategy = None
devices = [0]
strategy = DDPStrategy(find_unused_parameters=True)
if strategy is not None:
    devices = list(range(torch.cuda.device_count()))
# if len(devices) == 1:
#     strategy = None

if training_config["ema"]:
    callbacks.append(
        EMACallback(
            pl_module=ddpm,
            decay=training_config["ema_decay"])

    )

print("config: ", config)
trainer = Trainer(
    max_epochs=3000,
    accelerator="gpu",
    deterministic=False,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=20,
    callbacks=callbacks,
    profiler=None,
    # logger=wandb_logger,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    limit_train_batches=200,
    limit_val_batches=20,
    # replace_sampler_ddp=False,
    # resume_from_checkpoint=checkpoint_path,
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)
