import logging
import pathlib
import sys

import argparse
import numpy as np
from reactot.trainer.pl_trainer import SBModule

from rich.console import Console
from rich.logging import RichHandler

device = "cuda"

def setup_logger(log_dir: pathlib.Path) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = open(log_dir / "log.txt", "w")
    file_console = Console(file=log_file, width=150)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[RichHandler(), RichHandler(console=file_console)],
    )

def load_model(
    checkpoint_path="/mnt/beegfs/bulk/mirror/yuanqi/reactOT/reactot/checkpoint/RPSB-FT-Schedule/leftnet-ts_guess_NEBCI-xtb-ema-785cb2e522c3/sb-epoch=009-val_ep_scaled_err=0.0645.ckpt",
):
    print (checkpoint_path)
    model = SBModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
    )
    model = model.eval()
    model = model.to(device)

    model.training_config["use_sampler"] = False
    model.training_config["swapping_react_prod"] = False
    model.training_config["datadir"] = "./reactot/data/transition1x"

    model.setup(stage="fit", device=device, swapping_react_prod=False)
    return model

def main(opt):
    setup_logger(pathlib.Path(".log"))
    log = logging.getLogger(__name__)

    log.info("===== Start =====")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))

    model = load_model(opt.checkpoint)

    val_loader = model.val_dataloader(bz=opt.batch_size, shuffle=False)
    model.nfe = opt.nfe
    model.ddpm.opt = opt # hack :)

    if opt.dryrun:
        # dryrun: just first batch
        batch = next(iter(val_loader))
        r_pos, ts_pos, p_pos, x0_size, x0_other, rmsds = model.eval_sample_batch(
            batch,
            return_all=True,
        )  # 30s for nfe=100
    else:
        # full val set
        res, rmsds = model.eval_rmsd(
            val_loader,
            write_xyz=False,
            bz=opt.batch_size,
            refpath="ref_ts",
            localpath=f"{opt.solver}-{opt.method}/nfe{opt.nfe}/",
            # max_num_batch=10,
        )
        # np.savez(f"data/{opt.save}.npz", res=res, rmsds=rmsds)

    log.info(f"mean={np.mean(rmsds):.5f}, median={np.median(rmsds):.5f}, {len(rmsds)=}")
    log.info("===== End =====")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",     type=int,   default=72)
    parser.add_argument("--nfe",            type=int,   default=100)
    parser.add_argument("--save",           type=str,   default="debug")
    parser.add_argument("--dryrun",         action="store_true")

    parser.add_argument("--solver",         type=str,   choices=["ddpm", "ei", "ode"])
    parser.add_argument("--checkpoint",     type=str)

    # ei
    parser.add_argument("--order",          type=int, default=1)
    parser.add_argument("--diz",            type=str, default="linear", choices=["linear", "quad"])
    parser.add_argument("--normalize",      action="store_true")

    # ode
    parser.add_argument("--method",         type=str,   default="midpoint")
    parser.add_argument("--atol",           type=float, default=1e-2)
    parser.add_argument("--rtol",           type=float, default=1e-2)

    opt = parser.parse_args()
    main(opt)
