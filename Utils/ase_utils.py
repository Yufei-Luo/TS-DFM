from ase import Atoms
from ase.mep.neb import NEB, NEBTools, NEBOptimizer
from ase.optimize import MDMin, BFGS
from ase.calculators.calculator import Calculator, all_changes

import math

import numpy as np
import torch

from sella import Sella, IRC

from copy import deepcopy

class MLCalculator(Calculator):
    def __init__(
        self,
        model,
        implemented_properties=None,
        device=None,
        **kwargs,
    ):
        if not implemented_properties:
            implemented_properties = ["energy", "forces", "hessian"]
        self.implemented_properties = implemented_properties
        pin_memory = (device == 'cuda')

        super().__init__(**kwargs)

        # self.atoms_converter = atoms_converter
        self.model = model
        if device:
            self.device = device
            model.to(device)
        else:
            self.device = next(model.parameters()).device

    def calculate(
        self, atoms=None, properties=None, system_changes=None
    ):  # pylint:disable=unused-argument
        # if isinstance(atoms, Atoms):
        #     atoms = [atoms]

        if not system_changes:
            system_changes = all_changes

        if not properties:
            properties = self.implemented_properties

        if self.calculation_required(atoms, properties):
            super().calculate(atoms)
            x = torch.tensor(atoms.get_atomic_numbers()).to(self.device)
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float32).to(self.device)
            batch = torch.zeros_like(x).to(self.device)
            if "energy" in properties:
                energy = self.model.get_energy(x, pos, None, None, batch)
            if "forces" in properties:
                try:
                    forces = self.model.get_force(energy, pos)
                except:
                    energy, forces = self.model.get_energy_and_force(x, pos, None, None, batch)
            if "hessian" in properties:
                try:
                    hessian = self.model.get_hessian(forces, pos)
                except:
                    energy, forces = self.model.get_energy_and_force(x, pos, None, None, batch)
                    hessian = self.model.get_hessian(forces, pos)
            
            if "energy" in properties:
                energy = energy.detach().cpu().numpy().squeeze()
                self.results['energy'] = energy
            if "forces" in properties:
                forces = forces.detach().cpu().numpy()
                self.results['forces'] = forces
            if "hessian" in properties:
                hessian = hessian.detach().cpu().numpy()
                self.results['hessian'] = hessian

def construct_atoms(x, pos):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    atoms = Atoms(numbers=x, positions=pos)
    return atoms

def neb(calculator, x, reactant_pos, product_pos, transition_pos_guess=None, log=None, max_steps=500):
    # print log in cmd, set log = '-'
    # save log in files, set log = filepath
    def d_mae_loss(pred, target):
        size = pred.shape[0]
        src = np.arange(0, size)
        dst = np.arange(0, size)
        src = np.tile(src, size)
        dst = np.repeat(dst, size)
        mask = (src != dst)
        src = src[mask]
        dst = dst[mask]
        diff_pred = np.linalg.norm(pred[src] - pred[dst], ord=2, axis=-1)
        diff_true = np.linalg.norm(target[src] - target[dst], ord=2, axis=-1)
        loss = np.mean(np.abs(diff_pred - diff_true))
        return loss

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(reactant_pos, torch.Tensor):
        reactant_pos = reactant_pos.detach().cpu().numpy()
    if isinstance(product_pos, torch.Tensor):
        product_pos = product_pos.detach().cpu().numpy()
    if transition_pos_guess is not None and isinstance(transition_pos_guess, torch.Tensor):
        transition_pos_guess = transition_pos_guess.detach().cpu().numpy()

    reactant = construct_atoms(x, reactant_pos)
    product = construct_atoms(x, product_pos)

    if transition_pos_guess is None:
        atom_configs = [reactant] + [reactant.copy() for _ in range(9)] + [product]

        for atom_config in atom_configs:
            atom_config.calc = deepcopy(calculator)

        fmax_val = 0.16
        converged = True
        optimizer = BFGS(atom_configs[0], maxstep=0.005, logfile=log)
        while converged and fmax_val >= 0.01:
            atom_config_backup = deepcopy(atom_configs[0])
            converged = optimizer.run(fmax=fmax_val, steps=max_steps)
            forces = atom_configs[0].get_forces()
            max_force = math.sqrt((forces ** 2).sum(axis=1).max())
            if max_force > fmax_val:
                converged = False
            fmax_val /= 2
        if not converged:
            atom_configs[0] = atom_config_backup

        fmax_val = 0.16
        converged = True
        optimizer = BFGS(atom_configs[-1], maxstep=0.005, logfile=log)
        while converged and fmax_val >= 0.01:
            atom_config_backup = deepcopy(atom_configs[-1])
            converged = optimizer.run(fmax=fmax_val, steps=max_steps)
            forces = atom_configs[-1].get_forces()
            max_force = math.sqrt((forces ** 2).sum(axis=1).max())
            if max_force > fmax_val:
                converged = False
            fmax_val /= 2
        if not converged:
            atom_configs[-1] = atom_config_backup

        neb = NEB(atom_configs, climb=False, remove_rotation_and_translation=True)
        neb.interpolate(method="idpp")

    else:
        dmae_reactant = d_mae_loss(reactant_pos, transition_pos_guess).item()
        dmae_product = d_mae_loss(product_pos, transition_pos_guess).item()
        total_configs = 8
        reactant_configs = round(total_configs * dmae_reactant / (dmae_reactant + dmae_product))
        product_configs = total_configs - reactant_configs
        pred_trans = construct_atoms(x, transition_pos_guess)
        atom_config1 = [reactant] + [reactant.copy() for _ in range(reactant_configs)] + [pred_trans]
        atom_config2 = [pred_trans] + [product.copy() for _ in range(product_configs)] + [product]

        for atom_config in atom_config1:
            atom_config.calc = deepcopy(calculator)
        for atom_config in atom_config2:
            atom_config.calc = deepcopy(calculator)

        fmax_val = 0.16
        converged = True
        optimizer = BFGS(atom_config1[0], maxstep=0.005, logfile=log)
        while converged and fmax_val >= 0.01:
            atom_config_backup = deepcopy(atom_config1[0])
            converged = optimizer.run(fmax=fmax_val, steps=max_steps)
            forces = atom_config1[0].get_forces()
            max_force = math.sqrt((forces ** 2).sum(axis=1).max())
            if max_force > fmax_val:
                converged = False
            fmax_val /= 2
        if not converged:
            atom_config1[0] = atom_config_backup

        fmax_val = 0.16
        converged = True
        optimizer = BFGS(atom_config2[-1], maxstep=0.005, logfile=log)
        while converged and fmax_val >= 0.01:
            atom_config_backup = deepcopy(atom_config2[-1])
            converged = optimizer.run(fmax=fmax_val, steps=max_steps)
            forces = atom_config2[-1].get_forces()
            max_force = math.sqrt((forces ** 2).sum(axis=1).max())
            if max_force > fmax_val:
                converged = False
            fmax_val /= 2
        if not converged:
            atom_config2[-1] = atom_config_backup

        neb1 = NEB(atom_config1, climb=False, remove_rotation_and_translation=True)
        neb1.interpolate(method="idpp")
        neb2 = NEB(atom_config2, climb=False, remove_rotation_and_translation=True)
        neb2.interpolate(method="idpp")

        atom_configs = atom_config1 + atom_config2[1:]

    neb = NEB(atom_configs, climb=False, remove_rotation_and_translation=True)
    neb_opt = NEBOptimizer(neb, logfile=log)
    fmax_val = 1.0
    is_success = True
    while is_success and fmax_val >= 0.5:
        atom_configs_backup = deepcopy(atom_configs)
        is_success = neb_opt.run(fmax=fmax_val, steps=max_steps)
        max_force = neb_opt.get_residual()
        if max_force > fmax_val:
            is_success = False
        fmax_val /= 2
    if not is_success:
        print('NEB failed to converge')
        return atom_configs_backup, is_success
    
    neb = NEB(atom_configs, climb=True, remove_rotation_and_translation=True)
    neb_opt = NEBOptimizer(neb, logfile=log)
    fmax_val = 0.4
    is_success = True
    while is_success and fmax_val >= 0.05:
        atom_configs_backup = deepcopy(atom_configs)
        is_success = neb_opt.run(fmax=fmax_val, steps=max_steps)
        max_force = neb_opt.get_residual()
        if max_force > fmax_val:
            is_success = False
        fmax_val /= 2
    if is_success:
        return atom_configs, is_success
    else:
        print('NEB failed to converge')
        return atom_configs_backup, is_success

def Sella_Opt(x, trans_pos_init, calculator, log=None, max_steps=1000):
    # print log in cmd, set log = '-'
    # save log in files, set log = filepath
    pred_trans = construct_atoms(x, trans_pos_init)
    pred_trans_backup = deepcopy(pred_trans)
    pred_trans.calc = calculator

    dyn = Sella(pred_trans, logfile=log)
    try:
        converged = dyn.run(0.05, max_steps)
        return pred_trans, converged
    except:
        pred_trans = pred_trans_backup
        return pred_trans, False

def Sella_IRC(x, trans_pos_init, calculator, log=None, max_steps=1000):
    # print log in cmd, set log = '-'
    # save log in files, set log = filepath
    pred_trans = construct_atoms(x, trans_pos_init)
    pred_trans.calc = calculator
    pred_forward = deepcopy(pred_trans)
    pred_backward = deepcopy(pred_trans)
    opt_forward = IRC(pred_forward, logfile=log)
    opt_reverse = IRC(pred_backward, logfile=log)
    try:
        converged_f = opt_forward.run(fmax=0.1, steps=max_steps, direction='forward')
        converged_r = opt_reverse.run(fmax=0.1, steps=max_steps, direction='reverse')
        BFGS(pred_forward, maxstep=0.005, logfile=log).run(fmax=0.05, steps=max_steps)
        BFGS(pred_backward, maxstep=0.005, logfile=log).run(fmax=0.05, steps=max_steps)
        return pred_forward, pred_backward, (converged_f and converged_r)
    except:
        return pred_forward, pred_backward, False