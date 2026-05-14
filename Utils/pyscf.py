import numpy as np
from ase.calculators.calculator import (Calculator, CalculatorError, 
                    CalculatorSetupError, all_changes, all_properties, kpts2mp, FileIOCalculator)
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo
import pyscf

AU2KCALMOL = 627.509608
AU2EV = 27.2114
BOHR = 0.52917721092

def ase_atoms_to_pyscf(ase_atoms):
    '''Convert ASE atoms to PySCF atom.

    Note: ASE atoms always use A.
    '''
    return [[atom.symbol, atom.position] for atom in ase_atoms]

atoms_from_ase = ase_atoms_to_pyscf

def calculate_efh(
    atom,
    f=True,
    hess=False,
    return_metrics=False,
    xc="wb97x",
    basis="631g*",
    device='gpu',
):
    geomfile = ase_atoms_to_pyscf(atom)
    spin = 0
    mol = pyscf.M(
        atom=geomfile,
        unit="Ang",
        basis=basis,
    )
    mol.build()

    if device == 'gpu' or device == 'cuda':
        mf = dft.RKS(mol).to_gpu() if not spin else dft.UKS(mol).to_gpu()
    else:
        mf = dft.RKS(mol) if not spin else dft.UKS(mol)
    mf.xc = xc
    
    mf.conv_tol = 1e-6
    # mf.damp = 0.2
    mf.max_cycle = 200
    mf.max_memory = 32000
    mf.run()

    force = None
    force_rms = np.nan
    if mf.converged and f:
        force = mf.nuc_grad_method().kernel() * -1.0 / BOHR  * AU2EV
        force_rms = np.sqrt(np.mean(force**2))
        print("force rms (ev/A): ", force_rms)

    hessian = None
    if hess:
        hessian = mf.Hessian().kernel() * AU2EV / BOHR / BOHR
        freq_info = thermo.harmonic_analysis(mf.mol, hessian)
        print("freq: ", freq_info["freq_wavenumber"])

    if return_metrics:
        return (
            mf,
            force,
            hessian,
            force_rms,
            freq_info
        )
    return mf, force, hessian

# K-point sampling
def make_kpts(cell, nks):
    raise DeprecationWarning('Use cell.make_kpts(nks) instead.')

def count_negative_eig(x: list):
    count = 0
    for _x in x:
        if _x.imag > 0:
            count += 1
    return count

class PyscfCalculator(Calculator):
    def __init__(
        self,
        xc="wb97x",
        basis="631g*",
        implemented_properties=None,
        device=None,
        **kwargs,
    ):
        if not implemented_properties:
            implemented_properties = ["energy", "forces", "hessian"]
        self.implemented_properties = implemented_properties
        pin_memory = (device == 'cuda')
        self.xc = xc
        self.basis = basis

        super().__init__(**kwargs)

        # self.atoms_converter = atoms_converter
        if device:
            self.device = device

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
            geomfile = ase_atoms_to_pyscf(atoms)
            spin = 0
            mol = pyscf.M(
                atom=geomfile,
                unit="Ang",
                basis=self.basis,
            )
            mol.build()

            if self.device == 'gpu' or self.device == 'cuda':
                mf = dft.RKS(mol).to_gpu() if not spin else dft.UKS(mol).to_gpu()
            else:
                mf = dft.RKS(mol) if not spin else dft.UKS(mol)
            mf.xc = self.xc
            mf.conv_tol = 1e-6
            # mf.damp = 0.2
            mf.max_cycle = 200
            mf.max_memory = 32000
            mf.run()

            force = None
            if mf.converged and "forces" in properties:
                force = mf.nuc_grad_method().kernel()

            hessian = None
            if "hessian" in properties:
                hessian = mf.Hessian().kernel()

            if "energy" in properties:
                self.results['energy'] = mf.e_tot * AU2EV
            if "forces" in properties:
                self.results['forces'] = force * -1.0 / BOHR * AU2EV
            if "hessian" in properties:
                self.results['hessian'] = hessian * AU2EV / BOHR / BOHR

                
