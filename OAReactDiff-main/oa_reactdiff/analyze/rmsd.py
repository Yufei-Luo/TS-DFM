from typing import List
import numpy as np

from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import (
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
    KabschMatcher,
)
from pymatgen.io.xyz import XYZ

from torch import Tensor


def xh2pmg(xh):
    mol = Molecule(
        species=xh[:, -1].long().cpu().numpy(),
        coords=xh[:, :3].cpu().numpy(),
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd

def rmsd_dmae_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        aligned_mol, rmsd = bfm.fit(mol2)
        coord1 = mol1.cart_coords
        coord2 = aligned_mol.cart_coords
        diff1 = coord1[:, np.newaxis, :] - coord1[np.newaxis, :, :]
        diff2 = coord2[:, np.newaxis, :] - coord2[np.newaxis, :, :]
        dists1 = np.linalg.norm(diff1, ord=2, axis=-1)
        dists2 = np.linalg.norm(diff2, ord=2, axis=-1)
        dmae = np.mean(np.abs(dists1 - dists2))
        return rmsd, dmae 
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        aligned_mol, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            aligned_mol, rmsd = bfm.fit(mol2)
    coord1 = mol1.cart_coords
    coord2 = aligned_mol.cart_coords
    diff1 = coord1[:, np.newaxis, :] - coord1[np.newaxis, :, :]
    diff2 = coord2[:, np.newaxis, :] - coord2[np.newaxis, :, :]
    dists1 = np.linalg.norm(diff1, ord=2, axis=-1)
    dists2 = np.linalg.norm(diff2, ord=2, axis=-1)
    dmae = np.mean(np.abs(dists1 - dists2))
    return rmsd, dmae, aligned_mol


def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality=False,
    threshold=0.5,
    same_order=False,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)
    rmsd = rmsd_core(mol1, mol2, threshold)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd

def pymatgen_rmsd_dmae(
    mol1,
    mol2,
    ignore_chirality=False,
    threshold=0.5,
    same_order=False,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)
    rmsd, dmae, aligned_mol = rmsd_dmae_core(mol1, mol2, threshold, same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect, dmae_reflect, aligned_mol_reflect = rmsd_dmae_core(mol1, mol2_reflect, threshold)
        if rmsd < rmsd_reflect:
            return rmsd, dmae, aligned_mol
        else:
            return rmsd_reflect, dmae_reflect, aligned_mol_reflect
    return rmsd, dmae, aligned_mol
    


def batch_rmsd(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    idx: int = 1,
    threshold=0.5,
):
    rmsds = []
    out_samples_use = out_samples[idx]
    xh_use = xh[idx]
    nodes = fragments_nodes[idx].long().cpu().numpy()
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(nodes):
        end_ind += natoms
        mol1 = xh2pmg(out_samples_use[start_ind:end_ind])
        mol2 = xh2pmg(xh_use[start_ind:end_ind])
        try:
            rmsd = pymatgen_rmsd(mol1, mol2, ignore_chirality=True, threshold=threshold)
        except:
            rmsd = 1.0
        rmsds.append(min(rmsd, 1.0))
        start_ind = end_ind
    return rmsds

def batch_rmsd_dmae(    
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    idx: int = 1,
    threshold=0.5,
):
    rmsds = []
    dmaes = []
    aligned_mols = []
    out_samples_use = out_samples[idx]
    xh_use = xh[idx]
    nodes = fragments_nodes[idx].long().cpu().numpy()
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(nodes):
        end_ind += natoms
        mol1 = xh2pmg(xh_use[start_ind:end_ind])
        mol2 = xh2pmg(out_samples_use[start_ind:end_ind])
        try:
            rmsd, dmae, aligned_mol = pymatgen_rmsd_dmae(mol1, mol2, ignore_chirality=True, threshold=threshold)
        except:
            rmsd = 1.0
            dmae = 1.0
            aligned_mol = mol2
        rmsds.append(min(rmsd, 1.0))
        dmaes.append(min(dmae, 1.0))
        aligned_mols.append(aligned_mol)
        start_ind = end_ind
    return rmsds, dmaes, aligned_mols