#!/usr/bin/env python3
"""
Protein Dataset Preparation Script
Creates a pickled protein dataset from atom3d MSP dataset for easier loading in experiments
"""

import sys
import atom3d.datasets
from torch.utils.data import Dataset
import pickle
import torch
from tqdm import tqdm
import os
from pathlib import Path

# Add src to path to import our models
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load atom3d dataset
dataset = atom3d.datasets.load_dataset("msp/raw/MSP/data", "lmdb")

num_entries = len(dataset)

# Extended periodic table elements (covering more atoms that might appear in proteins)
periodic_table_elements = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

# Full amino acid names (3-letter codes)
amino_acids = [
    "ALA",  # Alanine
    "ARG",  # Arginine
    "ASN",  # Asparagine
    "ASP",  # Aspartic acid
    "CYS",  # Cysteine
    "GLU",  # Glutamic acid
    "GLN",  # Glutamine
    "GLY",  # Glycine
    "HIS",  # Histidine
    "ILE",  # Isoleucine
    "LEU",  # Leucine
    "LYS",  # Lysine
    "MET",  # Methionine
    "PHE",  # Phenylalanine
    "PRO",  # Proline
    "SER",  # Serine
    "THR",  # Threonine
    "TRP",  # Tryptophan
    "TYR",  # Tyrosine
    "VAL",  # Valine
    # Additional residues found in some proteins
    "SEC",  # Selenocysteine
    "PYL",  # Pyrrolysine
]

atom_to_index = {atom: idx for idx, atom in enumerate(periodic_table_elements)}
residue_to_idx = {res: idx for idx, res in enumerate(amino_acids)}


def build_residue_seq(resname, residue):
    """Build residue sequence from residue names and IDs"""
    if len(resname):
        result = [resname[0]]
    else:
        result = []
    for i in range(1, len(resname)):
        # Add residue to residue sequence if it's never been seen (in which case its id is higher)
        if residue[i] > residue[i - 1]:
            result.append(resname[i])
    return result


def trim_or_keep(atom_seq, residue_seq, counts, coordinates):
    """Trim sequences to maximum length or keep as is if shorter"""
    seq_len = residue_seq.size(0)

    if seq_len <= 100:
        return atom_seq, residue_seq, counts, coordinates
    
    start_idx = torch.randint(0, seq_len - 100 + 1, (1,)).item()
    short_residue_seq = residue_seq[start_idx : start_idx + 100]
    short_counts = counts[start_idx : start_idx + 100]
    atom_start_index = torch.sum(counts[:start_idx])
    atom_end_index = torch.sum(counts[: start_idx + 100])
    short_atom_seq = atom_seq[atom_start_index:atom_end_index]
    short_coordinates = coordinates[atom_start_index:atom_end_index]
    return short_atom_seq, short_residue_seq, short_counts, short_coordinates


def build_protein_list(dataset):
    """Build list of protein data from atom3d dataset"""
    protein_data = []
    for i in tqdm(range(len(dataset)), desc="Processing proteins"):
        p = dataset[i]["original_atoms"]
        # Only keep the main chain
        p = p[p["chain"] == p["chain"][0]]
        protein = {}
        
        # Convert atoms to indices
        atom_seq = torch.tensor([atom_to_index[atom] for atom in p["element"]])
        resname = p["resname"]
        residue = p["residue"]
        
        # Build residue sequence
        residue_seq = torch.tensor([
            residue_to_idx[res] for res in build_residue_seq(resname=resname, residue=residue)
        ])
        
        # Count atoms per residue
        _, counts = torch.unique(torch.tensor(residue), return_counts=True)
        
        # Extract coordinates
        atom_coords = []
        for j in range(len(p["x"])):
            x, y, z = p["x"][j], p["y"][j], p["z"][j]
            atom_coords.append(torch.tensor([x, y, z]))
        coordinates = torch.stack(atom_coords)
        
        # Trim sequences if too long
        short_atom_seq, short_residue_seq, short_counts, short_coordinates = (
            trim_or_keep(atom_seq, residue_seq, counts, coordinates)
        )

        protein["atom_sequence"] = short_atom_seq
        protein["amino_acid_sequence"] = short_residue_seq
        protein["counts"] = short_counts
        protein["coordinates"] = short_coordinates
        protein_data.append(protein)
    
    return protein_data


class ProteinDataset(Dataset):
    """Dataset class for protein structure prediction"""
    def __init__(self, protein_data):
        self.protein_data = protein_data  # protein_data is a list of dicts

    def __len__(self):
        return len(self.protein_data)

    def __getitem__(self, idx):
        protein = self.protein_data[idx]
        atom_sequence = protein["atom_sequence"]
        amino_acid_sequence = protein["amino_acid_sequence"]
        coordinates = protein["coordinates"].float()  # tensor of (x, y, z) coordinates
        counts = protein["counts"]
        return (atom_sequence, amino_acid_sequence, counts), coordinates


def get_max_atom_seq_len(dataset):
    """Get maximum atom sequence length in dataset"""
    max_len = 0
    for protein in dataset:
        seq_len = protein[0][0].size(0)
        if seq_len > max_len:
            max_len = seq_len
    return max_len


def pickle_dataset():
    """Create and save the pickled protein dataset"""
    print(f"Processing {num_entries} proteins from atom3d dataset...")
    protein_data = build_protein_list(dataset)
    d = ProteinDataset(protein_data)
    
    max_len = get_max_atom_seq_len(d)
    print(f"Maximum atom sequence length: {max_len}")
    print(f"Total proteins processed: {len(d)}")

    # Remove existing file if it exists
    output_path = "protein_dataset.pkl"
    try:
        os.remove(output_path)
        print(f"Removed existing {output_path}")
    except OSError:
        pass
    
    # Save new dataset
    with open(output_path, "wb") as f:
        pickle.dump(d, f)
    
    print(f"Protein dataset saved to {output_path}")
    print(f"Dataset contains {len(d)} proteins")
    print(f"Atom vocabulary size: {len(periodic_table_elements)}")
    print(f"Residue vocabulary size: {len(amino_acids)}")


if __name__ == "__main__":
    pickle_dataset()
