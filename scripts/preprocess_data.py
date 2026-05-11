import MDAnalysis as mda 
import numpy as np
from ase import Atoms 
from ase.io import write
from pathlib import Path 


# Getting the base data into a usable format
# Labelling with criterion and converting to extxyz 

OUTFILE = Path("data/processed/metad_10_labeled.extxyz")

def label_criterion(universe) -> str: 
    """ Criterion is the distance between the S atoms 
        with a small tolerance for the transition state
        d1 > d2 -- Product 
        d1 < d2 -- Educt 
        d1 ~ d2 -- Transition state """
    # mind the index based counting starting from 0 

    s1 = universe.select_atoms("index 24")[0]
    s2 = universe.select_atoms("index 59")[0]
    cym = universe.select_atoms("index 526")[0]
    d1 = np.linalg.norm(cym.position - s1.position)
    d2 = np.linalg.norm(s1.position - s2.position) 

    if abs(d1-d2) <= 0.1: 
        label = "TS"
    elif d1 > d2: 
        label = "EDUCT"
    elif d1 < d2:
        label = "PRODUCT"
    return label


def check_if_traj_ok(universe):
    """ Checking whether the trajectory contains
        any inf or nan values """ 
    for ts in universe.trajectory:
        pos = protein_system.positions
        if np.isnan(pos).any():
            print(f"NAN in frame {ts.frame}")
            break 
        if np.isinf(pos).any():
            print(f"Inf in frame {ts.frame}") 
            break
    print("Traj is healthy")
     
def atomname_to_element(name: str) -> str:
    """ This is to transform the GMX atom names into usable format for ase """ 
    name = str(name).strip()
    name = name.lstrip("0123456789")

    first = name[0].upper()
    if name.upper() == "NA":
        return "Na"
    if name.upper() == "CL":
        return "Cl"
    if first in {"C", "H", "N", "O", "S"}:
        return first

    raise ValueError(f"Cannot infer element name from {name}")

u = mda.Universe("data/unprocessed/metad_10.gro", "data/unprocessed/metad_10.xtc")
protein_system = u.select_atoms("not resname SOL and not name LA")

print(u.atoms.n_atoms)
print(protein_system.atoms.n_atoms)
print(set(protein_system.residues.resnames))
check_if_traj_ok(u)

# collecting frames
frames = [] 

# counting states 
t_state = 0 
educts = 0 
products = 0 

for ts in u.trajectory:
    pos = protein_system.positions.copy()
    symbols = [atomname_to_element(name) for name in protein_system.names] 
    label = label_criterion(u)
    atoms = Atoms(
        symbols=symbols,
        positions=pos,
        cell=ts.dimensions[:3],
        pbc=True
    )
    atoms.arrays["atom_names"] = protein_system.names.astype(str)
    atoms.arrays["res_names"] = protein_system.resnames.astype(str)
    atoms.arrays["resids"] = protein_system.resids.astype(int) 

    atoms.info["label"] = label
    atoms.info["frame"] = ts.frame
    frames.append(atoms) 

    if label == "TS":
        t_state += 1 
    if label == "EDUCT":
        educts += 1 
    if label == "PRODUCT":
        products += 1 
    

print(f"Number of educts {educts}, products: {products}, and transition states: {t_state}")

if not OUTFILE.exists():
    write(OUTFILE, frames) 
    print("Files written!")
else: 
    print("File already there")


