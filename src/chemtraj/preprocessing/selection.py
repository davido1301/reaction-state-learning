import MDAnalysis as mda 
from pathlib import Path 

def select_protein(top: Path, traj: Path, selection = "not resname SOL and not name LA and not name NA") -> mda.AtomGroup: 
    """ Select only the protein part of the system 
    - Solvent 
    - Na 
    """
    universe = mda.Universe(top, traj)

    protein_system = universe.select_atoms(selection)
    return protein_system

