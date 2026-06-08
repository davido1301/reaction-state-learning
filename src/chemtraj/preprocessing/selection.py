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

def extract_res_id_and_name(residues) -> List[str]:
    """ Take residues and extraxt ids and names for further processing
    return 2 lists """ 

    resindexs = [] 
    resnames = []
    for item in residues:
        resindex = item[1]
        resname = item[2] 

        resindexs.append(resindex)
        resnames.append(resname) 

    return resindexs, resnames
