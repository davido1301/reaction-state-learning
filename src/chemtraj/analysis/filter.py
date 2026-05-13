import MDAnalysis as mda 
from chemtraj.preprocessing.selection import select_protein
from MDAnalysis.lib.distances import distance_array 


TOP = "data/unprocessed/metad_10.gro"
TRAJ = "data/unprocessed/metad_10.xtc" 



def residues_near_reacting_atoms(
        universe: mda.Universe,
        reactive_selection: str, 
        protein_selection: str = "protein",
        cutoff: float = 8.0,
) -> list[tuple[str, int, str]]:
    """ Find protein residues near reactive atoms over all frames
    """ 
    protein = select_protein(TOP, TRAJ) 
    reactive_atoms = universe.select_atoms(reactive_selection)

    if len(reactive_atoms) != 3: 
        raise ValueError(f"Expected 3 atoms, got {len(reactive_atoms)}")

    nearby = set() 

    for ts in universe.trajectory:
        for residue in protein.residues:
            residue_atoms = residue.atoms.select_atoms("not name H")

            distances = distance_array(
                residue_atoms.positions,
                reactive_atoms.positions,
                box=ts.dimensions,
            )
            if distances.min() < cutoff:
                nearby.add((residue.segid, int(residue.resid), residue.resname)) 

    return sorted(nearby) 


universe = mda.Universe(TOP, TRAJ) 
nearby_res = residues_near_reacting_atoms(
    universe=universe,
    reactive_selection="index 24 59 526", # S S S 
    )

id_res = []
for _ in nearby_res: 
    id_res.append(_[1])

print(id_res)
