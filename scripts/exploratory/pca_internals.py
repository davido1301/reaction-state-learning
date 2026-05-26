import MDAnalysis as mda 
from MDAnalysis.analysis.dihedrals import Dihedral
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from ase.io import read
from chemtraj.preprocessing.selection import select_protein
from chemtraj.analysis.filter import residues_near_reacting_atoms 
from chemtraj.analysis.pca import make_pca_df, pca_pair_plot
from chemtraj.representations.internal_coord import get_dihedrals
DB = "data/processed/metad_10_labeled.extxyz"
TOP = "data/unprocessed/metad_10.gro"
TRAJ = "data/unprocessed/metad_10.xtc" 

universe = mda.Universe(TOP,TRAJ)
protein_system = select_protein(TOP,TRAJ)

ress = residues_near_reacting_atoms(
    universe=universe,
    reactive_selection="index 24 59 526",
)

id_res = [] 
for _ in ress:
    id_res.append(_[1])


phi_groups = [] 

for id in id_res:
    res = universe.residues[id]
    ag = res.phi_selection() 
    if ag is not None: 
        phi_groups.append(ag) 
    else:
        print(f"No phi for res id {id}") 

phi = Dihedral(phi_groups).run()
phi_angles = phi.results.angles 


test = get_dihedrals(universe, id_res)
print(test[0])


exit()
X = np.column_stack([
    np.sin(phi_angles),
    np.cos(phi_angles),
])

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=5) 
scores = pca.fit_transform(X_scaled) 

pca_df = make_pca_df(scores) 
print(pca_df.head()) 

atoms_list = read(DB, index=":")
rows = []
for i, atoms in enumerate(atoms_list):
    rows.append({
                "frame_id": i,
                "label": atoms.info.get("label")
    })
labels_df=pd.DataFrame(rows) 
label = labels_df["label"] 


pca_pair_plot(pca_df, label)

