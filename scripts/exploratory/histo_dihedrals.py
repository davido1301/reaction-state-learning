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
import seaborn as sns 
from pathlib import Path


DB = "data/processed/metad_10_labeled.extxyz"
TOP = "data/unprocessed/metad_10.gro"
TRAJ = "data/unprocessed/metad_10.xtc" 

RESULTS_HIST = Path("src/chemtraj/results/histo_dihedrals_labeled/")

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
        print(f"No phi res id {id}") 

phi_angles = get_dihedrals(universe, id_res) 


atoms_list = read(DB, index=":")
rows = [] 
for i, atoms in enumerate(atoms_list):
    rows.append({
                "frame_id": i, 
                "label": atoms.info.get("label")
    })

labels_df = pd.DataFrame(rows) 
label = labels_df["label"]

print(phi_angles.shape)
print(len(labels_df))
print(len(id_res))


for resid, phi_angle_value in zip(id_res, phi_angles.T):
    labels_df[f"phi_res_{resid}"] = phi_angle_value 

phi_cols = [col for col in labels_df.columns if col.startswith("phi_res")]

for feature in phi_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(data=labels_df,
    x=feature,
    hue="label",
    kde=True,
    element="step",
    stat="density",
)
    filename= f"{feature}_histogram.png"
    save_path = RESULTS_HIST / filename
    plt.title(feature)
    plt.xlabel(f"Dihedral angle {feature}")
    plt.ylabel("Density") 
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
