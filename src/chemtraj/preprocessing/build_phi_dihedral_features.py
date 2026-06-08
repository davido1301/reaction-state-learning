import MDAnalysis as mda 
import pandas as pd 
from pathlib import Path 

from chemtraj.analysis.filter import residues_near_reacting_atoms
from chemtraj.representations.internal_coord import get_dihedrals
from chemtraj.preprocessing.selection import extract_res_id_and_name

TOP = "data/unprocessed/combined_metad/protein_GSSG.gro"
TRAJ = "data/unprocessed/combined_metad/all_walkers_no_water.xtc"
LABELS = "data/processed/metad_all_labels.csv"

OUT = Path("data/processed/features/phi_dihedrals_all_metad.parquet") 
OUT.parent.mkdir(parents=True, exist_ok=True)


universe = mda.Universe(TOP,TRAJ) 

ress = residues_near_reacting_atoms(
    universe=universe,
        reactive_selection="index 41 59 526"
)

id_res, res_names = extract_res_id_and_name(ress)

print("Sel resids", id_res) 
print("Number of res ", len(id_res))

phi_angles = get_dihedrals(universe, id_res)
labels_df = pd.read_csv(LABELS) 

print("Phi angle shape", phi_angles.shape)
print("labels df shape", labels_df.shape) 

feature_df = labels_df[["frame_id", "label"]].copy()
for resid, resname , phi_angle_values in zip(id_res, res_names, phi_angles.T):
    feature_df[f"phi_res_{resid}_{resname}"] = phi_angle_values
feature_df.to_parquet(OUT, index=False) 
print("Saved lol")
print(feature_df.head()) 
print("shape df", feature_df.shape) 


    
