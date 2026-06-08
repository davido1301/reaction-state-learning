import MDAnalysis as mda 
from MDAnalysis.analysis.dihedrals import Dihedral
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from ase.io import read
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from chemtraj.preprocessing.selection import select_protein, extract_res_id_and_name
from chemtraj.analysis.filter import residues_near_reacting_atoms 
from chemtraj.analysis.pca import make_pca_df, pca_pair_plot
from chemtraj.representations.internal_coord import get_dihedrals
from chemtraj.utils.io import dataframe_from_extxyz, read_config
import yaml
import seaborn as sns
from chemtraj.analysis.lda import run_lda, print_lda_weights, plot_lda



PATH_YAML = "src/chemtraj/configs/test.yaml"

config = read_config(PATH_YAML)
universe = mda.Universe(config.top,config.traj)
protein_system = select_protein(config.top,config.traj)
residues = residues_near_reacting_atoms(
        universe=universe,
        reactive_selection=config.reactive_selection,
)

resid, resname = extract_res_id_and_name(residues)
phi_angles = get_dihedrals(universe, resid) 

# constructing df 
df = dataframe_from_extxyz(config.db) 

for residx, resnamex, phi_angle_value in zip(resid, resname, phi_angles.T):
    df[f"phi_res_{residx}_{resnamex}"] = phi_angle_value 

phi_columns = [col for col in df.columns if col.startswith("phi_res_")]

# Selection for LDA atm 
X = df.iloc[:, 2:35].values # wonky
y = df.iloc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
# can catch artifacts otherwise bc periodic 
X_rad = np.deg2rad(X) 
X_sin = np.sin(X_rad) 
X_cos = np.cos(X_rad) 
X_trig = np.concatenate([X_sin, X_cos], axis=1) 

lda, X_test, X_train, y_train = run_lda(X_trig, y)

print(config.results_dir)
filename = "lda_plot.png" 
save_dir= config.results_dir / filename

plot_lda(X_train, y_train, saving=True, results_dir=save_dir)


phi_cols_trig = (
    [f"{col}_sin" for col in phi_columns] + 
    [f"{col}_cos" for col in phi_columns]
)

print_lda_weights(lda, phi_cols_trig)
