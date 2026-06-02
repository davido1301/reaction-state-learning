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
from chemtraj.utils.io import dataframe_from_extxyz



# For that .yaml config missing needs to be added later on see issues
DB = "data/processed/metad_10_labeled.extxyz"
TOP = "data/unprocessed/metad_10.gro"
TRAJ = "data/unprocessed/metad_10.xtc" 
REACTIVE_SELECTION = "index 24 59 529" 

universe = mda.Universe(TOP,TRAJ)
protein_system = select_protein(TOP,TRAJ)

residues = residues_near_reacting_atoms(
        universe=universe,
        reactive_selection=REACTIVE_SELECTION
)

resid, resname = extract_res_id_and_name(residues)

phi_angles = get_dihedrals(universe, resid) 

# constructing df 
df = dataframe_from_extxyz(DB) 
for residx, resnamex, phi_angle_value in zip(resid, resname, phi_angles.T):
    df[f"phi_res_{residx}_{resnamex}"] = phi_angle_value 

phi_columns = [col for col in df.columns if col.startswith("phi_res_")]

# must go into lda file

X = df.iloc[:, 2:35].values # wonky
y = df.iloc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y) 

# can catch artifacts otherwise bc periodic 

X_rad = np.deg2rad(X) 
X_sin = np.sin(X_rad) 
X_cos = np.cos(X_rad) 

X_trig = np.concatenate([X_sin, X_cos], axis=1) 



X_train, X_test, y_train, y_test = train_test_split(X_trig,y, test_size=0.2) 

lda = LinearDiscriminantAnalysis(n_components=2) 
X_train = lda.fit_transform(X_train, y_train) 
X_test = lda.transform(X_test) 



tmp_Df = pd.DataFrame(X_train, columns=['LDA Component 1','LDA Component 2'])
tmp_Df['Class']=y_train

sns.FacetGrid(tmp_Df, hue ="Class",
              height = 6).map(plt.scatter,
                              'LDA Component 1',
                              'LDA Component 2')

plt.legend(loc='upper right')
plt.show()


phi_cols_trig = (
    [f"{col}_sin" for col in phi_coles] + 
    [f"{col}_cos" for col in phi_coles]
)
lda1_weights = pd.Series(
    lda.scalings_[:, 0],
    index=phi_cols_trig
).sort_values(key=abs, ascending=False)

print("Lda1 weights",lda1_weights)

lda2_weights = pd.Series(
        lda.scalings_[:, 1],
        index=phi_cols_trig
).sort_values(key=abs, ascending=False)

print("LDA2 weights", lda2_weights)
