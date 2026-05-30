import MDAnalysis as mda 
from ase.io import read 
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from chemtraj.utils.io import dataframe_from_extxyz 
from chemtraj.analysis.filter import residues_near_reacting_atoms 
from chemtraj.preprocessing.selection import select_protein
from chemtraj.representations.internal_coord import get_dihedrals

DB = "data/processed/metad_10_labeled.extxyz"
TOP = "data/unprocessed/metad_10.gro"
TRAJ = "data/unprocessed/metad_10.xtc" 


df = dataframe_from_extxyz(DB) 

universe = mda.Universe(TOP,TRAJ) 
protein_system = select_protein(TOP, TRAJ) 



ress = residues_near_reacting_atoms(
    universe=universe,
    reactive_selection="index 24 59 526"
)

id_res = []
for _ in ress:
    id_res.append(_[1])

phi_angles = get_dihedrals(universe, id_res)

for resid, phi_angle_value in zip(id_res, phi_angles.T):
    df[f"phi_res_{resid}"] = phi_angle_value 

phi_coles = [col for col in df.columns if col.startswith("phi_res")] 


X = df.iloc[:, 2:35].values 
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


exit()
plt.savefig("/home/davido/Projects/HiWi_Maike/reaction-state-learning/src/chemtraj/results/lda/lda_filter8.png", dpi=300)


