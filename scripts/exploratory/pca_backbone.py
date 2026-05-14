import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path 
import seaborn as sns

import MDAnalysis as mda 
from MDAnalysis.analysis import pca, align 

from chemtraj.preprocessing.selection import select_protein 

from ase.io import read

DB = "data/processed/metad_10_labeled.extxyz"

TOP = "data/unprocessed/metad_10.gro"
TRAJ = "data/unprocessed/metad_10.xtc" 
result_dir = Path("src/chemtraj/results/pca/")

u = mda.Universe(TOP,TRAJ)
aligner = align.AlignTraj(u, u, select="backbone", in_memory=True).run()

pc = pca.PCA(u, select="backbone",
             align=True,
             n_components=None).run()

backbone = u.select_atoms("backbone") 
n_bb = len(backbone) 
print(f"There are {n_bb} backbone atoms") 


print("Shape components", pc.p_components.shape) 
print(f"PC1 {pc.variance[0]}")

for i in range(3):
    print(f"Cumulated variance {pc.cumulated_variance[i]}")

plt.plot(pc.cumulated_variance[:5])
plt.xlabel("Principal component")
plt.ylabel("Cumlative variance") 

plot_path = result_dir / "pca_backbone_5comp.png"
# plt.savefig(plot_path, dpi=300)

transformed = pc.transform(backbone, n_components=3)
print(transformed.shape) 
print(transformed[0]) 


atoms_list = read(DB, index=":")
rows = [] 
for i, atoms in enumerate(atoms_list):
    rows.append({
                "frame_id": i,
                "label": atoms.info.get("label"),
    })

labels_df = pd.DataFrame(rows) 
print(labels_df.head())
label = labels_df["label"]

pca_df = pd.DataFrame({
                      "frame_id": np.arange(transformed.shape[0]),
                      "PC1": transformed[:, 0],
                      "PC2": transformed[:, 1],
                      "PC3": transformed[:, 2],
                      "label": label
}) 


unique_labels = sorted(pca_df["label"].dropna().unique())
palette = dict(zip(
               unique_labels,
               sns.color_palette("tab10", n_colors=len(unique_labels))
))



g = sns.PairGrid(pca_df, hue="label", palette=palette)

g.map(plt.scatter, marker=".")
g.add_legend(title="Label")
plot_path = result_dir / "pca_comp_over_label.png"

plt.savefig(plot_path, dpi=300)



