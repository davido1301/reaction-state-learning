import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path 

import MDAnalysis as mda 
from MDAnalysis.analysis import pca, align 

from chemtraj.preprocessing.selection import select_protein 

TOP = "data/unprocessed/metad_10.gro"
TRAJ = "data/unprocessed/metad_10.xtc" 
result_dir = Path("src/chemtraj/results/pca/")

u = mda.Universe(TOP,TRAJ)
def pca_cum_comp()
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
plt.savefig(plot_path, dpi=300)





