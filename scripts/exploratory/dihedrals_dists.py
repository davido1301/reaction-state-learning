import MDAnalysis as mda 
from MDAnalysis.analysis.dihedrals import Dihedral
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from ase.io import read
import seaborn as sns 
from pathlib import Path

#  DB = "data/processed/metad_10_labeled.extxyz"
#  TOP = "data/unprocessed/metad_10.gro"
#  TRAJ = "data/unprocessed/metad_10.xtc"

RESULTS = Path("src/chemtraj/results/dihedrals_and_SS_dist/")

#  u = mda.Universe(TOP,TRAJ)
#  # Residues of interest
#  residues = {
#      "Cys35": u.residues[34],
#      "Pro36": u.residues[35],
#      "Tyr37": u.residues[36],
#      "Cys38": u.residues[37],
#      "Ser80": u.residues[79],
#      "Val81": u.residues[80],
#  }

#  # Build dihedral objects once
#  phi_objects = {}
#  for name, res in residues.items():
#      phi_sel = res.phi_selection()
#      phi_objects[name] = phi_sel.dihedral

#  # Atom selections for S-S distance
#  s1 = u.select_atoms("resid 35 and name SG")
#  s2 = u.select_atoms("resid 2 and name SG")
#  s3 = u.select_atoms("resid 5 and name SG")

#  # Storage
#  phi_data = {name: [] for name in residues}
#  dists1 = []
#  dists2 = []

#  # Single trajectory pass
#  for ts in u.trajectory:
#      for name, dih in phi_objects.items():
#          phi_data[name].append(dih.value())

#      dist1 = np.linalg.norm(s1.positions[0] - s2.positions[0])
#      dist2 = np.linalg.norm(s2.positions[0] - s3.positions[0])
#      dists1.append(dist1 / 10)  # Angstrom to nm
#      dists2.append(dist2 / 10)  # Angstrom to nm

#  # Rolling averages
#  window_size = 100
#  dists1_smooth = pd.Series(dists1).rolling(window=window_size).mean()
#  dists2_smooth = pd.Series(dists2).rolling(window=window_size).mean()
#  phi_smooth = {
#      name: pd.Series(values).rolling(window=window_size).mean()
#      for name, values in phi_data.items()
#  }

#  df = pd.DataFrame(
#      {
#          "phi_Cys35": phi_smooth["Cys35"],
#          "phi_Pro36": phi_smooth["Pro36"],
#          "phi_Tyr37": phi_smooth["Tyr37"],
#          "phi_Cys38": phi_smooth["Cys38"],
#          "phi_Ser80": phi_smooth["Ser80"],
#          "phi_Val81": phi_smooth["Val81"],
#          "ss_dists1": dists1,
#          "ss_dists2": dists2,
#      }
#  )

#  atoms_list = read(DB, index=":")
#  rows = []
#  for i, atoms in enumerate(atoms_list):
#      rows.append({
#                  "frame_id": i,
#                  "label": atoms.info.get("label")
#      })

#  labels_df = pd.DataFrame(rows)
#  label = labels_df["label"]

#  df["label"] = labels_df["label"]

out_parquet = RESULTS / "dihedrals_and_SS_dist.parquet"
#  df.to_parquet(out_parquet, index=False, compression="snappy")
#  print(f"Saved to {out_parquet}")

df = pd.read_parquet(out_parquet)

df_plot = df.dropna()
sns.set_style("ticks")
g = sns.pairplot(
    df_plot,
    hue="label",
    vars=["phi_Cys35", "phi_Pro36", "phi_Tyr37", "phi_Cys38", "phi_Ser80",
          "phi_Val81", "ss_dists1", "ss_dists2"],
    kind="scatter",
    diag_kind="hist",
    plot_kws={"s": 20, "alpha": 0.6, "edgecolor": "none"},
    diag_kws={"bins": 25}
)

plt.show()
