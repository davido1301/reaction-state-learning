import MDAnalysis as mda 
from MDAnalysis.analysis.dihedrals import Dihedral
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from ase.io import read
import seaborn as sns 
from pathlib import Path

RESULTS = Path("src/chemtraj/results/dihedrals_and_SS_dist/")

out_parquet = RESULTS / "dihedrals_and_SS_dist_1.parquet"

df = pd.read_parquet(out_parquet)

data = np.loadtxt("data/processed/water_dists_1.out", skiprows=1)
cym_water_dist = data[:, 1]
s1_water_dist = data[:, 2]
s2_water_dist = data[:, 3]

window_size = 100
cym_water_dist_smooth = pd.Series(cym_water_dist).rolling(window=window_size).mean()
s1_water_dist_smooth = pd.Series(s1_water_dist).rolling(window=window_size).mean()
s2_water_dist_smooth = pd.Series(s2_water_dist).rolling(window=window_size).mean()

df["cym_water_dist"] = cym_water_dist_smooth
df["s1_water_dist"] = s1_water_dist_smooth
df["s2_water_dist"] = s2_water_dist_smooth

df_plot = df.dropna()
sns.set_style("ticks")
g = sns.pairplot(
    df_plot,
    hue="label",
    vars=["phi_Cys35", "phi_Ser80",
          "phi_Val81", "cym_water_dist", "s1_water_dist", "s2_water_dist"],
    kind="scatter",
    diag_kind="hist",
    plot_kws={"s": 20, "alpha": 0.6, "edgecolor": "none"},
    diag_kws={"bins": 25}
)

plt.show()
