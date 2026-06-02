import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from pathlib import Path 
import gc

df = pd.read_parquet("/home/davido/Projects/HiWi_Maike/reaction-state-learning/data/processed/features/phi_dihedrals_all_metad.parquet") 


RESULTS_HIST = Path("src/chemtraj/results/histo_dihedrals_labeled/all_metad/")
RESULTS_HIST.mkdir(parents=True, exist_ok=True) 

print(df.shape) 
print(df.head()) 

phi_cols = [col for col in df.columns if col.startswith("phi_res")] 

correlation_matrix = df[phi_cols].corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap of Phi Dihedrals")
plt.savefig(RESULTS_HIST / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


for i, feature in enumerate(phi_cols):
    plt.figure(figsize=(6, 4))

    sns.histplot(
        data=df,
        x=feature,
        hue="label",
        kde=False,
        element="step",
        stat="density",
    )

    filename = f"{feature}_histogram_all_metad.png"
    save_path = RESULTS_HIST / filename

    plt.title(feature)
    plt.xlabel(f"Dihedral angle {feature}")
    plt.ylabel("Density")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if i % 10 == 0:
        print(f"Saved {i + 1}/{len(phi_cols)} histograms")
        gc.collect()
