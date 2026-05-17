import numpy as np 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path 

import MDAnalysis as mda 
from MDAnalysis.analysis import pca, align 

from chemtraj.preprocessing.selection import select_protein 


def make_pca_df(transformed):
    """
    Create DataFrame from PCA-transformed coordinates
    params:
        transformed: np.ndarray Shape (n_frames, n_comp)
    returns:
        pd.DataFrame 
            columns: frame_id, PC1, PC2 ... 

    """ 
    pca_df = pd.DataFrame({
                          "frame_id": np.arange(transformed.shape[0])
    })
    for i in range(transformed.shape[1]):
        pca_df[f"PC{i+1}"] = transformed[:, i]
    return pca_df

def cumulated_pca_comp_plot(pca):
    for i in range(3):
        print(f"Cumulated Variance {pca.cumulated_variance[i]}")
    plt.plot(pca.cumulated_variance[:5])
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative variance") 
    plt.show()


def pca_pair_plot(pca_df, label):
    pca_df["label"] = label # for coloring purposes 
    unique_labels = sorted(pca_df["label"].dropna().unique())

    palette = dict(zip(
                   unique_labels,
                   sns.color_palette("tab10", n_colors=len(unique_labels))
    ))
    g = sns.PairGrid(pca_df, hue="label", palette=palette)
    g.map(plt.scatter, marker=".")
    g.add_legend(title="Label")
    plt.show()

def get_cosine_content(pca, backbone): 
    """ 
    pca: pca object
    backbone: selection mda universe 

        Cosine content shows how similar the PC is over the traj
        to a cosine function
        it indicates whether the PC converged
        Take with a grain of salt for directional processes
    """ 
    tranformed = pca.transform(backbone, n_components=3)

    for i in range(3):    
        cc = pca.cosine_content(transformed, i)

    return cc
