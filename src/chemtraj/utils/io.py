from ase.io import read, write 
import pandas as pd



def dataframe_from_extxyz(extxyz):
    """Take extxyz with labels and return pd dataframe"""
    atoms_list = read(extxyz, index=":")
    rows = [] 
    for i, atoms in enumerate(atoms_list):
        rows.append({
                    "frame_id": id,
                    "label": atoms.info.get("label")
        })
    df = pd.DataFrame(rows) 
    return df
