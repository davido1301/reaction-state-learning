from ase.io import read, write 
import pandas as pd
import yaml
from pathlib import Path 
from dataclasses import dataclass


@dataclass 
class Config: 
    top: Path 
    traj: Path 
    db: Path 
    reactive_selection: str | None = None
    parquet_path: Path | None = None 
    results_dir: Path | None = None 

def read_config(yaml_path: str | Path) -> Config:
    yaml_path = Path(yaml_path) 

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found at {yaml_path}") 

    with open(yaml_path, "r") as f:
        conf = yaml.safe_load(f) 

    if conf is None: 
        raise ValueError(f"Config file is empty")
    

    required_keys = ["TOP", "TRAJ"]
    missing = [key for key in required_keys if key not in conf or conf[key] is None] 


    if missing: 
        raise KeyError(f"Missing required keys: {missing}") 

    return Config(
        db=Path(conf["DB"]),
        top=Path(conf["TOP"]),
        traj=Path(conf["TRAJ"]),
        reactive_selection=conf["REACTIVE_SELECTION"] if conf.get("REACTIVE_SELECTION") else None,
        parquet_path=Path(conf["PARQUET_PATH"]) if conf.get("PARQUET_PATH") else None,
        results_dir=Path(conf["RESULTS_DIR"]) if conf.get("RESULTS_DIR") else None
    )


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


def read_confiiiiig(yaml_path):
    """ Read in yaml
    DB: ?
    TOP:
    TRAJ: 
    REACTIVE SELECTION: (optional)
    """ 

    with open(yaml_path, "r") as f:
        conf = yaml.safe_load(f)
        DB = conf["DB"] 
        TOP = conf["TOP"]
        TRAJ = conf["TRAJ"]  
        
        if conf["REACTIVE_SELECTION"]:
            REACTIVE_SELECTION = conf["REACTIVE_SELECTION"]
        if conf["PARQUET_PATH"]:
            PARQUET_PATH = conf["PARQUET_PATH"]
        return DB, TOP, TRAJ, REACTIVE_SELECTION, PARQUET_PATH
