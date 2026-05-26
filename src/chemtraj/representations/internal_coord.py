import MDAnalysis as mda 
import numpy as np 
from MDAnalysis.analysis.dihedrals import Dihedral 

from sklearn.decomposition import PCA 



def get_dihedrals(universe, ids):
    """ MDA Universe + ids of residues 
        --> list Dihedrals 
    """ 

    phi_groups = []
    for id in ids:
        res = universe.residues[id]
        ag = res.phi_selection() 
        if ag is not None:
            phi_groups.append(ag)

    phi = Dihedral(phi_groups).run()
    phi_angles = phi.results.angles 
    return phi_angles

