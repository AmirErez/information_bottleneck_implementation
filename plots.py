import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable as smap # colormap for plotting


# ==========================================================
# ==========================================================
# Helper function for plotting

def get_coluer(n, cmap='viridis'):
    """
    Get color maps for plotting.

    Input:
    n : number of colors
    (optional)
    cmap : string of cmap name

    Returns
    Numpy array with n rows and 4 columns representing rgb encoding and
    alpha (transparency scale)
    """
    x = np.arange(float(n))
    coluer_map = smap(cmap=cmap)
    coluer = coluer_map.to_rgba(x)
    return coluer

# ==========================================================
# ==========================================================
# Helper function for plotting

def manifold_scatter()
