import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable as smap # colormap for plotting
from scipy.integrate import trapz
import matplotlib.pyplot as plt

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

def manifold_hist(data, mdata, savename=False, nbins=25, yfrac=0.1):
    yd, xd = get_hist_density(data, nbins=nbins)
    ym, xm = get_hist_density(mdata, nbins=nbins)
    fig = plt.figure(figsize=(5, 4))
    plt.plot(xd, yd / yd.max(), '-', color='r', linewidth=2.5)
    plt.plot(xm, ym / ym.max(), '-', color='b', linewidth=2.5)
    plt.plot(data,
             _get_sample_points_for_hist([0, 1.],
                                         data.size,
                                         yfrac),
             '.', color='r', label='Data',
             alpha=0.75)
    plt.plot(mdata,
             _get_sample_points_for_hist([0, 1.],
                                         mdata.size,
                                         yfrac),
             '^', color='b', label='Manifold',
             alpha=0.75, mec='none')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('Relative Density', fontsize=15)
    plt.ylim([0, 1])
    plt.legend(loc=0)
    plt.tight_layout()
    if savename:
        _save(savename, fig)

def _get_sample_points_for_hist(ylims, N, yfrac):
    delta = ylims[1] - ylims[0]
    mu = delta * yfrac + ylims[0]
    return mu + mu**1.5*np.random.randn(N)

def get_hist_density(data, nbins=25):
    y, x = np.histogram(data, nbins)
    x = x[1:] - 0.5*(x[1] - x[0])
    p = y / trapz(y, x)
    return [p, x]

def _save(savename, fig):
    plt.savefig(savename, fmt=savename.split('.')[-1])
