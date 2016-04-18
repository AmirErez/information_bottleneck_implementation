"""
Run implementation of Chigirev and Bialek NIPs 2003.
"""
import simulation_data as sim
import numpy as np
import os
os.chdir('/Users/oboe/Documents/my_library/papers/ml/NIPs2003Chigirev')
import manifold as m
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable as smap # colormap for plotting
from scipy.integrate import trapz

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
# semi circle example
# ==========================================================
# ==========================================================
# simulate the data

sim_data = sim.semi_circle(500, 10., 0.75)
sim_data = np.hstack([sim_data,
                      sim.semi_circle(500, 5., 0.75)])
# find the manifold
manifold = m.fit(sim_data, 5., 500, max_iter=5000)

# =================================
# plot solution
plt.figure()
plt.plot(sim_data[0, :], sim_data[1, :], 'o',
         color='b', mec='none', alpha=0.5,
         ms=5)
plt.plot(manifold[0][0, :], manifold[0][1, :], 'o',
         color='k', ms=10, alpha=0.75, mec='none')
plt.show(block=False)


# =================================
# compute the correlation function, remember
# that the correlation helps find the optimal number of dimensions
# of the manifold
corr = m.corr_dim(manifold[0].transpose())
corr_d = m.corr_dim(sim_data.transpose())

# Slope of the correlation is approximately equal to the optimal
# dimension of the data or manifold.  Note that the manifold's optimal dim
# is 1
plt.figure(figsize=(5, 4))
plt.plot(corr[0], corr[1], '-o', color='k',
         label='Manifold Correlation')
plt.plot(corr_d[0], corr_d[1], '-o', color='b',
         label='Data Correlation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('log(r)', fontsize=15)
plt.ylabel('log(C(r))', fontsize=15)
plt.legend(loc=0)
plt.tight_layout()
plt.show(block = False)



# ==========================================================
# ==========================================================
# Gaussian mixture example
# ==========================================================
# ==========================================================
# simulate the data
sim_data = np.hstack([sim.bivariate_gaussian(1000, [0, 0], [1., 1.], 0),
                      sim.bivariate_gaussian(1000, [3., 3.], [1., 1.], 0)])
sim_data = np.array(sim_data)


# fit manifold
manifold = m.fit(sim_data, 5., 100, max_iter=5000)


plt.figure(figsize=(5, 4))
plt.plot(sim_data[0, :], sim_data[1, :], '.', alpha=0.5,
         label='Data')
plt.plot(manifold[0][0, :], manifold[0][1, :], 'o',
         color='k', ms=7.5, alpha=0.75,
         label='Manifold')
plt.legend(loc=0)
plt.show(block=False)


# ====================================================
# ====================================================
# Estimate the fraction of cells in each population using manifold points
# ====================================================
# ====================================================

N = 2000 # number of data points
M = 50 # number of manifold points
manifold_data = []
mixture_N = []
dist = np.linspace(0.5, 3.5, 5)
for wdist in dist:
    tmp_data = []
    tmp_mixture_N = []
    for w in range(100):
        tmp_mixture_N += [np.random.randint(0, N, 1)[0]]
        # simulate two bivariate gaussians
        sim_data = np.hstack([sim.bivariate_gaussian(tmp_mixture_N[-1],
                                                     [0, 0], [1., 1.], 0),
                              sim.bivariate_gaussian(N-tmp_mixture_N[-1],
                                                     [wdist, wdist],
                                                     [1., 1.], 0)])
        tmp_data += [m.fit(sim_data, 5., M, max_iter=5000)[0]]
        print 'Iter : ' + str(w) + ' {:0.2f}'.format(wdist)
    manifold_data += [tmp_data]
    mixture_N = [tmp_mixture_N]
mixture_N = np.array(mixture_N).astype('f')
mixture_fraction = mixture_N / N

coluer = get_coluer(len(manifold_data))

plt.figure()
mixture_fraction_estimate = []
count = 0
for wdata in manifold_data:
    y, x = np.histogram(wdata[0, :], 100)
    xplot = x[1:] - 0.5*(x[1] - x[0])
    z = trapz(y, xplot)
    plt.plot(xplot, y/z, '-o',
             color=coluer[count, :], alpha=0.75,
             linewidth=2.5)
    mixture_fraction_estimate += [float(np.sum(y[xplot < 1.5])) / M]
    count += 1
plt.show(block=False)


rho = np.corrcoef([mixture_fraction,
                   mixture_fraction_estimate])
plt.figure()
plt.plot(mixture_fraction,
         mixture_fraction_estimate, 'o',
         label=r'$\rho$ = ' + '{:0.2f}'.format(rho[0, 1]))
plt.xlabel('True Mixture Frcation', fontsize=15)
plt.ylabel('Estimated Mixture Fraction', fontsize=15)
plt.legend(loc=0)
