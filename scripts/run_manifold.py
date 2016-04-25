"""
Run implementation of Chigirev and Bialek NIPs 2003.
"""
import simulation_data as sim
import numpy as np
import os
os.chdir('/Users/oboe/Documents/my_library/papers/ml/NIPs2003Chigirev')
import manifold as m
import matplotlib.pyplot as plt
from scipy.integrate import trapz
%matplotlib inline


# ==========================================================
# ==========================================================
# semi circle example
# ==========================================================
# ==========================================================
# simulate the data
m = reload(m)
sim_data = sim.semi_circle(500, 10., 0.75)
sim_data = np.hstack([sim_data,
                      sim.semi_circle(500, 5., 0.75)])
# find the manifold
manifold = m.manifold(sim_data, 1.5, 500, max_iter=5000)

# =================================
# plot solution
plt.figure()
plt.plot(sim_data[0, :], sim_data[1, :], 'o',
         color='b', mec='none', alpha=0.5,
         ms=5)
plt.plot(manifold.manifold[0, :],
         manifold.manifold[1, :], 'o',
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
