"""
Run implementation of Chigirev and Bialek NIPs 2003 on mixture of gaussians.
"""

import os
os.chdir('/Users/oboe/Documents/chigirev')
import simulation_data as sim
import numpy as np
import manifold as m
import binary_classification_stats as bcs
import matplotlib.pyplot as plt
%matplotlib inline
# TODO make plotting functions to separate simulation code from plots
# TODO Test the ability to identify Gaussians separated by different distances, use AUC to measure the separation of population.


# ====================================================
# ====================================================
# Estimate the fraction of cells in each population using manifold points
# ====================================================
# ====================================================
m = reload(m)
No = 500 # number of data points
N1 = 500
M = 100 # number of manifold points
labels = np.hstack([np.zeros(No), np.ones(N1)])
roc_raw_data = []
roc_manifold = []
dist = np.linspace(0.1, 4, 10)
for wdist in dist:
    # simulate two gaussians
    sim_data = np.hstack([np.random.randn(No),
                          wdist + np.random.randn(N1)])
    tmp_data = sim_data.reshape(1, No + N1)
    tmp_manifold = m.manifold(tmp_data, 1.5, M, max_iter=5000)
    roc_raw_data += [bcs.ROC(tmp_data, labels)]
    mlabels = tmp_manifold.get_manifold_labels(tmp_data, labels)
    roc_manifold += [bcs.ROC(tmp_manifold.manifold.squeeze(), mlabels)]


idx = 9
plt.plot(roc_raw_data[idx].data,
         roc_raw_data[idx].labels,'.', color='r')
plt.plot(roc_manifold[idx].data,
         roc_manifold[idx].labels, 'x', color='b')
plt.ylim([-0.1, 1.1])


idx = 0
rfp, rtp = roc_raw_data[idx].roc_curve()
mfp, mtp = roc_manifold[idx].roc_curve()
plt.figure()
plt.plot([0, 1], [0, 1], '--', color='k',
         linewidth=2.5, alpha=0.5,
         label='Random, AUROC : 0.5')
plt.plot(rfp, rtp, '-', color='r', linewidth=2.5,
         label='Raw Data, AUROC : {:0.2f}'.format(roc_raw_data[idx].auroc()))
plt.plot(mfp, mtp, '-', color='b', linewidth=2.5,
         label='Manifold Data, AUCROC : {:0.2f}'.format(roc_manifold[idx].auroc()))
plt.legend(loc=0)
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.tight_layout()

data_auroc = [wroc.auroc() for wroc in roc_raw_data]
manifold_auroc = [wroc.auroc() for wroc in roc_manifold]
plt.figure()
plt.plot(dist, data_auroc, '-', color='r',
         linewidth=2.5, label='Raw Data')
plt.plot(dist, manifold_auroc, '-', color='blue',
         linewidth=2.5, label='Manifold')
plt.xlabel(r'$\mu_1 - \mu_0$', fontsize = 20)
plt.ylabel('AUROC', fontsize=15)
plt.ylim([0.5, 1.1])
