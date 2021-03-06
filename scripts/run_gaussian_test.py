"""
Run implementation of Chigirev and Bialek NIPs 2003 on mixture of gaussians.
"""

import os
os.chdir('/Users/oboe/Documents/chigirev')
import tools.simulation_data as sim
import numpy as np
import tools.manifold as m
import tools.binary_classification_stats as bcs
from tools import plots
%matplotlib inline

import matplotlib.pyplot as plt

# TODO:0 make plotting functions to separate simulation code from plots
# DONE:0 Test the ability to identify Gaussians separated by different distances, use AUC to measure the separation of population.

# ====================================================
# ====================================================
# Simulate and applying information bottleneck to samples from a mixture of
# gaussians that are separated by various distances.
# ====================================================
# ====================================================

No = 500  # number of negatives
N1 = 500  # number of positives
M = 100  # number of manifold points
labels = np.hstack([np.zeros(No), np.ones(N1)])
roc_raw_data = []
roc_manifold = []
dist = np.linspace(0.1, 4, 20)
for wdist in dist:
    # simulate two gaussians
    sim_data = np.hstack([np.random.randn(No),
                          wdist + np.random.randn(N1)])
    tmp_data = sim_data.reshape(1, No + N1)
    tmp_manifold = m.manifold(tmp_data, 1.5, M, max_iter=5000)
    roc_raw_data += [bcs.ROC(tmp_data, labels)]
    mlabels = tmp_manifold.get_manifold_labels(tmp_data, labels)
    roc_manifold += [bcs.ROC(tmp_manifold.manifold.squeeze(), mlabels)]

# ============================================================
# ============================================================
# plot results
# ============================================================
# ============================================================

idx = 15
plots.manifold_hist(roc_raw_data[idx].data,
                    roc_manifold[idx].data,
                    savename='figs/bimodal_gauss_sampling.png')


# ============================================================
# ============================================================
# roc analysis
# ============================================================
# ============================================================

idx = 8
rfp, rtp = roc_raw_data[idx].roc_curve()
mfp, mtp = roc_manifold[idx].roc_curve()
plt.figure(figsize=(5, 4))
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
plt.savefig('figs/roc.png')

data_auroc = [wroc.auroc() for wroc in roc_raw_data]
manifold_auroc = [wroc.auroc() for wroc in roc_manifold]
plt.figure(figsize=(5, 4))
plt.plot(dist, data_auroc, '-', color='r',
         linewidth=2.5, label='Raw Data')
plt.plot(dist, manifold_auroc, '-', color='blue',
         linewidth=2.5, label='Manifold')
plt.xlabel(r'$\mu_1 - \mu_0$', fontsize = 20)
plt.ylabel('AUROC', fontsize=15)
plt.ylim([0.5, 1.1])
plt.tight_layout()
plt.savefig('figs/auroc.png', fmt='png')
