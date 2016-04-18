import numpy as np
from scipy.spatial.distance import pdist


def fit(x, l, n, tol=0.1, max_iter=1000):
    '''
    Implementation of Chigirev and Bialek NIPs 2003.

    Input:
    x = data, (dimensions, samples)
    l = lagrange multiplier, 2*variance of gaussian
    n = number of manifold points

    Returns:
    g = manifold positions
    p_t = stochastic matrix
    p = probability of manifold position
    '''
    #initialize manifold points
    n_samp = x.shape[1]
    g = [0,x[:,np.random.randint(0,n_samp-1,n)]]
    #intialize the stochastic map
    #iterate over samples
    p_t = []
    for w in range(n_samp):
        #get normalization
        z = np.sum([prob_map(x[:,w],g[1][:,wcount],l) for wcount in range(n)])
        #get iniitial transition prob, rows = samples, cols = manifold
        p_t += [[prob_map(x[:,w],g[1][:,wcount],l) / z for wcount in range(n)]]
    #p_t rows are samples, columns are manifold points
    p_t = [0,np.array(p_t)]
    #perform algorithm
    n_iter = 0
    dg = tol*5
    while (dg > tol) & (n_iter < max_iter):
        #compute prob of each manifold point
        p = np.sum(p_t[1],0) / n_samp

        #update gammas
        g[0] = g[1]
        temp = []
        #interate over each manifold point
        for wg in range(n):
            temp += [[np.sum(x[wdim,:]*p_t[1][:,wg]) / (n_samp*p[wg]) for wdim in range(x.shape[0])]]
        g[1] = np.array(temp).transpose()

        p_t[0] = p_t[1]
        #update partition function, and transition probs
        for w in range(n_samp):
            z = np.sum([p[wcount] * prob_map(x[:,w],g[1][:,wcount],l) for wcount in range(n)])
            temp = [p[wcount] * prob_map(x[:,w],g[1][:,wcount],l) / z for wcount in range(n)]
            p_t[1][w,:] = np.array(temp)

        #compute delta gamma
        dg = np.sqrt(np.sum((g[1] - g[0])**2,0)).max()
        #update counter
        n_iter += 1
    print 'Iterations to convergence : {}'.format(n_iter)
    print 'dg : {}'.format(dg)
    if np.isnan(dg):
        a = b
    return [g[1],p_t[0],p,n_iter]

def prob_map(x,g,l):
    return np.exp(-np.dot(x-g, x-g) / l)


def corr_dim(m, rpoints=100):
    dist = pdist(m)
    r = np.logspace(np.log10(dist[dist != 0].min()),
                    np.log10(dist.max()),rpoints)
    c = []
    for w in r:
        c += [np.sum(heaviside(dist,w))]
    return [r,np.array(c)]


def heaviside(x,r):
    temp = r - x
    temp[temp < 0] = 0
    temp[temp != 0] = 1.
    return temp
