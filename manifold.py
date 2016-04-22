import numpy as np
from scipy.spatial.distance import pdist

class manifold:
    def __init__(self, x, l, m, tol=0.1, max_iter=1000):
        self.sigma = l
        self.m = m
        self.tol = tol
        self.max_iter = max_iter
        self._fit(x)

    def _fit(self, x):
        """
        Implementation of Chigirev and Bialek NIPs 2003.

        Input:
        x = data, (dimensions, samples)
        l = lagrange multiplier, variance of gaussian
        n = number of manifold points

        Returns:
        g = manifold positions
        p_t = stochastic matrix
        p = probability of manifold position
        """
        self.manifold = x[:, np.random.randint(0, x.shape[1]-1, self.m)]
        g = self.manifold
        self.prob_manifold = np.ones(self.m).astype('f') / self.m
        n_iter = 0
        dg = self.tol*5
        # perform optimization
        while (dg > self.tol) & (n_iter < self.max_iter):
            # compute the transition matrix
            self._compute_transition_prob(x)
            # compute prob of each manifold point
            self._compute_p_manifold()
            # update the manifold points
            self._compute_manifold(x)
            # compute dg, an exit criteria parameter
            dg = np.sum((self.manifold - g)**2,0).max()
            # update g, manifold points from previous iteration
            g = self.manifold
            # update while loop counter
            n_iter += 1
        self.exit_iterations = n_iter

    def _compute_transition_prob(self, x):
        """
        Returns:
        n samples by m manifold points
        """
        # manifold probability
        prob_m = self.prob_manifold
        prob_m = np.tile(prob_m.reshape(1, self.m), (x.shape[1], 1))

        # multiply manifold prob by distortion
        p = prob_m * self._prob_map(x)

        # compute normalization, col normalized
        z = np.sum(p, 1)
        z = np.tile(z.reshape(z.size, 1), (1, self.m))

        self.transition_matrix = p / z

    def _compute_manifold(self, x):
        """
        Compute manifold points

        Returns:
        (d, m) numpy array
        """
        tmp = np.dot(np.transpose(self.transition_matrix),
                     np.transpose(x)) / x.shape[1]
        z = np.tile(self.prob_manifold.reshape(1, self.m), (x.shape[0], 1))
        self.manifold = np.transpose(tmp) / z

    def _compute_p_manifold(self):
        """
        Compute P(t).

        Returns:
        m manifold point array
        """
        self.prob_manifold = np.mean(self.transition_matrix, 0)

    def _prob_map(self, x):
        """
        Probability of Data given Manifold Point.
        Input:
        Data : d (rows) the dimension of vector space data lives in, by n samples (columns)

        Return :
        (n samples, m manifold points) numpy array
        """
        d = self.manifold.shape[0]
        n = x.shape[1]
        maha = np.zeros(shape=(n, self.m))
        for w in range(self.m):
            mu = np.tile(self.manifold[:, w].reshape(d, 1),
                         (1, n))
            maha[:, w] = np.sum((x-mu)**2, 0) / self.sigma**2
        tmp = np.exp(-0.5 * maha)
        return np.exp(-0.5 * maha)

    def get_manifold_labels(self, raw_data, labels):
        Z = np.sum(self.transition_matrix, 0)
        Z = Z.reshape(1, self.m)
        Z = np.tile(Z, (self.transition_matrix.shape[0], 1))
        T = self.transition_matrix / Z
        Lm = np.dot(np.transpose(T), labels.reshape(labels.size, 1)).squeeze()
        Lm[Lm >= 0.5] = 1.
        Lm [Lm != 1.] = 0.
        return Lm

    def corr_dim(m, rpoints=100):
        """Compute correlation."""
        dist = pdist(m)
        r = np.logspace(np.log10(dist[dist != 0].min()),
                        np.log10(dist.max()), rpoints)
        c = []
        for w in r:
            c += [np.sum(heaviside(dist, w))]
        return [r, np.array(c)]

    def heaviside(self, x, r):
        """Heaviside step function."""
        temp = r - x
        temp[temp < 0] = 0
        temp[temp != 0] = 1.
        return temp
