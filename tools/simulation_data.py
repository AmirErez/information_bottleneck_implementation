import numpy as np


def semi_circle(n, r, sig):
    """
    Sample from a semi circle.

    Input:
    r = radius
    n = samples
    sig = noise

    Return:
    numpy array of samples
    """
    theta = np.pi*np.random.rand(n)
    r = r + sig*np.random.randn(n)
    data = [r*np.cos(theta), r*np.sin(theta)]
    data = np.array(data)
    return data


def bivariate_gaussian(n, mu, sig, rho):
    """
    Sample bivariate gaussian using cholesky decomp.

    Input:
    n = data points
    mu = list representing mean vector
    sig = list representing variances
    rho = correlatino coefficient

    Returns:
    Numpy array of samples
    """
    if type(mu) == list:
        mu = np.array(mu)
    mu = mu.reshape(mu.size, 1)
    sigma = np.array([[sig[0], rho*np.sqrt(sig[0]*sig[1])],
                      [rho*np.sqrt(sig[0]*sig[1]), sig[1]]])
    L = np.linalg.cholesky(sigma)
    data = mu + np.dot(L, np.random.randn(mu.shape[0], n))
    return data
