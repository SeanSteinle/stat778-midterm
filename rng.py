import numpy as np

def runif(n: int=1, seed: int=42, p: int=214783647, m: int=16807):
    """A simple linear congruential generator. Generates uniformly between [0,1]."""
    results = []
    In = seed
    for i in range(n):
        In1 = (m*In%p) #generate new random
        results.append(In1/p) #scale and append to ret list
        In = In1
    return results

def rbern(p: float):
    return 1 if p <= runif(1)[0] else 0

def rbinom(n: int, p: float):
    return sum([p < randunif for randunif in runif(n)])

def rexp(n: int=1, l: float=1.0):
    """A univariate exponential random number generator, generates n samples for rate l."""
    U = runif(n)
    X = [np.log(u)/-l for u in U]
    return X

def rmvexp(n: int, cov_mat: np.ndarray):
    """A multivariate exponential random number sampler, creates n samples from cov_mat covariance matrix."""
    exp_samples = rexp(1, len(cov_mat))
    cov_mat_sqrt = np.linalg.cholesky(cov_mat)
    mvexp_sample = exp_samples * cov_mat_sqrt
    return mvexp_sample