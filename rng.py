import numpy as np
from numpy.random import exponential
from numpy.linalg import cholesky, LinAlgError

def rmvexp(n: int, cov_mat: np.ndarray, seed: int=42):
    """A multivariate exponential random number sampler, creates n samples from cov_mat covariance matrix."""
    exp_samples = exponential(1, (n,len(cov_mat))) #sample n*p samples of univariate exponential with rate = 1
    try:
        cov_mat_sqrt = np.transpose(cholesky(cov_mat)) #calculate sqrt of matrix using cholesky decomposition, transpose to get upper triangular matrix
    except LinAlgError as e:
        print("Matrix was either not positive definite or was not symmetric!")
    mvexp_sample = np.dot(exp_samples,cov_mat_sqrt) #multiply to get sample
    return mvexp_sample