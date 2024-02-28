import numpy as np
import math

def runif(n: int=1, seed: int=42, p: int=214783647, m: int=16807):
    """A simple linear congruential generator. Generates uniformly between [0,1]."""
    results = []
    In = seed #TODO: how to make this easily modifiable? env var? global?
    for i in range(n):
        In1 = (m*In%p) #generate new random
        results.append(In1/p) #scale and append to ret list
        In = In1
    return results

def rbern(p: float, seed: int=42):
    """A univariate bernoulli random number generator, generates 1 sample for probability p."""
    return 1 if p <= runif(1, seed)[0] else 0

def rbinom(n: int, p: float, seed: int=42):
    """A univariate binomial random number generator, generates n samples for probability p."""
    return sum([p < randunif for randunif in runif(n, seed)])

def rstdnorm(n: int=1, seed: int=42):
    """A univariate standard normal random number generator, generates n samples."""
    stream = runif(n,seed)
    results = []
    for i in range(0,len(stream),2):
        X,Y = stream[i],stream[i+1]
        #X,Y = sigma*X+mu,sigma*Y+mu #TODO: can make rnorm with this, but giving https://stackoverflow.com/questions/27784528/numpy-division-with-runtimewarning-invalid-value-encountered-in-double-scalars ?
        Z0 = ((-2*np.log(X))**0.5)*(math.cos(2*math.pi*Y)) #don't need to generate the pair?
        results.append(Z0)
        Z1 = ((-2*np.log(X))**0.5)*(math.sin(2*math.pi*Y))
        results.append(Z1)
    return results

def rinvstdnorm(x: int, seed: int=42):
    """A univariate inverse standard normal random number generator, generates n samples."""
    N = 1000
    return sum([x <= sample for sample in rstdnorm(N,seed)])/N

def rexp(n: int=1, l: float=1.0, seed: int=42):
    """A univariate exponential random number generator, generates n samples for rate l."""
    U = runif(n, seed)
    X = [np.log(u)/-l for u in U]
    return X

def rmvexp(n: int, cov_mat: np.ndarray, seed: int=42):
    """A multivariate exponential random number sampler, creates n samples from cov_mat covariance matrix."""
    exp_samples = np.asarray(rexp(n*len(cov_mat), 1, seed)).reshape(n,len(cov_mat)) #is this right?
    cov_mat_sqrt = np.linalg.cholesky(cov_mat) #TODO: add error checking here
    mvexp_sample = np.dot(exp_samples,cov_mat_sqrt)
    return mvexp_sample

#Questions:
#1. am I sampling correctly in rmvexp? I sample 200*30 (n*len(cov_mat)) times from the univariate exponential,
#then I take the dot product by the cov_mat.
#2. When I sample the bernoulli distribution to get the linear systematic component, the same runif() draw is
#behind every call. This makes sequential draws highly correlated, but they should be independent.
    #2b. I have a similar problem in my rinvstdnorm function. I can just rerun it many times to get a better estimate though.
    #One idea: generate a stream? this is how I solved the problem in rnorm
