import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.random import random
from rng import rmvexp

np.random.seed(42)

def problem1b():
    cov_mat = np.genfromtxt("data/1b_matrix.csv", delimiter=',')
    mvexp_samples = rmvexp(200, cov_mat)
    print(f"Created samples: {mvexp_samples}")
    return mvexp_samples

def problem1c(X: np.ndarray):
    Y = []
    for i in range(len(X)):
        sub_deviates = X[i,1]+X[i,2]-X[i,6]-X[i,7]-X[i,11]+X[i,13] #compile X from matrix components
        p = norm.cdf(sub_deviates) #get p from x->inverse stdnormal
        y = 0 if p >= random() else 1 #sample from bernoulli w/ prob p
        Y.append(y)
    print(f"Created linear systematic component Y: {Y}")

    #cleaning up results with pandas
    df = pd.DataFrame(X)
    df.columns = ["X"+str(i) for i in range(1,31)]
    df["Y"] = Y
    df.to_csv("data/p1_result.csv")
    return df
    
if __name__ == "__main__":
    print("Solving problem 1b...")
    random_deviates = problem1b()
    print("Solving problem 1c...")
    problem1c(random_deviates)