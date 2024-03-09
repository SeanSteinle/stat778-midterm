import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.random import random
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from utils.rng import rmvexp
from utils.parallelize import make_groups, run_cv, aggregate_partitioned_results

np.random.seed(42)

#problem 1a is implemented as rmvexp()

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
    df.to_csv("data/p1_result.csv", index=False)
    return df
    
def problem2a(df):
    n_repeats, n_jobs = 10, 5
    reps = make_groups(n_repeats, n_jobs)

    n_splits = 10
    rkf = RepeatedKFold(n_repeats=n_repeats,n_splits=n_splits)
    rkf_scores = Parallel(n_jobs=n_jobs)(delayed(run_cv)(df, rkf) for r in reps)
    rkf_model_results = aggregate_partitioned_results(rkf_scores)
    #TODO: save RKF scores as json output

    #plot results and save to plots folder
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    plt.boxplot([rkf_model_results['LogisticRegression()']['mspe'],rkf_model_results['LinearRegression()']['mspe']])
    ax.set_xticklabels(['Logistic Regression', 'Linear Regression'])
    plt.xlabel("Model Type")
    plt.ylabel("Mean Squared Prediction Error")
    plt.title("MSPE For Logistic and Linear Regression with Repeated K-Fold Cross Validation")
    plt.savefig("plots/mspe_loglin_loo.png")
    raise NotImplementedError

def problem2b():
    raise NotImplementedError

def problem2c():
    raise NotImplementedError

if __name__ == "__main__":
    print("Solving problem 1b...")
    random_deviates = problem1b()
    print("Solving problem 1c...")
    df = problem1c(random_deviates)
    print("Solving problem 2a...")
    problem2a(df)
    print("Solving problem 2b...")
    problem2b(df)
    print("Solving problem 2c...")
    problem2c()