import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import norm
from numpy.random import random
from joblib import Parallel, delayed
from time import time
from sklearn.model_selection import RepeatedKFold, LeaveOneOut

from rng import rmvexp
from parallelize import make_groups, run_cv, aggregate_partitioned_results

seed = 42
np.random.seed(seed)

#Note: problem 1a is implemented as rmvexp()

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
    df.to_csv(f"data/p1_result_{seed}.csv", index=False)
    return df
    
def problem2a(df):
    #configure repeated k-fold cross-validation (rkf)
    n_repeats, n_splits, n_jobs = 10, 10, 5
    reps = make_groups(n_repeats, n_jobs)
    rkf = RepeatedKFold(n_repeats=n_repeats,n_splits=n_splits)
    
    #run and time rkf
    st = time()
    rkf_scores = Parallel(n_jobs=n_jobs)(delayed(run_cv)(df, rkf) for r in reps)
    runtime = time() - st

    #aggregate results, add metadata to dictionary, save as json
    rkf_model_results = aggregate_partitioned_results(rkf_scores)
    rkf_model_results['repeats'] = n_repeats
    rkf_model_results['splits'] = n_splits
    rkf_model_results['runtime'] = runtime
    rkf_model_results['jobs'] = n_jobs
    with open(f"data/rkf_results_{seed}.json", "w") as f: 
        json.dump(rkf_model_results, f)

    #plot results and save to plots folder
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    plt.boxplot([rkf_model_results['LogisticRegression()']['mspe'],rkf_model_results['LinearRegression()']['mspe']])
    ax.set_xticklabels(['Logistic Regression', 'Linear Regression'])
    plt.xlabel("Model Type")
    plt.ylabel("Mean Squared Prediction Error")
    plt.title("MSPE For Logistic and Linear Regression with Repeated K-Fold Cross Validation")
    plt.savefig(f"plots/mspe_loglin_rkf_{seed}.png")

    return rkf_model_results

def problem2b(df):
    #configure leave one out cross-validation (loo)
    n_jobs = 5
    loo = LeaveOneOut()

    #run and time loo
    st = time()
    loo_scores = Parallel(n_jobs=n_jobs)(delayed(run_cv)(df, loo) for i in range(1)) #this looks cl
    runtime = time() - st
    
    #add metadata to results dictionary, save as json
    loo_model_results = aggregate_partitioned_results(loo_scores)
    loo_model_results['runtime'] = runtime
    loo_model_results['jobs'] = n_jobs
    with open(f"data/loo_results_{seed}.json", "w") as f: 
        json.dump(loo_model_results, f)

    #plot results
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    plt.boxplot([loo_model_results['LogisticRegression()']['mspe'],loo_model_results['LinearRegression()']['mspe']])
    ax.set_xticklabels(['Logistic Regression', 'Linear Regression'])
    plt.xlabel("Model Type")
    plt.ylabel("Mean Squared Prediction Error")
    plt.title("MSPE For Logistic and Linear Regression with Leave One Out Cross Validation")
    plt.savefig(f"plots/mspe_loglin_loo_{seed}.png")

    return loo_model_results

def problem2c():
    raise NotImplementedError

if __name__ == "__main__":
    print("Solving problem 1b...")
    random_deviates = problem1b() #generate X
    print("Solving problem 1c...")
    df = problem1c(random_deviates) #generate Y
    print("Solving problem 2a...")
    rkf_results = problem2a(df) #estimate Y w/ LinR+LogR, use RKF validation
    print("Solving problem 2b...")
    loo_results = problem2b(df) #estimate Y w/ LinR+LogR, use LOO validation
    #print("Solving problem 2c...")
    #problem2c()