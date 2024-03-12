import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import norm
from numpy.random import random
from joblib import Parallel, delayed
from time import time
from sklearn.model_selection import RepeatedKFold, LeaveOneOut, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

from rng import rmvexp
from metrics import gcv, mspe
from parallelize import make_groups, run_cv, aggregate_partitioned_results

#suppress non-convergence warnings for logistic regression
from warnings import filterwarnings
filterwarnings('ignore')

#global variables
seed = 42
np.random.seed(seed)
cov_mat = np.genfromtxt("data/1b_matrix.csv", delimiter=',')

#Note: problem 1a is implemented as rmvexp()

def problem1b():
    mvexp_samples = rmvexp(200, cov_mat, seed)
    print(f"Created samples: {mvexp_samples}")
    return mvexp_samples

def problem1c(X: np.ndarray, save_path=""):
    Y = []
    for i in range(len(X)):
        sub_deviates = X[i,1]+X[i,2]-X[i,6]-X[i,7]-X[i,11]+X[i,13] #compile X from matrix components
        p = norm.cdf(sub_deviates) #get p from x->inverse stdnormal
        y = 0 if p >= random() else 1 #sample from bernoulli w/ prob p
        Y.append(y)

    #cleaning up results with pandas
    df = pd.DataFrame(X)
    df.columns = ["X"+str(i) for i in range(1,31)]
    df["Y"] = Y
    if save_path != "": #function is used outside of 1c, only save path and print results in 1c
        print(f"Created linear systematic component Y: {Y}")
        df.to_csv(save_path, index=False)
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

def problem2c(df, rkf_results, loo_results):
    results = {"Logistic": {}, "Linear": {}}

    #train generalized model
    linr = LinearRegression()
    X,y = df.drop("Y", axis=1),df["Y"]
    linr.fit(X,y)

    #extract predictions, calculate gcv approximation
    preds = linr.predict(X)
    q = len(linr.coef_)
    linr_gcv_error = gcv(preds,y,q)

    #aggregate average errors for rkf and loo, output results
    rkf_mspes = rkf_results['LinearRegression()']['mspe'] 
    linr_rkf_avg_mspe = sum(rkf_mspes)/len(rkf_mspes) 

    loo_mspes = loo_results['LinearRegression()']['mspe'] 
    linr_loo_avg_mspe = sum(loo_mspes)/len(loo_mspes) 
    
    results["Linear"] = {
        "RKF": linr_rkf_avg_mspe,
        "LOO": linr_loo_avg_mspe,
        "GCV": linr_gcv_error
    }

    print(f"(Linear Regression) RKF Average MSPE: {linr_rkf_avg_mspe}\tLOO Average MSPE: {linr_loo_avg_mspe}\tGCV Approximation: {linr_gcv_error}")

    #NOTE: here I estimate Logistic Regression errors too. this wasn't explicitly asked for,
    #but allows us to look at a clearer picture in 3c when we analyze all of our error estimations.
    
    #NOTE: also, I didn't make this code into functions bc code is only repeated twice. if this was for 3 or more
    #model types it would make sense to.

    #train generalized model
    logr = LogisticRegression()
    X,y = df.drop("Y", axis=1),df["Y"]
    logr.fit(X,y)

    #extract predictions, calculate gcv approximation
    preds = logr.predict(X)
    q = len(logr.coef_)
    logr_gcv_error = gcv(preds,y,q)

    #aggregate average errors for rkf and loo, output results
    rkf_mspes = rkf_results['LogisticRegression()']['mspe'] 
    logr_rkf_avg_mspe = sum(rkf_mspes)/len(rkf_mspes) 

    loo_mspes = loo_results['LogisticRegression()']['mspe'] 
    logr_loo_avg_mspe = sum(loo_mspes)/len(loo_mspes) 

    results["Logistic"] = {
        "RKF": logr_rkf_avg_mspe,
        "LOO": logr_loo_avg_mspe,
        "GCV": logr_gcv_error
    }
    return results

def problem3a(df1):
    #train linr+logr models on df1
    M = []
    for m in [LogisticRegression(), LinearRegression()]:
        X,y = df1.drop("Y", axis=1),df1["Y"]
        M.append(m.fit(X,y))

    #sample new dataset (df3a), calculate model errors for predictions on df3a
    X3a = rmvexp(10000, cov_mat, seed)
    df3a = problem1c(X3a)
    errors = []
    for m in M:
        X,y = df3a.drop("Y", axis=1),df3a["Y"]
        preds = m.predict(X)
        errors.append(mspe(preds,y))
    linr_cond_error, logr_cond_error = errors

    return df3a, linr_cond_error, logr_cond_error
    

def problem3b(df3a):
    #create 1,000 data sets (df3b_sets)
    X = [rmvexp(200, cov_mat, seed) for _ in range(1000)] #sample from mv exponential distribution
    df = [problem1c(X[i]) for i in range(1000)] #calculate true Ys

    #train 1,000 linr+logr models
    M = []
    for i in range(len(df)):
        models_for_set = []
        for m in [LogisticRegression(), LinearRegression()]:
            X,y = df[i].drop("Y", axis=1),df[i]["Y"]
            models_for_set.append(m.fit(X,y))
        M.append(models_for_set)

    #calculate model errors for predictions against df3a for all 1,000 models
    linr_mspes,logr_mspes = [],[]
    X,y = df3a.drop('Y', axis=1), df3a['Y']
    for (linr,logr) in M:
        preds = linr.predict(X)
        linr_mspes.append(mspe(preds,y))
        preds = logr.predict(X)
        logr_mspes.append(mspe(preds,y))

    #average model error for each model
    linr_uncond_error, logr_uncond_error = sum(linr_mspes)/len(linr_mspes), sum(logr_mspes)/len(logr_mspes) 

    return linr_uncond_error, logr_uncond_error

def problem3c(cv_errors, conuncond_errors):
    answer = f"""The cross-validation estimations of error for each model are as follows:\n{cv_errors}.
    The conditional and unconditional estimations of error for each model are as follows:\n{conuncond_errors}.

    Whether cross-validation better predicts conditional or unconditional error seems to depend on the model type. In the case
    of logistic regression the conditional estimation is more similar to the majority of the cross-validation metrics.
    The opposite is true for linear regression: unconditional estimation is more similar to the cross-validation estimates.
    That said cross-validation estimates both conditional and unconditional much better in logistic regression than in 
    linear regression.
    """

    with open("data/3c_answer.txt", "w") as f:
        f.write(answer)

    print(answer)

if __name__ == "__main__":
    #note: 1a is solved by rmvexp() in rng.py
    print("Solving problem 1b...")
    random_deviates = problem1b() #generate X
    print("Solving problem 1c...")
    df1 = problem1c(random_deviates, f"data/p1_result_{seed}.csv") #generate Y
    print("Solving problem 2a...")
    rkf_results = problem2a(df1) #estimate Y w/ LinR+LogR, use RKF validation and MSPE error
    print("Solving problem 2b...")
    loo_results = problem2b(df1) #estimate Y w/ LinR+LogR, use LOO validation and MSPE error
    print("Solving problem 2c...")
    cv_errors = problem2c(df1, rkf_results, loo_results) #estimate Y w/ Linr+LogR, but score with generalized cross-validation approximation
    print("Solving problem 3a...")
    df3b, linr_cond_error, logr_cond_error = problem3a(df1)
    print("Solving problem 3b...")
    linr_uncond_error, logr_uncond_error = problem3b(df3b)
    
    conuncond_errors = { #conditional and unconditional error dictionary
        "Logistic": {"Conditional": logr_cond_error,
                     "Unconditional": logr_uncond_error},
        "Linear": {"Conditional": linr_cond_error,
                   "Unconditional": linr_uncond_error}
    }

    print("Solving problem 3c...")
    problem3c(cv_errors, conuncond_errors)
