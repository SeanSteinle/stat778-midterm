import pandas as pd
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from metrics import mspe

models = [LogisticRegression(), LinearRegression()]

def run_cv(df: pd.DataFrame, cv_technique):
    """Runs cross validation scheme specified by cv_technique on data in df."""
    scores = {}
    metrics = {
        "mspe": make_scorer(mspe),
    }
    for model in [LogisticRegression(), LinearRegression()]:
        X,y = df.drop("Y", axis=1),df["Y"] #assuming only X and y are present.
        score = cross_validate(model, X, y, cv=cv_technique, scoring=metrics)
        scores[str(model)] = score
    return scores

#define high-level parallelization
def make_groups(num_reps: int, num_jobs: int):
    """Divides num_reps by num_jobs such that no group is more than one larger than another."""
    r = num_reps%num_jobs
    groupsize = int(num_reps/num_jobs)
    groups = []
    for i in range(num_jobs):
        if r > 0:
            groups.append(groupsize+1)
            r-=1
        else:
            groups.append(groupsize)
    return groups

def run_jobs(n_jobs: int, df: pd.DataFrame, cv_technique, n_repeats: int=10, n_splits: int=10):
    """High-level function which parallelizes run_cv for n_jobs."""
    reps = make_groups(n_repeats, n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(run_cv)(df, r, n_splits, cv_technique) for r in reps) #this doesn't 
    return results

def aggregate_partitioned_results(scores: list):
    """Brings partitioned data back together under a model->metric->values dictionary structure."""
    model_results = {str(model_name):{"mspe":[]} for model_name in [LogisticRegression(),LinearRegression()]}
    for partition_n,partition_data in enumerate(scores):
        for model_n,model_name in enumerate(partition_data.keys()): #model_n is 0 indexed
            model_results[model_name]["mspe"].extend(partition_data[model_name]["test_mspe"])
    return model_results