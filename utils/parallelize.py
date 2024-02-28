import pandas as pd
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from utils.metrics import mspe

models = [LogisticRegression(), LinearRegression()]

def all_models_repK(df: pd.DataFrame, n_repeats: int, n_splits: int):
    """Runs n_repeats repetitions of k-fold cross validations where k=n_splits."""
    scores = {}
    metrics = {
        "mspe": make_scorer(mspe),
    }
    for model in models:
        X,y = df.drop("y", axis=1),df["y"] #assuming only X and y are present.
        rkf = RepeatedKFold(n_repeats=n_repeats,n_splits=n_splits)
        score = cross_validate(model, X, y, cv=rkf, scoring=metrics)
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

def run_jobs(n_jobs: int, df: pd.DataFrame, n_repeats: int=10, n_splits: int=10):
    """High-level function which parallelizes all_models_repK for n_jobs."""
    reps = make_groups(n_repeats, n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(all_models_repK)(df, r, n_splits) for r in reps)
    return results

def aggregate_partitioned_results(scores: list):
    """Brings partitioned data back together under a model->metric->values dictionary structure."""
    model_results = {str(model_name):{"mspe":[]} for model_name in models}
    for partition_n,partition_data in enumerate(scores):
        for model_n,model_name in enumerate(partition_data.keys()): #model_n is 0 indexed
            model_results[model_name]["mspe"].extend(partition_data[model_name]["test_mspe"])
    return model_results