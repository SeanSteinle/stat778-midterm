from numpy import ndarray

def mspe(preds: ndarray, gt: ndarray):
    """Calculates the mean squared prediction error between a list of predictions, preds, and a list of ground truth labels, gt."""
    if len(preds) != len(gt): return ValueError("predictions and ground-truth vectors must be the same length!")
    preds,gt = preds.tolist(),gt.tolist() #TODO: may be inefficient!
    n = len(preds)
    
    sse = sum([(gt[i] - preds[i])**2 for i in range(n)]) #calculate sume of square errors
    return sse/n

def gcv(preds: ndarray, gt: ndarray, q: int):
    """Calculates the generalized cross-validation approximation"""
    if len(preds) != len(gt): return ValueError("predictions and ground-truth vectors must be the same length!")
    preds,gt = preds.tolist(),gt.tolist() #TODO: may be inefficient!
    n = len(preds)
    
    sgse = sum([((gt[i]-preds[i])/1-(q/n))**2 for i in range(n)]) #calculate sum of generalized squared error
    return sgse/n
    