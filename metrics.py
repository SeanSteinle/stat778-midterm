def mspe(preds, gt):
    """Calculates the mean squared prediction error between a list of predictions, preds, and a list of ground truth labels, gt."""
    if len(preds) != len(gt): return ValueError("predictions and ground-truth vectors must be the same length!")
    preds,gt = preds.tolist(),gt.tolist() #TODO: may be inefficient!
    sse = sum([(gt[i] - preds[i])**2 for i in range(len(gt))])
    return sse/len(gt)

def gcv():
    """Calculates the generalized cross-validation approximation"""
    