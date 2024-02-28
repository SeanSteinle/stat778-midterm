#TODO: implement GCV

def mspe(preds, gt):
    """Calculates the mean squared prediction error between a list of predictions, preds, and a list of ground truth labels, gt."""
    if len(preds) != len(gt): return ValueError("predictions and ground-truth vectors must be the same length!")
    preds,gt = preds.tolist(),gt.tolist()
    sse = sum([(gt[i] - preds[i])**2 for i in range(len(gt))])
    return sse/len(gt)

def gcv(preds, gt):
    """Calculates the generalized cross-validation approximation"""
    if len(preds) != len(gt): return ValueError("predictions and ground-truth vectors must be the same length!")
    preds,gt = preds.tolist(),gt.tolist()
    #q = model params. can get from m object with m.get_params() after m.fit(). best way to do this? global models? idk.
    sse = sum([((gt[i] - preds[i])/(1-(q/len(gt))))**2 for i in range(len(gt))])
    return sse/len(gt)