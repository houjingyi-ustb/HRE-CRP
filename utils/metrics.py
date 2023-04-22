import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def acc(y_true, y_pred):
    y_true.cpu().numpy()
    y_pred.cpu().numpy()
    score = (y_pred == y_true).mean()

    return score
    
    
def cen(y_true, y_pred):
    y_true.cpu().numpy()
    y_pred.cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    score = (tp / (tp + fn)) / (fp / (tn + fp))
    return score


def mcc(y_true, y_pred):
    y_true.cpu().numpy()
    y_pred.cpu().numpy()
    score = matthews_corrcoef(y_true, y_pred)
    return score

