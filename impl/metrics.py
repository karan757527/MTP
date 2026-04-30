from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import numpy as np


def binaryf1(pred, label):
    '''
    pred, label are numpy array
    can process multi-label target
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return f1_score(label_i, pred_i, average="micro")


def microf1(pred, label):
    '''
    multi-class micro-f1
    '''
    pred_i = np.argmax(pred, axis=1)
    return f1_score(label, pred_i, average="micro")


def auroc(pred, label):
    '''
    calculate auroc
    '''
    return roc_auc_score(label, pred)


def binaryprauc(pred, label):
    '''
    Binary PR-AUC (average precision). Primary metric for the Elliptic2
    ablation study — appropriate for the 42:1 class imbalance where
    ROC-AUC is misleadingly optimistic.
    pred: raw logits or probabilities, shape (N,) or (N, 1).
    label: 0/1 binary labels, shape (N,) or (N, 1).
    '''
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1).astype(np.int64)
    return average_precision_score(label, pred)
