import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def accuracy(y_true, y_pred):
    y_pred_classes = torch.argmax(y_pred, dim=1)
    return accuracy_score(y_true.cpu().numpy(), y_pred_classes.cpu().numpy())

def auc_score(y_true, y_pred):
    if len(torch.unique(y_true)) == 2:
        return roc_auc_score(y_true.cpu().numpy(), y_pred[:, 1].cpu().numpy())
    else:
        return roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), multi_class='ovr')

def precision(y_true, y_pred, average='weighted'):
    y_pred_classes = torch.argmax(y_pred, dim=1)
    return precision_score(y_true.cpu().numpy(), y_pred_classes.cpu().numpy(), average=average)

def recall(y_true, y_pred, average='weighted'):
    y_pred_classes = torch.argmax(y_pred, dim=1)
    return recall_score(y_true.cpu().numpy(), y_pred_classes.cpu().numpy(), average=average)

def f1(y_true, y_pred, average='weighted'):
    y_pred_classes = torch.argmax(y_pred, dim=1)
    return f1_score(y_true.cpu().numpy(), y_pred_classes.cpu().numpy(), average=average)

def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2).item()

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def classification_error(y_true, y_pred):
    return 1.0 - accuracy(y_true, y_pred)