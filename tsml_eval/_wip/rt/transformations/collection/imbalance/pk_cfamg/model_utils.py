import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def resample_from_normal(mean, var):
    epsilon = 1e-6
    qz_gaussian = torch.distributions.Normal(loc=mean, scale=var + epsilon)
    qz = qz_gaussian.rsample()
    return qz


def adjust_input(X_train, X_test, dim_need='2'):
    if dim_need == '2':
        if len(X_train.shape) > 2 or len(X_test.shape) > 2:
            if isinstance(X_train, torch.Tensor) or isinstance(X_test, torch.Tensor):
                X_train = X_train.squeeze(1)
                X_test = X_test.squeeze(1)
            elif isinstance(X_train, np.ndarray) or isinstance(X_test, np.ndarray):
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
    elif dim_need == '3':
        if len(X_train.shape) < 3 or len(X_test.shape) < 3:
            if isinstance(X_train, torch.Tensor) or isinstance(X_test, torch.Tensor):
                X_train = X_train.unsqueeze(1)
                X_test = X_test.unsqueeze(1)
            elif isinstance(X_train, np.ndarray) or isinstance(X_test, np.ndarray):
                X_train = np.expand_dims(X_train, axis=1)
                X_test = np.expand_dims(X_test, axis=1)
    return X_train, X_test


def computer_f1_gmeans_auc(y_test, y_pred, y_proba):
    f1 = f1_score(y_test, y_pred, average='binary')

    AUC = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
    return f1, g_mean, AUC
