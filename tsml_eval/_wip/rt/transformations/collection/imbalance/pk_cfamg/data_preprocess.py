import random
import warnings
from collections import Counter
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(dataset_name=None, root_path=None):
    X_train = np.load(os.path.join(root_path, dataset_name, "X_train.npy"))
    y_train = np.load(os.path.join(root_path, dataset_name, "y_train.npy")).astype(int)
    print(Counter(y_train))
    X_train_pos, X_train_neg = X_train[y_train == 1], X_train[y_train == 0]
    y_train_pos, y_train_neg = y_train[y_train == 1], y_train[y_train == 0]
    ir = X_train_neg.shape[0] / X_train_pos.shape[0]

    print(f"Dataset name: {dataset_name}, The number of Positive sample : {len(y_train_pos)}")

    dataset = {
        "train_data": (X_train, y_train),
        "train_data_pos": (X_train_pos, y_train_pos),
        "train_data_neg": (X_train_neg, y_train_neg),
        "ir": ir,
    }

    return dataset, ir


def set_seed(seed, cudnn_deterministic=False):
    """
    Set all seed
    :param seed: seed
    :param cudnn_deterministic: whether set CUDNN deterministic
    """
    if seed is not None:
        print(f'Global seed set to {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        """
        Even if `torch.backends.cudnn.deterministic` is set to False,
        the reproducibility of other random operations in PyTorch can still be ensured by setting the random seed with
         `torch.manual_seed`.
        """
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! You may see unexpected behavior when restarting '
                      'from checkpoints.')


class MinMaxScaler():
    def fit_transform(self, data):
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


def create_dataLoader2(dataset, batch_size):
    """
    dataset : dict
    data :　tuple (Positive Sample, Negative Sample)
    """

    X_pos, y_pos = dataset["train_data_pos"]
    X_neg, y_neg = dataset["train_data_neg"]
    num_pos, num_neg = X_pos.shape[0], X_neg.shape[0]
    batch_size = batch_size if batch_size < num_pos else num_pos

    assert X_pos.ndim == 3

    X_pos, X_neg = torch.tensor(X_pos, dtype=torch.float32), torch.tensor(X_neg, dtype=torch.float32)
    y_pos, y_neg = torch.tensor(y_pos, dtype=torch.int64), torch.tensor(y_neg, dtype=torch.int64)

    # if num_pos % batch_size == 1 or num_neg % batch_size == 1:
    pos_dataloader = DataLoader(TensorDataset(X_pos, y_pos), batch_size=batch_size, shuffle=False, drop_last=True)
    neg_dataloader = DataLoader(TensorDataset(X_neg, y_neg), batch_size=batch_size, shuffle=False, drop_last=True)
    # else:
    #     pos_dataloader = DataLoader(TensorDataset(X_pos, y_pos), batch_size=batch_size, shuffle=False)
    #     neg_dataloader = DataLoader(TensorDataset(X_neg, y_neg), batch_size=batch_size, shuffle=False)

    _, feat_dim, seq_len = X_pos.shape
    return pos_dataloader, neg_dataloader, feat_dim, seq_len, batch_size
