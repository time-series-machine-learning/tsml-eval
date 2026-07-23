import math
import numpy as np
import torch.nn as nn
import torch


class HiddenLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, model_type, dropout_list=None):
        super().__init__()

        self.mlp = nn.Sequential()
        if model_type == 'encoder':
            if isinstance(hidden_dim, list):
                assert isinstance(dropout_list, list)
                for hidden, dropout in zip(hidden_dim, dropout_list):
                    self.mlp.append(nn.Dropout(dropout))
                    self.mlp.append(nn.Linear(input_dim, hidden))
                    self.mlp.append(nn.ReLU())
                    input_dim = hidden
            else:
                assert not isinstance(dropout_list, list)
                self.mlp.append(nn.Dropout(dropout_list))
                self.mlp.append(nn.Linear(input_dim, hidden_dim))
        elif model_type == 'decoder':
            if isinstance(hidden_dim, list):
                assert isinstance(dropout_list, list)
                hidden_dim = list(reversed(hidden_dim))
                dropout_list = list(reversed(dropout_list))
                for hidden, dropout in zip(hidden_dim, dropout_list):
                    self.mlp.append(nn.Dropout(dropout))
                    self.mlp.append(nn.Linear(input_dim, hidden))
                    self.mlp.append(nn.ReLU())
                    input_dim = hidden
            else:
                assert not isinstance(dropout_list, list)
                self.mlp.append(nn.Dropout(dropout_list))
                self.mlp.append(nn.Linear(input_dim, hidden_dim))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), -1)
        return self.mlp(x)
