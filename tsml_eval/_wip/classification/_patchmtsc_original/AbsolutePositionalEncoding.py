import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

"""class tAPE(nn.Module):

    def __init__(self, config, d_model_patch, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.patch_len = config['patch_len']
        self.stride = config['stride']
        self.padding_patch = config['padding_patch']
        # self.context_window = config['seq_len']
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        self.context_window = seq_len
        patch_num = int((self.context_window - self.patch_len) / self.stride + 1)
        patch_num += 1

        if patch_num % 2 == 0:
            patch_num += 0
        else:
            patch_num += 1

        if patch_num % 2 == 0:
            pass
        else:
            print("context_window:", self.context_window)
            print("patch_len:", self.patch_len)
            print("stride:", self.stride)
            print("patch_num:", patch_num)
            assert patch_num % 2 == 0, "Wrong parameters: patch_len!"


        #d_model是tensor在网络处理时的维度，max_len是序列的长度
        pe = torch.zeros(self.patch_len, patch_num)  # positional encoding #[8, 13]

        #pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, self.patch_len, dtype=torch.float).unsqueeze(1) #[8, 1] 8是序列长度
        # torch.arange(0, patch_num, 2): 从 0 到 d_model-1 之间的等差数列张量,步长为 2。
        div_term = torch.exp(torch.arange(0, patch_num, 2).float() * (-math.log(10000.0) / patch_num)) #[7], 7是维度的一半

        # ::2 表示选择从第 0 列开始,每隔 2 列的元素
        #当patch_num为奇数时会报错
        pe[:, 0::2] = torch.sin((position * div_term)*(self.patch_len/patch_num)) #[0., 2., 4., 6., 8.]
        pe[:, 1::2] = torch.cos((position * div_term)*(self.patch_len/patch_num)) #[1., 3., 5., 7., 9.]
        pe = scale_factor * pe.unsqueeze(0)

        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x): # [96, 12, 16]
        x = x + self.pe
        return self.dropout(x)"""

class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding #[100, 16]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  #[100, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #[8]

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len)) #d_model=16/max_len=100
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):

        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)



class AbsolutePositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

        # distance = torch.matmul(self.pe, self.pe[10])
        # import matplotlib.pyplot as plt

        # plt.plot(distance.detach().numpy())
        # plt.show()

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe
        # distance = torch.matmul(self.pe, self.pe.transpose(1,0))
        # distance_pd = pd.DataFrame(distance.cpu().detach().numpy())
        # distance_pd.to_csv('learn_position_distance.csv')
        return self.dropout(x)