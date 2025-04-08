# The base classifier aim to use Time-Series-Library https://github.com/thuml/Time-Series-Library
# better in tsml-eval frame

import numpy as np
import gc
import os
import torch
import torch.nn as nn
from torch import optim
from aeon.classification.base import BaseClassifier
from sklearn.utils import check_random_state
from .models import TimesNet
from .ts_data_provider import tsloader



class DeepModelClassifier(BaseClassifier):
    """Ensemble classifier that wraps and uses multiple instances of a deep model classifier."""

    def __init__(self, args, random_state=None):
        self.args = args
        self.random_state = random_state
        self.model_dict = {
            'TimesNet': TimesNet,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        super().__init__()

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        # model input depends on data
        # self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _fit(self, X, y):
        """Fit ensemble of deep model classifiers."""
        rng = check_random_state(self.random_state)


        return self

    def _predict(self, X):
        """Predict class labels using majority voting from ensemble classifiers."""
        probs = self._predict_proba(X)
        return np.array([
            self.classes_[int(np.argmax(prob))] for prob in probs
        ])

    def _predict_proba(self, X):
        """Predict class probabilities by averaging probabilities from ensemble classifiers."""
        probs = np.zeros((X.shape[0], len(self.classes_)))

        for model in self.classifiers_:
            probs += model.predict_proba(X)

        probs /= self.n_classifiers
        return probs
