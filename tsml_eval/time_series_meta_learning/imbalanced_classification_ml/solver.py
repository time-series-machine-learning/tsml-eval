import gc
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data.dataloader import DataLoader

from tsml_eval.time_series_meta_learning.base_torch import BaseDeepClassifier_Pytorch
from tsml_eval.time_series_meta_learning.imbalanced_classification_ml.data import (
    PairDataset,
    Task_Data,
    TripletDataset,
)
from tsml_eval.time_series_meta_learning.imbalanced_classification_ml.model import (
    TimeSeriesSiameseNetwork,
)


class Siamese_Classifier(BaseDeepClassifier_Pytorch):
    """Siamese Networks (Meta-learning) PyTorch Classifier for Time Series Data."""

    def __init__(
        self,
        problem_path,
        dataset,
        resample_id,
        predefined_resample,
        datasetlists,
        callbacks=None,
        batch_size=32,
        file_path="./",
        save_best_model=False,
        save_last_model=False,
        save_init_model=False,
        best_file_name="best_model",
        last_file_name="last_model",
        init_file_name="init_model",
        verbose=False,
        loss=None,
        metrics="accuracy",
        random_state=None,
        optimizer=None,
        meta_train_model=True,
        meta_train_loop=20000,
        n_features=100,
        n_epochs=10,
        n_classes=2,
        distance_metric="euclidean",
    ):
        super().__init__(
            batch_size=batch_size,
            random_state=random_state,
            last_file_name=last_file_name,
        )
        self.problem_path = problem_path
        self.dataset = dataset
        self.n_classes = n_classes
        self.resample_id = resample_id
        self.predefined_resample = predefined_resample
        self.datasetlists = datasetlists

        self.data_utils = Task_Data(
            problem_path=problem_path,
            dataset=dataset,
            resample_id=resample_id,
            predefined_resample=predefined_resample,
            datasetlists=datasetlists,
            K_support=5,
            K_Query=5,
        )
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.meta_train_model = meta_train_model
        self.meta_train_loop = meta_train_loop
        self.model = None
        self.distance_metric = distance_metric

        self.callbacks = callbacks
        self.file_path = file_path
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        self.save_init_model = save_init_model
        self.best_file_name = best_file_name
        self.init_file_name = init_file_name
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.support_data = None
        self.support_labels = None

        self.history = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self, input_channel: int, n_features: int) -> nn.Module:
        model = TimeSeriesSiameseNetwork(
            input_channels=input_channel, embedding_size=n_features
        )
        return model

    def meta_train(self):
        """
        Meta-train the model.
        Using the meta-learning approach, the model is trained on a set of tasks.
        The task is constructed by the UCR dataset except the dataset used to fit and test.

        Returns
        -------
        - history: A dictionary containing training and validation loss and acc.
        """
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        os.makedirs(self.file_path, exist_ok=True)
        model = self.build_model(input_channel=1, n_features=self.n_features)
        criterion = nn.BCEWithLogitsLoss() if self.loss is None else self.loss
        optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=0.001)
            if self.optimizer is None
            else self.optimizer
        )
        model = model.to(self.device)

        for epoch in range(self.meta_train_loop):
            (
                support_data,
                support_labels,
                query_data,
                query_labels,
                val_data,
                val_labels,
            ) = self.data_utils.get_meta_train_task(val=True)
            train_dataset = PairDataset(
                support_data, query_data, support_labels, query_labels
            )
            val_dataset = PairDataset(
                support_data, val_data, support_labels, val_labels
            )
            dataloaders = {
                "train": DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True
                ),
                "val": DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False
                ),
            }

            # Training phase
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for support_series, query_series, labels in dataloaders["train"]:
                support_series, query_series, labels = (
                    support_series.to(self.device),
                    query_series.to(self.device),
                    labels.to(self.device),
                )
                support_embeddings, query_embeddings = model(
                    support_series, query_series
                )
                preds = torch.sigmoid(
                    torch.norm(support_embeddings - query_embeddings, dim=1)
                )
                loss = criterion(preds, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * support_series.size(0)
                total += labels.size(0)
                correct += ((preds > 0.5) == labels).sum().item()

            train_loss = running_loss / len(train_dataset)
            train_acc = correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation phase
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for support_series, query_series, labels in dataloaders["val"]:
                    support_series, query_series, labels = (
                        support_series.to(self.device),
                        query_series.to(self.device),
                        labels.to(self.device),
                    )
                    support_embeddings, query_embeddings = model(
                        support_series, query_series
                    )
                    preds = torch.sigmoid(
                        torch.norm(support_embeddings - query_embeddings, dim=1)
                    )
                    loss = criterion(preds, labels)

                    # Statistics
                    val_loss += loss.item() * support_series.size(0)
                    total += labels.size(0)
                    correct += ((preds > 0.5) == labels).sum().item()
                val_loss = val_loss / len(val_dataset)
                val_acc = correct / total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model_to_file(
                    model=model, file_name="meta_trained_best_model.pth"
                )

            # Save immediate model every 1000 epochs
            if (epoch + 1) % 1000 == 0:
                self.save_model_to_file(
                    model=model,
                    file_name=f"meta_trained_immediate_model_epoch_{epoch + 1}.pth",
                )

        # Save the last model
        self.save_model_to_file(model=model, file_name="meta_trained_last_model.pth")

        self.history = history

        gc.collect()
        return self

    def _fit(self, X, y):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_cases, n_channels, n_timepoints)
        y : np.ndarray
            The training data class labels of shape (n_cases,).


        Returns
        -------
        self : object
        """
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        model = self.build_model(input_channel=X.shape[1], n_features=self.n_features)
        if self.meta_train_model:
            model = self.load_model(
                model=model, file_name="meta_trained_last_model.pth"
            )

        criterion = nn.BCEWithLogitsLoss() if self.loss is None else self.loss
        optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=0.001)
            if self.optimizer is None
            else self.optimizer
        )
        model = model.to(self.device)

        for epoch in range(self.n_epochs):
            support_data, support_labels, query_data, query_labels = (
                self.data_utils.sample_data(data=X, labels=y)
            )
            num_val_data = int(0.3 * len(query_data))
            assert num_val_data >= 1
            val_data, val_labels = (
                query_data[:num_val_data],
                query_labels[:num_val_data],
            )
            query_data, query_labels = (
                query_data[num_val_data:],
                query_labels[num_val_data:],
            )
            train_dataset = PairDataset(
                support_data, query_data, support_labels, query_labels
            )
            val_dataset = PairDataset(
                support_data, val_data, support_labels, val_labels
            )
            dataloaders = {
                "train": DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True
                ),
                "val": DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False
                ),
            }

            # Training phase
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for support_series, query_series, labels in dataloaders["train"]:
                support_series, query_series, labels = (
                    support_series.to(self.device),
                    query_series.to(self.device),
                    labels.to(self.device),
                )
                support_embeddings, query_embeddings = model(
                    support_series, query_series
                )
                preds = torch.sigmoid(
                    torch.norm(support_embeddings - query_embeddings, dim=1)
                )
                loss = criterion(preds, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * support_series.size(0)
                total += labels.size(0)
                correct += ((preds > 0.5) == labels).sum().item()

            train_loss = running_loss / len(train_dataset)
            train_acc = correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation phase
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for support_series, query_series, labels in dataloaders["val"]:
                    support_series, query_series, labels = (
                        support_series.to(self.device),
                        query_series.to(self.device),
                        labels.to(self.device),
                    )
                    support_embeddings, query_embeddings = model(
                        support_series, query_series
                    )
                    preds = torch.sigmoid(
                        torch.norm(support_embeddings - query_embeddings, dim=1)
                    )
                    loss = criterion(preds, labels)

                    # Statistics
                    val_loss += loss.item() * support_series.size(0)
                    total += labels.size(0)
                    correct += ((preds > 0.5) == labels).sum().item()
            val_loss = val_loss / len(val_dataset)
            val_acc = correct / total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model_to_file(model=model, file_name="best_model.pth")

        # Save the last model
        self.save_model_to_file(model=model, file_name="last_model.pth")

        self.history = history
        self.support_data = X
        self.support_labels = y
        model = self.build_model(input_channel=X.shape[1], n_features=self.n_features)
        self.model = self.load_model(model=model, file_name="last_model.pth")
        gc.collect()
        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input data."""
        self.model.to(self.device)
        self.model.eval()
        support_data = torch.tensor(self.support_data).to(self.device)
        query_data = torch.tensor(X)
        classes = torch.unique(self.support_labels)
        n_classes = len(classes)
        probs = torch.zeros(
            query_data.size(0), n_classes, device=self.device
        )  # (n_query, n_classes)
        with torch.no_grad():
            support_embeddings = self.model.forward_once(support_data)
            for start_idx in range(0, query_data.size(0), 10):
                end_idx = min(start_idx + 10, query_data.size(0))
                query_batch = query_data[start_idx:end_idx].to(self.device)
                query_embeddings = self.model.forward_once()
                distances = torch.cdist(
                    query_embeddings, support_embeddings, p=2
                )  # (batch_size, n_support)
                similarities = -distances
                normalized_similarities = torch.softmax(similarities, dim=1)
                batch_probs = torch.zeros(
                    query_batch.size(0), n_classes, device=self.device
                )  # (batch_size, n_classes)

                for i, cls in enumerate(classes):
                    class_mask = (self.support_labels == cls).float()  # (n_support,)
                    class_similarities = normalized_similarities * class_mask.unsqueeze(
                        0
                    )
                    batch_probs[:, i] = class_similarities.sum(dim=1)
                # Store batch probabilities
                probs[start_idx:end_idx] = batch_probs
        return probs.cpu().numpy()
