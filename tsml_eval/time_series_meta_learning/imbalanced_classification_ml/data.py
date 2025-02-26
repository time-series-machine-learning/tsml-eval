import os

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from tsml_eval.utils.datasets import load_experiment_data
from tsml_eval.utils.resampling import (
    make_imbalance,
    resample_data,
    stratified_resample_data,
)


class Task_Data:
    """
    docstring for Task_Data
    for imbalanced classification task data construction
    output: Meta_train(Support set, Query set), Meta_test(Support set, Query set) data
    """

    def __init__(
        self,
        problem_path=r"C:\Users\cq2u24\OneDrive - University of Southampton\Documents\Downloads\Data\Data",
        dataset="OSULeaf",
        resample_id=0,
        predefined_resample=False,
        datasetlists=r"C:\Users\cq2u24\OneDrive - University of Southampton\Documents\Downloads\Data\classification10.txt",
        K_support=5,
        K_Query=5,
    ):
        self.problem_path = problem_path
        self.dataset = dataset

        self.resample_id = resample_id
        self.predefined_resample = predefined_resample

        self.datasetlists = self.exclude_current_dataset(datasetlists, self.dataset)
        self.K_support = K_support
        self.K_query = K_Query

        self.base_rng = check_random_state(self.resample_id)

    @staticmethod
    def exclude_current_dataset(file_path, current_dataset):
        """
        Reads a file containing datasets line-by-line and generates a list excluding the current dataset.

        Parameters
        ----------
            file_path (str): Path to the dataset list file.
            current_dataset (str): The dataset to exclude.

        Returns
        -------
            list: A list of datasets excluding the current dataset.
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read datasets and exclude the current one
        with open(file_path) as file:
            datasets = [
                line.strip()
                for line in file
                if line.strip() and line.strip() != current_dataset
            ]
        return datasets

    def construct_data(self, dataset):
        """
        Construct dataset and combine the train and test data to a single dataset

        Returns
        -------
            tuple: dataset and labels
        """
        X_train, y_train, X_test, y_test, resample = load_experiment_data(
            self.problem_path, dataset, self.resample_id, self.predefined_resample
        )
        if isinstance(X_train, np.ndarray):
            is_array = True
        elif isinstance(X_train, list):
            is_array = False
        else:
            raise ValueError(
                "X_train must be a np.ndarray array or list of np.ndarray arrays"
            )

        # add both train and test to a single dataset
        all_labels = np.concatenate((y_train, y_test), axis=None)
        all_data = (
            np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
        )
        return all_data, all_labels

    @staticmethod
    def sample_data(
        data,
        labels,
        n_label_1_support=5,
        n_label_0_support=10,
        n_label_1_query=5,
        n_label_0_query=10,
        base_rng=None,
    ):
        """
        Randomly sample data points with specific label counts and split into support and query sets.

        Parameters
        ----------
            data (array-like): The dataset.
            labels (array-like): Corresponding labels (binary: '0' and '1').
            n_label_1_support (int): Total number of label '1' support samples to pick.
            n_label_0_support (int): Total number of label '0' support samples to pick.
            n_label_1_query (int): Total number of label '1' query samples to pick.
            n_label_0_query (int): Total number of label '0' query samples to pick.

        Returns
        -------
            tuple: (support_data, support_labels, query_data, query_labels)
        """
        # Generate a unique seed for this sampling
        task_seed = base_rng.randint(0, 2**16 - 1)
        rng = check_random_state(task_seed)

        # Separate data based on labels
        if isinstance(labels[0], str):
            label_1_indices = np.where(labels == "1")[0]
            label_0_indices = np.where(labels == "0")[0]
        elif isinstance(labels[0], (int, float)):
            label_1_indices = np.where(labels == 1)[0]
            label_0_indices = np.where(labels == 0)[0]
        else:
            raise ValueError("Unknown label format")

        assert (
            len(label_1_indices) > n_label_1_support + 1
            and len(label_0_indices) > n_label_0_support
        )

        # Sample the required number of indices for each label
        size1 = (
            n_label_1_support + n_label_1_query
            if n_label_1_support + n_label_1_query <= len(label_1_indices)
            else len(label_1_indices)
        )
        size0 = (
            n_label_0_support + n_label_0_query
            if n_label_0_support + n_label_0_query <= len(label_0_indices)
            else len(label_0_indices)
        )

        sampled_label_1_indices = rng.choice(label_1_indices, size=size1, replace=False)
        sampled_label_0_indices = rng.choice(label_0_indices, size=size0, replace=False)

        # Split into support and query sets
        support_label_1_indices = sampled_label_1_indices[:n_label_1_support]
        query_label_1_indices = sampled_label_1_indices[n_label_1_support:]

        support_label_0_indices = sampled_label_0_indices[:n_label_0_support]
        query_label_0_indices = sampled_label_0_indices[n_label_0_support:]

        # Combine the indices for support and query sets
        support_indices = np.concatenate(
            [support_label_1_indices, support_label_0_indices]
        )
        query_indices = np.concatenate([query_label_1_indices, query_label_0_indices])

        # Shuffle the indices within each set
        rng.shuffle(support_indices)
        rng.shuffle(query_indices)

        # Extract the data and labels for support and query sets
        support_data = data[support_indices]
        support_labels = labels[support_indices]

        query_data = data[query_indices]
        query_labels = labels[query_indices]

        return support_data, support_labels, query_data, query_labels

    def get_meta_train_task(self, val=False):
        """
        Construct the meta-train task data
        val: bool, whether to use validation data or not if True,
             the query set will be split into query and validation set

        Returns
        -------
            tuple: Meta-train support set and query set data
        """
        task_seed = self.base_rng.randint(0, 2**16 - 1)
        rng = check_random_state(task_seed)
        dataset = rng.choice(self.datasetlists)
        data, labels = self.construct_data(dataset)
        support_data, support_labels, query_data, query_labels = self.sample_data(
            data,
            labels,
            self.K_support,
            3 * self.K_support,
            self.K_query,
            3 * self.K_query,
        )
        if val:
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
            return (
                support_data,
                support_labels,
                query_data,
                query_labels,
                val_data,
                val_labels,
            )
        else:
            return support_data, support_labels, query_data, query_labels

    def get_meta_train_data(self):
        task_seed = self.base_rng.randint(0, 2**16 - 1)
        rng = check_random_state(task_seed)
        dataset = rng.choice(self.datasetlists)
        data, labels = self.construct_data(dataset)
        return data, labels

    def get_meta_test_task(self, imbalance_ratio=None):
        """
        Construct the meta-test task data

        Returns
        -------
            tuple: Meta-test support set and query set data
        """
        if imbalance_ratio is None:
            imbalance_ratio = [90, 10]
        X_train, y_train, X_test, y_test, resample = load_experiment_data(
            self.problem_path, self.dataset, self.resample_id, self.predefined_resample
        )

        if resample:
            X_train, y_train, X_test, y_test = stratified_resample_data(
                X_train, y_train, X_test, y_test, random_state=self.resample_id
            )

        X_train, y_train = make_imbalance(
            X_train,
            y_train,
            sampling_ratio=imbalance_ratio,
            random_state=self.resample_id,
        )
        X_test, y_test = make_imbalance(
            X_test,
            y_test,
            sampling_ratio=imbalance_ratio,
            random_state=self.resample_id,
        )

        return X_train, y_train, X_test, y_test


class PairDataset(Dataset):
    def __init__(self, support_data, query_data, support_labels, query_labels):
        """
        Parameters
        ----------
        - support_data: Tensor of support set time series (N_support, channels, timepoints).
        - query_data: Tensor of query set time series (N_query, channels, timepoints).
        - support_labels: Tensor of support set labels (N_support,).
        - query_labels: Tensor of query set labels (N_query,).
        """
        self.support_data = support_data
        self.support_labels = support_labels
        self.query_data = query_data
        self.query_labels = query_labels

        self.n_support = support_data.size(0)
        self.n_query = query_data.size(0)

    def __len__(self):
        return self.n_query * self.n_support

    def __getitem__(self, index):
        # Compute indices for support and query samples
        query_idx = index // self.n_support
        support_idx = index % self.n_support

        query_sample = self.query_data[query_idx]
        support_sample = self.support_data[support_idx]
        label = torch.tensor(
            self.query_labels[query_idx] == self.support_labels[support_idx]
        ).long()

        return (
            torch.tensor(support_sample).float(),
            torch.tensor(query_sample).float(),
            label,
        )


class TripletDataset(Dataset):
    def __init__(self, data, labels, resample_id=None):
        """
        Parameters
        ----------
        - data: Tensor of input time series data (N_samples, channels, timepoints).
        - labels: Tensor of class labels (N_samples,).
        - resample_id: int, seed for random state to ensure reproducibility.
        """
        self.data = data
        self.labels = labels
        self.resample_id = resample_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Set random seed for reproducibility
        if self.resample_id is not None:
            torch.manual_seed(self.resample_id + idx)

        # Anchor
        anchor = self.data[idx]
        anchor_label = self.labels[idx]

        # Positive: Randomly sample from the same class
        positive_idx = torch.randint(0, len(self.data), (1,)).item()
        while self.labels[positive_idx] != anchor_label or positive_idx == idx:
            positive_idx = torch.randint(0, len(self.data), (1,)).item()
        positive = self.data[positive_idx]

        # Negative: Randomly sample from a different class
        negative_idx = torch.randint(0, len(self.data), (1,)).item()
        while self.labels[negative_idx] == anchor_label:
            negative_idx = torch.randint(0, len(self.data), (1,)).item()
        negative = self.data[negative_idx]

        return anchor, positive, negative
