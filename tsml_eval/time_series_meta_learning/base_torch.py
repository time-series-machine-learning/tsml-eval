import os.path
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from aeon.classification.base import BaseClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state


class BaseDeepClassifier_Pytorch(BaseClassifier):
    """
    Abstract base class for deep learning time series classifiers using PyTorch.

    This class builds on BaseClassifier, integrating PyTorch-based deep learning
    methods for training, prediction, and probability estimation.

    Parameters
    ----------
    batch_size : int, default=40
        Training batch size for the model.
    random_state : int or None, default=None
        Random state for reproducibility.
    last_file_name : str, default="last_model"
        The name of the file for saving the last trained model.
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non_deterministic": True,
        "cant_pickle": True,
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        batch_size: int = 32,
        random_state: Optional[int] = None,
        last_file_name: str = "last_model",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.random_state = random_state
        self.last_file_name = last_file_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.history = None
        self.data_utils = None

    @abstractmethod
    def build_model(self, input_channel: int, n_classes: int) -> nn.Module:
        """
        Define the PyTorch model architecture.

        Parameters
        ----------
        input_channel : int
            channel of the input data (channels, timepoints).
        n_classes : int
            Number of classes for classification.

        Returns
        -------
        model : nn.Module
            The PyTorch model.
        """
        ...

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data."""
        probs = self._predict_proba(X)
        return np.argmax(probs, axis=1)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input data."""
        self.model.eval()
        X_tensor = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def convert_y_to_torch(self, y: np.ndarray) -> torch.Tensor:
        """Convert y to the required one-hot PyTorch format."""
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        y = y.reshape(len(y), 1)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        y = onehot_encoder.fit_transform(y)
        return torch.tensor(y, dtype=torch.float32).to(self.device)

    def save_model_to_file(self, model, file_name="last_model.pth"):
        """Save the trained PyTorch model."""
        model_path = os.path.join(self.file_path, file_name)
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, file_name: str):
        """Load a pre-trained PyTorch model."""
        model_path = os.path.join(self.file_path, file_name)
        model.load_state_dict(torch.load(model_path))
        return model

    @staticmethod
    def summary_model(model, input_size):
        """
        Prints a summary of the PyTorch model.
        Args:
            model: The PyTorch model (nn.Module).
            input_size: Tuple representing the size of the input tensor (e.g., (batch_size, channels, timepoints)).
        """

        def register_hook(module):
            def hook(module, inputs, outputs):
                module_name = module.__class__.__name__
                module_str = f"{module_name:>20}"
                input_shape = str(list(inputs[0].size()))
                output_shape = str(list(outputs.size()))
                num_params = sum(p.numel() for p in module.parameters())
                summary_lines.append(
                    f"{module_str:>20} | {input_shape:>20} -> {output_shape:>20} | Params: {num_params}"
                )

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and module != model
            ):
                hooks.append(module.register_forward_hook(hook))

        hooks = []
        summary_lines = [
            "{:<25} {:<25} {:<25} {:<10}".format(
                "Layer", "Input Shape", "Output Shape", "Param #"
            )
        ]

        # Register hooks
        model.apply(register_hook)

        # Generate dummy input and do a forward pass
        dummy_input = torch.randn(*input_size)
        model(dummy_input)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Print the summary
        print("\n".join(summary_lines))
