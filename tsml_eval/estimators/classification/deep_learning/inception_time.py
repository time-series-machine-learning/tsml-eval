# -*- coding: utf-8 -*-
"""InceptionTime classifier."""

__author__ = ["James-Large", "TonyBagnall", "MatthewMiddlehurst"]

import numpy as np
from aeon.classification.base import BaseClassifier
from aeon.utils.validation._dependencies import _check_dl_dependencies
from sklearn.utils import check_random_state

from tsml_eval.estimators.networks.base_classifier import BaseDeepClassifier
from tsml_eval.estimators.networks.inception_time import InceptionTimeNetwork


class InceptionTimeClassifier(BaseClassifier):
    """InceptionTime ensemble classifier.

    Ensemble of IndividualInceptionTimeClassifiers, as desribed in [1].

    Parameters
    ----------
    n_classifiers=5,
    n_filters: int,
    use_residual: boolean,
    use_bottleneck: boolean,
    depth: int
    kernel_size: int, specifying the length of the 1D convolution
     window
    batch_size: int, the number of samples per gradient update.
    bottleneck_size: int,
    nb_epochs: int, the number of epochs to train the model
    callbacks: list of tf.keras.callbacks.Callback objects
    random_state: int, seed to any needed random actions
    verbose: boolean, whether to output extra information
    model_name: string, the name of this model for printing and
     file writing purposes
    model_save_directory: string, if not None; location to save
     the trained keras model in hdf5 format

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    """

    _tags = {"capability:multivariate": True}

    def __init__(
        self,
        n_classifiers=5,
        n_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        kernel_size=40,
        batch_size=64,
        nb_epochs=1500,
        callbacks=None,
        random_state=0,
        verbose=False,
    ):
        self.n_classifiers = n_classifiers
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.classifers_ = []

        super(InceptionTimeClassifier, self).__init__()

    def _fit(self, X, y):
        self.classifers_ = []
        rng = check_random_state(self.random_state)

        for _ in range(0, self.n_classifiers):
            cls = IndividualInceptionTimeClassifier(
                n_filters=self.n_filters,
                use_bottleneck=self.use_bottleneck,
                bottleneck_size=self.bottleneck_size,
                depth=self.depth,
                kernel_size=self.kernel_size,
                batch_size=self.batch_size,
                nb_epochs=self.nb_epochs,
                callbacks=self.callbacks,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                verbose=self.verbose,
            )
            cls.fit(X, y)
            self.classifers_.append(cls)

        return self

    def _predict(self, X) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        probs = np.zeros((X.shape[0], self.n_classes_))

        for cls in self.classifers_:
            probs += cls.predict_proba(X)

        probs = probs / self.n_classifiers

        return probs

    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        param1 = {
            "n_classifiers": 2,
            "n_epochs": 10,
            "batch_size": 4,
            "kernel_size": 4,
            "use_residual": False,
            "use_bottleneck": True,
        }

        param2 = {
            "n_classifiers": 3,
            "n_epochs": 12,
            "batch_size": 6,
            "use_bias": True,
        }

        return [param1, param2]


class IndividualInceptionTimeClassifier(BaseDeepClassifier, InceptionTimeNetwork):
    """Single InceptionTime classifier.

    Parameters
    ----------
    n_filters: int, default = 32
    use_residual: boolean, default = True
    use_bottleneck: boolean, default = True
    bottleneck_size: int, default = 32
    depth: int, default = 6
    kernel_size: int, default = 40
        specifies the length of the 1D convolution window.
    batch_size: int, default = 64
        the number of samples per gradient update.
    nb_epochs: int, default = 1500
        the number of epochs to train the model.
    callbacks: callable or None, default None
        list of tf.keras.callbacks.Callback objects.
    random_state: int, default = 0
        seed to any needed random actions.
    verbose: boolean, default = False
        whether to output extra information

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    """

    def __init__(
        self,
        n_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        kernel_size=40,
        batch_size=64,
        nb_epochs=1500,
        callbacks=None,
        random_state=0,
        verbose=False,
    ):
        _check_dl_dependencies(severity="error")
        super(IndividualInceptionTimeClassifier, self).__init__()
        # predefined
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose

    def build_model(self, input_shape, n_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output
             layer

        Returns
        -------
        output : a compiled Keras Model
        """
        from tensorflow import keras

        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(n_classes, activation="softmax")(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        if self.callbacks is None:
            self.callbacks = []

        if not any(
            isinstance(callback, keras.callbacks.ReduceLROnPlateau)
            for callback in self.callbacks
        ):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
            self.callbacks.append(reduce_lr)

        return model

    def _fit(self, X, y):
        """
        Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = (n_instances, n_dimensions, series_length)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        validation_X : a nested pd.Dataframe, or array-like of shape =
        (n_instances, series_length, n_dimensions)
            The validation samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
            Unless strictly defined by the user via callbacks (such as
            EarlyStopping), the presence or state of the validation
            data does not alter training in any way. Predictions at each epoch
            are stored in the model's fit history.
        validation_y : array-like, shape = [n_instances]
            The validation class labels.

        Returns
        -------
        self : object
        """
        self.random_state = check_random_state(self.random_state)
        y_onehot = self.convert_y_to_keras(y)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        if self.batch_size is None:
            self.batch_size = int(min(X.shape[0] / 10, 16))
        else:
            self.batch_size = self.batch_size
        self.model_ = self.build_model(self.input_shape, self.n_classes_)

        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )

        #        self.save_trained_model()
        #        self._is_fitted = True

        return self

    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        param1 = {
            "n_epochs": 10,
            "batch_size": 4,
            "kernel_size": 4,
            "use_residual": False,
            "use_bottleneck": True,
        }

        param2 = {
            "n_epochs": 12,
            "batch_size": 6,
            "use_bias": True,
        }

        return [param1, param2]
