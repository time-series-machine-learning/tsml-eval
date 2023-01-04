.. _installation:

Installation
============

Install from PyPi
-----------------

The easiest way to install ``tsml-eval`` is using ``pip``:

.. code-block:: bash

    pip install tsml-eval

To install a specific `release <https://github.com/time-series-machine-learning/
tsml-eval/releases>`_ version, specify the version number when installing:

.. code-block:: bash

    pip install tsml-eval==0.1.0

Install fixed dependency versions for a release
-----------------------------------------------

``tsml-eval`` `releases <https://github.com/time-series-machine-learning/tsml-eval
/releases>`_ contain a ``requirements.txt`` file that lists the versions of all
dependencies used to generate results at the time of the release.

To install the dependencies using this file, run:

.. code-block:: bash

    pip install -r requirements.txt

Install latest in-development version from GitHub
-------------------------------------------------

The latest development version of ``tsml-eval`` can be installed directly from GitHub
using ``pip``:

.. code-block:: bash

    pip install git+https://github.com/time-series-machine-learning/tsml-eval.git@main

Install for developers
----------------------

To install ``tsml-eval`` for development, first clone the GitHub repository:

.. code-block:: bash

    git clone https://github.com/time-series-machine-learning/tsml-eval.git

Then install the package in editable mode with developer dependencies:

.. code-block:: bash

    pip install --editable .[dev]

We recommend setting up pre-commit hooks to automatically format code and check for
common issues before committing:

.. code-block:: bash

    pre-commit install
