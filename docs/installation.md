# Installation

## Install from PyPi

The easiest way to install ``tsml-eval`` is using ``pip``:

```{code-block} console
pip install tsml-eval
```

Some estimators require additional dependencies. You can install these individually as
required, or install all of them using the ``all_extras`` extra dependency set:

```{code-block} console
pip install tsml-eval[all_extras]
```

All extra dependency sets can be found in the [pyproject.toml](https://github.com/time-series-machine-learning/tsml-eval/blob/main/pyproject.toml)
file ``[project.optional-dependencies]`` options.

To install a specific [release](https://github.com/time-series-machine-learning/tsml-eval/releases)
version, specify the version number when installing:

```{code-block} console
pip install tsml-eval==0.1.0
```

```{code-block} console
pip install tsml-eval[all_extras]==0.1.0
```

## Install fixed dependency versions for a publication

``tsml-eval`` [publications](publications.md) contain a ``requirements.txt`` file that
lists the versions of all  dependencies used to generate results at the time of the
release.

To install the dependencies using this file, run:

```{code-block} console
pip install -r requirements.txt
```

## Install latest in-development version from GitHub

The latest development version of ``tsml-eval`` can be installed directly from GitHub
using ``pip``:

```{code-block} console
pip install git+https://github.com/time-series-machine-learning/tsml-eval.git@main
```

## Install for developers

To install ``tsml-eval`` for development, first clone the GitHub repository:

```{code-block} console
git clone https://github.com/time-series-machine-learning/tsml-eval.git
```

Then install the package in editable mode with developer dependencies:

```{code-block} console
pip install --editable .[dev]
```

We recommend setting up pre-commit hooks to automatically format code and check for
common issues before committing:

```{code-block} console
pre-commit install
```
