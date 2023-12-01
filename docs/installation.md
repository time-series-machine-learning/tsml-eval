# Installation

The following contains information on installing `tsml-eval` for users and developers
with write access. Those who wish to contribute to `tsml-eval` without write access
will need to create a fork, see the [aeon](https://www.aeon-toolkit.org/en/stable/developer_guide/dev_installation.html)
and [sklearn](https://scikit-learn.org/stable/developers/contributing.html#how-to-contribute)
documentation on contributing and developer installation for guidance.

We recommend setting up a fresh virtual environment or the conda equivalent before
installing `tsml-eval`. See the [aeon guide](https://www.aeon-toolkit.org/en/stable/installation.html#using-a-pip-venv)
for setup information.

## Install from PyPi

The easiest way to install `tsml-eval` is using `pip`:

```console
pip install tsml-eval
```

Some estimators require additional dependencies. You can install these individually as
required, or install all of them using the `all_extras` extra dependency set:

```console
pip install tsml-eval[all_extras]
```

```{note}
    If this results in a "no matches found" error, it may be due to how your shell
    handles special characters. Try surrounding the dependency portion with quotes i.e.

    pip install tsml-eval"[all_extras]"
```

All extra dependency sets can be found in the [pyproject.toml](https://github.com/time-series-machine-learning/tsml-eval/blob/main/pyproject.toml)
file `[project.optional-dependencies]` options.

To install a specific [release](https://github.com/time-series-machine-learning/tsml-eval/releases)
version, specify the version number when installing:

```console
pip install tsml-eval==0.1.0
```

```console
pip install tsml-eval[all_extras]==0.1.0
```

## Install from conda-forge

`tsml-eval` is also available on [conda-forge](https://anaconda.org/conda-forge/tsml-eval).

```console
conda create -n tsml-env -c conda-forge tsml-eval
conda activate tsml-env
```

Currently for conda installations, optional dependencies must be installed separately.

## Install fixed dependency versions for a publication

`tsml-eval` [publications](publications.md) contain a `static_publication_reqs.txt`
file that lists the versions of all dependencies used to generate results at the time
of the release.

To install the dependencies using this file, run:

```console
pip install -r static_publication_reqs.txt
```

## Install the latest in-development version from GitHub

The latest development version of `tsml-eval` can be installed directly from GitHub
using `pip`:

```console
pip install git+https://github.com/time-series-machine-learning/tsml-eval.git@main
```

The latest development version of dependencies can be installed this way, i.e. for
`aeon`

```console
pip install git+https://github.com/aeon-toolkit/aeon.git@main
```

If you have a different version of `tsml-eval` or a dependency installed, you must
uninstall it first before installing the development version.

## Install for developers with write access

To install `tsml-eval` for development, first clone the GitHub repository:

```console
git clone https://github.com/time-series-machine-learning/tsml-eval.git
```

Then install the package in editable mode with developer dependencies:

```console
pip install --editable .[dev]
```

We recommend setting up pre-commit hooks to automatically format code and check for
common issues before committing:

```console
pre-commit install
```
