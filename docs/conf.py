# -*- coding: utf-8 -*-
"""Configuration file for the Sphinx documentation builder."""
import inspect
import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
ON_READTHEDOCS = os.environ.get("READTHEDOCS") == "True"
if not ON_READTHEDOCS:
    sys.path.insert(0, os.path.abspath("../"))
RTD_VERSION = os.environ.get("READTHEDOCS_VERSION", "local")

import tsml_estimator_evaluation


# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tsml-estimator-evaluation"
copyright = "2022, Matthew Middlehurst"
author = "Matthew Middlehurst"
version = tsml_estimator_evaluation.__version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_gallery.load_style",
    "numpydoc",
    "nbsphinx",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", ".ipynb_checkpoints", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


def linkcode_resolve(domain, info):
    """Return URL to source code for sphinx.ext.linkcode."""

    def find_source():
        # try to find the file and line number, used in sktime and tslearn conf.py.
        # originally based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L393
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(
            fn, start=os.path.dirname(tsml_estimator_evaluation.__file__)
        )
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "tsml_estimator_evaluation/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"

    return (
        "https://github.com/time-series-machine-learning/tsml-estimator-evaluation"
        "/blob/%s/%s"
        % (
            version,
            filename,
        )
    )
