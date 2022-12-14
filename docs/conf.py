# -*- coding: utf-8 -*-
"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# import tsml_estimator_evaluation

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
ON_READTHEDOCS = os.environ.get("READTHEDOCS") == "True"
if not ON_READTHEDOCS:
    sys.path.insert(0, os.path.abspath(".."))
RTD_VERSION = os.environ.get("READTHEDOCS_VERSION", "local")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tsml-estimator-evaluation"
copyright = "2022, Matthew Middlehurst"
author = "Matthew Middlehurst"
# version = tsml_estimator_evaluation.__version__
version = "0.0.1"


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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nature"
html_static_path = ["_static"]
