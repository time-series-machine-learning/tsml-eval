"""Configuration file for the Sphinx documentation builder."""

# tsml-eval documentation master file, created by
# sphinx-quickstart on Wed Dec 14 00:20:27 2022.

import inspect
import os
import sys

import tsml_eval

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

version = tsml_eval.__version__
release = tsml_eval.__version__

github_tag = f"v{version}"

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
on_readthedocs = os.environ.get("READTHEDOCS") == "True"
if not on_readthedocs:
    sys.path.insert(0, os.path.abspath(".."))
else:
    rtd_version = os.environ.get("READTHEDOCS_VERSION")
    if rtd_version == "latest":
        github_tag = "main"


# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tsml-eval"
copyright = "The tsml developers (BSD-3 License)"
author = "Matthew Middlehurst"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinxext.opengraph",
    "numpydoc",
    "nbsphinx",
    "sphinx_design",
    "sphinx_issues",
    "sphinx_copybutton",
    "sphinx_remove_toctrees",
    "sphinx_reredirects",
    "versionwarning.extension",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", ".ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The main toctree document.
master_doc = "index"

# auto doc/summary

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# numpydoc

# see http://stackoverflow.com/q/12206334/562769
numpydoc_show_class_members = True
# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

numpydoc_validation_checks = {"all"}

# Link to GitHub repo for github_issues extension
issues_github_path = "time-series-machine-learning/tsml-eval"

# sphinx-copybutton

copybutton_exclude = ".linenos, .gp, .go"

# sphinx-remove-toctrees configuration
# see https://github.com/pradyunsg/furo/pull/674

remove_from_toctrees = ["auto_generated/*"]

# sphinx-reredirects
redirects = {"redirect": "index"}

# nbsphinx

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_timeout = 600  # seconds, set to -1 to disable timeout

current_file = "{{ env.doc2path(env.docname, base=None) }}"

# add link to original notebook at the bottom and add Binder launch button
# points to latest stable release, not main
notebook_url = f"https://github.com/time-series-machine-learning/tsml-eval/tree/{github_tag}/{current_file}"  # noqa
binder_url = f"https://mybinder.org/v2/gh/time-series-machine-learning/tsml-eval/{github_tag}?filepath={current_file}"  # noqa
nbsphinx_epilog = f"""
----

Generated using nbsphinx_. The Jupyter notebook can be found here_.
|Binder|_

.. _nbsphinx: https://nbsphinx.readthedocs.io/
.. _here: {notebook_url}
.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: {binder_url}
"""

# MyST Parser configuration

# When building HTML using the sphinx.ext.mathjax (enabled by default),
# Myst-Parser injects the tex2jax_ignore (MathJax v2) and mathjax_ignore (MathJax v3)
# classes in to the top-level section of each MyST document, and adds some default
# configuration. This ensures that MathJax processes only math, identified by the
# dollarmath and amsmath extensions, or specified in math directives. We here silence
# the corresponding warning that this override happens.
suppress_warnings = ["myst.mathjax"]

# Recommended by sphinx_design when using the MyST Parser
myst_enable_extensions = ["colon_fence"]

myst_heading_anchors = 2

# linkcode


def linkcode_resolve(domain, info):
    """Return URL to source code for sphinx.ext.linkcode."""

    def find_source():
        # try to find the file and line number, used in aeon and tslearn conf.py.
        # originally based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L393
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(tsml_eval.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "tsml_eval/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"

    return (
        "https://github.com/time-series-machine-learning/tsml-eval/blob/{}/{}".format(
            version,
            filename,
        )
    )


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_show_sourcelink = False

# logos
html_favicon = "images/logo/aeon-favicon.ico"

html_theme_options = {
    "sidebar_hide_name": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/time-series-machine-learning/tsml-eval/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_logo": "tsml-black.png",
    "dark_logo": "tsml-white.png",
    # "light_css_variables": {
    #     "color-brand-primary": "#005E80",
    #     "color-brand-content": "#F05F05",
    # },
    # "dark_css_variables": {
    #     "color-brand-primary": "#00ACEB",
    #     "color-brand-content": "#FB9456",
    # },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/time-series-machine-learning/tsml-eval/",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 1024 1024" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                <path d="M511.6 76.3C264.3 76.2 64 276.4 64 523.5 64 718.9 189.3 885 363.8 946c23.5 5.9 19.9-10.8 19.9-22.2v-77.5c-135.7 15.9-141.2-73.9-150.3-88.9C215 726 171.5 718 184.5 703c30.9-15.9 62.4 4 98.9 57.9 26.4 39.1 77.9 32.5 104 26 5.7-23.5 17.9-44.5 34.7-60.8-140.6-25.2-199.2-111-199.2-213 0-49.5 16.3-95 48.3-131.7-20.4-60.5 1.9-112.3 4.9-120 58.1-5.2 118.5 41.6 123.2 45.3 33-8.9 70.7-13.6 112.9-13.6 42.4 0 80.2 4.9 113.5 13.9 11.3-8.6 67.3-48.8 121.3-43.9 2.9 7.7 24.7 58.3 5.5 118 32.4 36.8 48.9 82.7 48.9 132.3 0 102.2-59 188.1-200 212.9a127.5 127.5 0 0 1 38.1 91v112.5c.8 9 0 17.9 15 17.9 177.1-59.7 304.6-227 304.6-424.1 0-247.2-200.4-447.3-447.5-447.3z"></path>
            </svg>
            """,  # noqa: E501
            "class": "",
        },
        {
            "name": "ReadTheDocs",
            "url": "https://readthedocs.org/projects/tsml-eval/",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" role="img" viewBox="0 0 24 24" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                <path d="M7.732 0a59.316 59.316 0 0 0-4.977.218V24a62.933 62.933 0 0 1 3.619-.687c.17-.028.34-.053.509-.078.215-.033.43-.066.644-.096l.205-.03zm1.18.003V22.96a61.042 61.042 0 0 1 12.333-.213V1.485A60.859 60.859 0 0 0 8.912.003zm1.707 1.81a.59.59 0 0 1 .015 0c3.06.088 6.125.404 9.167.95a.59.59 0 0 1 .476.686.59.59 0 0 1-.569.484.59.59 0 0 1-.116-.009 60.622 60.622 0 0 0-8.992-.931.59.59 0 0 1-.573-.607.59.59 0 0 1 .592-.572zm-4.212.028a.59.59 0 0 1 .578.565.59.59 0 0 1-.564.614 59.74 59.74 0 0 0-2.355.144.59.59 0 0 1-.04.002.59.59 0 0 1-.595-.542.59.59 0 0 1 .54-.635c.8-.065 1.6-.114 2.401-.148a.59.59 0 0 1 .035 0zm4.202 2.834a.59.59 0 0 1 .015 0 61.6 61.6 0 0 1 9.167.8.59.59 0 0 1 .488.677.59.59 0 0 1-.602.494.59.59 0 0 1-.076-.006 60.376 60.376 0 0 0-8.99-.786.59.59 0 0 1-.584-.596.59.59 0 0 1 .582-.583zm-4.211.097a.59.59 0 0 1 .587.555.59.59 0 0 1-.554.622c-.786.046-1.572.107-2.356.184a.59.59 0 0 1-.04.003.59.59 0 0 1-.603-.533.59.59 0 0 1 .53-.644c.8-.078 1.599-.14 2.4-.187a.59.59 0 0 1 .036 0zM10.6 7.535a.59.59 0 0 1 .015 0c3.06-.013 6.125.204 9.167.65a.59.59 0 0 1 .498.67.59.59 0 0 1-.593.504.59.59 0 0 1-.076-.006 60.142 60.142 0 0 0-8.992-.638.59.59 0 0 1-.592-.588.59.59 0 0 1 .573-.592zm1.153 2.846a61.093 61.093 0 0 1 8.02.515.59.59 0 0 1 .509.66.59.59 0 0 1-.586.514.59.59 0 0 1-.076-.005 59.982 59.982 0 0 0-8.99-.492.59.59 0 0 1-.603-.577.59.59 0 0 1 .578-.603c.382-.008.765-.012 1.148-.012zm1.139 2.832a60.92 60.92 0 0 1 6.871.394.59.59 0 0 1 .52.652.59.59 0 0 1-.577.523.59.59 0 0 1-.076-.004 59.936 59.936 0 0 0-8.991-.344.59.59 0 0 1-.61-.568.59.59 0 0 1 .567-.611c.765-.028 1.53-.042 2.296-.042z"></path>
            </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
}
