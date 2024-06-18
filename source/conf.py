# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('.'))

import src


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'STIM'
copyright = '2024, Humphreys Lab'
author = 'Qiao Xuanyuan'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # https://myst-nb.readthedocs.io
    "myst_nb",
    'sphinx.ext.autosummary', 
    "sphinx.ext.autodoc",
    'sphinx.ext.napoleon',
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx.ext.viewcode"
]
source_suffix = [".rst", ".md"]

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"


# myst-nb plotly
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]


# -- Options for warnings -------------------------------------------------

# Need for MyST-NB extension
suppress_warnings = ["mystnb.unknown_mime_type"]


# -- Options for MyST-NB extension -------------------------------------------------

# Turn off notebooks execution
nb_execution_mode = "off"

myst_heading_anchors = 4
html_show_sourcelink = False
html_show_sphinx = False
