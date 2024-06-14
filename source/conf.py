# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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
]
source_suffix = [".rst", ".md"]

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

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
