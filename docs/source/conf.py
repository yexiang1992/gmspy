# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
import sys
from pathlib import Path


this_dir = Path(__file__).resolve().parent.parent.parent
about = {}
with open(this_dir / "gmspy" / "__about__.py") as f:
    d = exec(f.read(), about)
__version__ = about["__version__"]

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = 'gmspy'
copyright = '2025, yexiang yan'
author = 'yexiang yan'

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    'nbsphinx',
    "jupyter_sphinx",
]

nbsphinx_allow_errors = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = "index"

# Output file base name for HTML help builder.
htmlhelp_basename = "gmspydoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "gmspy.tex",
        "gmspy Documentation",
        "Yexiang Yan",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "gmspy", "gmspy Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "gmspy",
        "gmspy Documentation",
        author,
        "gmspy",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# --- Sphinx-Gallery options -----------------------------------------------------------------------

# sphinx_gallery_conf = {
#     # convert rst to md for ipynb
#     "pypandoc": False,
#     # path to your examples scripts
#     "examples_dirs": ["../../examples/"],
#     # path where to save gallery generated examples
#     "gallery_dirs": ["sphinx_gallery_examples"],
#     # Patter to search for example files
#     "filename_pattern": r"\.py",
#     # Remove the 'Download all examples' button from the top level gallery
#     "download_all_examples": False,
#     # Sort gallery example by file name instead of number of lines (default)
#     "within_subsection_order": FileNameSortKey,
#     # directory where function granular galleries are stored
#     "backreferences_dir": None,
#     # Modules for which function level galleries are created.  In
#     "doc_module": "gmspy",
#     "image_scrapers": ("matplotlib", ),
#     "first_notebook_cell": ("%matplotlib inline\n"),
# }

# sphinx_gallery_conf = {
#      'examples_dirs': "../../examples/",   # path to your example scripts
#      'gallery_dirs': "sphinx_gallery_examples",  # path to where to save gallery generated output
# }
