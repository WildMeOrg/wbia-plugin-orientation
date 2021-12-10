# -*- coding: utf-8 -*-
from sphinx.ext.autodoc import between
import sphinx_rtd_theme  # NOQA
import sys
import os

# Dont parse IBEIS args
os.environ['IBIES_PARSE_ARGS'] = 'OFF'
os.environ['UTOOL_AUTOGEN_SPHINX_RUNNING'] = 'ON'

sys.path.append(os.path.abspath('../'))

autosummary_generate = True

modindex_common_prefix = ['_']
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

master_doc = 'index'

html_theme = 'sphinx_rtd_theme'
html_theme_path = [
    '_themes',
]

# -- Project information -----------------------------------------------------

project = 'wbia_orientation'
copyright = '2019, Wild Me'
author = 'Olga Moskvyak, Jason Parham'

# The short X.Y version
version = '0.1.0.dev0'

# The full version, including alpha/beta/rc tags
release = '0.1.0.dev0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.imgmath',
    'sphinx.ext.napoleon',
]


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect('autodoc-process-docstring', between('^.*IGNORE.*$', exclude=True))
    return app
