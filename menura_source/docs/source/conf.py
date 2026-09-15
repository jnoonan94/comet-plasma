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
import os
import sys ## Documenting by producing doxygen *xml*
import shutil

sys.path.append( "../ext/breathe/" ) ## Documenting by producing doxygen *xml*
sys.path.insert(0, os.path.abspath('./../..'))
sys.path.append('../../notebooks')

# -- Copying notebooks -------------------------------------------------------
nb_source = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               "..", "..", "notebooks"))
nb_dest = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "notebooks"))

if os.path.exists(nb_dest):
    shutil.rmtree(nb_dest)
os.mkdir(nb_dest)

for root, dirs, files in os.walk(nb_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(nb_source, nb_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".ipynb", ".md", ".rst"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(nb_source, nb_dest)
            shutil.copyfile(source_filename, dest_filename)



# -- Project information -----------------------------------------------------

project = 'menura'
copyright = '2021, etienne.behar'
author = 'etienne.behar'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#
html_logo = 'fig/menura_logo2_white.png'

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'fig/menura_logo2_thumb.png'




# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_extra_path = ['../build/html'] ## Documenting by producing *doxygen* html
breathe_projects = { "menura_breathe": "../build/xml" } ## Documenting by producing doxygen *xml*
breathe_default_project = "menura_breathe" ## Documenting by producing doxygen *xml*
