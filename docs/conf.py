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
import subprocess
import sys
#sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Linking READTHEDOCS/SPHinx to Doxygen -----------------------------------
def configureDoxyfile(input_dir, output_dir):
    with open('Doxyfile.in', 'r') as file :
        filedata = file.read()

    filedata = filedata.replace('@DOXYGEN_INPUT_DIR@', input_dir)
    filedata = filedata.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)

    with open('Doxyfile', 'w') as file:
        file.write(filedata)

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

breathe_projects = {}

if read_the_docs_build:
    input_dir = '../include'
    output_dir = 'build'
    configureDoxyfile(input_dir, output_dir)
    subprocess.call('doxygen', shell=True)
    breathe_projects['linSolver'] = output_dir + '/xml'
else:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- Project information -----------------------------------------------------
master_doc = "index"
project = 'linSolver'
copyright = '2020, Siavosh Shadpey'
author = 'Siavosh Shadpey'

# The full version, including alpha/beta/rc tags
release = '0.0'

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "breathe",
    "sphinx.ext.inheritance_diagram"
]

# Breathe Configuration
breathe_default_project = "linSolver"

# TB = Top to Bottom view
inheritance_graph_attrs = dict(rankdir="TB", size='""')

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False

todo_include_todos = False
add_module_names = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_add_permalinks = ""
html_theme_path = ["."]
#html_theme = "semantic_sphinx"
html_theme_options = {
    "navbar_links": [
    	("Home", master_doc),
        ("Documentation", "documentation"),
        ("Contact", "contact")]
}
html_sidebars = {"**": ["about.html", "navigation.html", "searchbox.html"]}

source_suffix = ['.rst', '.md']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

autodoc_default_flags = ['members', 'show-inheritance']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"


def setup(app):
    app.add_css_file(
        "https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css"
    )
    app.add_css_file("default.css")