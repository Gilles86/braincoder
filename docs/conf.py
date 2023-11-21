# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Braincoder'
copyright = '2023, Gilles de Hollander'
author = 'Gilles de Hollander'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.duration', 
              'sphinx_gallery.gen_gallery']

autosummary_generate = True

# Generate the plots for the gallery
plot_gallery = "True"

sphinx_gallery_conf = {
    "doc_module": "braincoder",
    "reference_url": {"braincoder": None},
    "examples_dirs": "../examples/",
    "gallery_dirs": "auto_examples",
    # Ignore the function signature leftover by joblib
    # "ignore_pattern": r"func_code\.py",
    # "show_memory": not sys.platform.startswith("win"),
    "remove_config_comments": True,
    "nested_sections": True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'piccolo_theme'
html_static_path = ['_static']
