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

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ['css/custom.css']

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.duration', 
              'sphinx_gallery.gen_gallery', 'sphinxcontrib.bibtex', 'sphinx.ext.mathjax']

bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_reference_style = "author_year"
bibtex_footreference_style = "author_year"
bibtex_footbibliography_header = ""

autosummary_generate = True

# Generate the plots for the gallery
plot_gallery = "True"

sphinx_gallery_conf = {
    "doc_module": "braincoder",
    "reference_url": {"braincoder": None},
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    # Ignore the function signature leftover by joblib
    "ignore_pattern": r"func_code\.py",
    "filename_pattern": '.*',
    # "show_memory": not sys.platform.startswith("win"),
    "remove_config_comments": True,
    "nested_sections": True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
font_awesome = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/"
html_css_files += [
    "custom.css",
    (
        "https://cdnjs.cloudflare.com/ajax/libs/"
        "font-awesome/5.15.4/css/all.min.css"
    ),
    f"{font_awesome}fontawesome.min.css",
    f"{font_awesome}solid.min.css",
    f"{font_awesome}brands.min.css",
]
html_theme = 'furo'
html_static_path = ['_static']
