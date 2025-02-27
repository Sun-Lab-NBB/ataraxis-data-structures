# Configuration file for the Sphinx documentation builder.
import importlib_metadata

# -- Project information -----------------------------------------------------
project = 'ataraxis-data-structures'
# noinspection PyShadowingBuiltins
copyright = '2024, Ivan Kondratyev (Inkaros) & Sun Lab'
authors = ['Ivan Kondratyev (Inkaros)', 'Edwin Chen']
release = importlib_metadata.version("ataraxis-data-structures")  # Extracts the project version from the .toml file.

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # To build documentation from python source code docstrings.
    'sphinx.ext.napoleon',       # To read google-style docstrings (works with autodoc module).
    'sphinx_autodoc_typehints',  # To parse typehints into documentation
    'sphinx_rtd_theme',          # To format the documentation html using ReadTheDocs format.
    'sphinx_rtd_dark_mode',       # Enables dark mode for RTD theme.
]

templates_path = ['_templates']
exclude_patterns = []

# Google-style docstring parsing configuration for napoleon extension
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Additional sphinx-typehints configuration
sphinx_autodoc_typehints = True
always_document_param_types = False
typehints_document_rtype = True
typehints_use_rtype = True
typehints_defaults = 'comma'
simplify_optional_unions = True
typehints_formatter = None
typehints_use_signature = False
typehints_use_signature_return = False

# RTD dark mode starts in light mode
default_dark_mode = False

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Directs sphinx to use RTD theme
