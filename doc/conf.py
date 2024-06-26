# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'xpic'
copyright = '2024, Vladislav Kurshakov'
author = 'Vladislav Kurshakov'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = []

exclude_patterns = []

# Sphinx will warn about all references where the target cannot be found.
nitpicky = True

highlight_language = 'cpp'
primary_domain = 'cpp'

latex_elements = {
  # Font size in points, 12pt is the maximum available
  'pointsize': '10pt',

  # Removes empty pages; no distinction on odd-even pages
  'extraclassoptions': 'openany,oneside',

  # Uncomment to enable Cyrillic fonts support
  # 'fontenc': '\\usepackage[T2A,T1]{fontenc}',
}
