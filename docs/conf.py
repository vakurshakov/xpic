# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'xpic'

copyright = '2025, Vladislav Kurshakov'

author = 'Владислав Куршаков'

release = '1.0'

language = 'ru'

extensions = []

exclude_patterns = []

nitpicky = True

highlight_language = 'cpp'

primary_domain = 'cpp'

latex_theme = 'howto'

latex_elements = {
  'pointsize': '10pt',

  # Removes empty pages; no distinction on odd-even pages
  'extraclassoptions': 'openany',
}
