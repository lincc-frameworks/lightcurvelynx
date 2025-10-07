# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
from importlib.metadata import version

# Define path to the code to be documented **relative to where conf.py (this file) is kept**
sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lightcurvelynx"
copyright = "2024, LINCC Frameworks"
author = "LINCC Frameworks"
release = version("lightcurvelynx")
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.mathjax", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]

extensions.append("autoapi.extension")
extensions.append("nbsphinx")

# Allow notebook execution errors to not break the build
nbsphinx_allow_errors = True

# -- sphinx-copybutton configuration ----------------------------------------
extensions.append("sphinx_copybutton")
# Enable tab-set and tab-item directives used in .rst files
extensions.append("sphinx_design")
## sets up the expected prompt text from console blocks, and excludes it from
## the text that goes into the clipboard.
copybutton_exclude = ".linenos, .gp"
copybutton_prompt_text = ">> "

## lets us suppress the copy button on select code blocks.
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"

templates_path = []
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# This assumes that sphinx-build is called from the root directory
master_doc = "index"
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
# Remove namespaces from class/method signatures
add_module_names = False

autoapi_type = "python"
autoapi_dirs = ["../src"]
autoapi_ignore = ["*/__main__.py", "*/_version.py"]
autoapi_add_toc_tree_entry = False
autoapi_member_order = "bysource"
# Additional configuration to skip private members
autoapi_python_class_content = "class"
autoapi_generate_api_docs = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

# The key is to NOT include "private-members" in autoapi_options
# This should be sufficient to exclude private members by default


# Try direct configuration approach
def autoapi_skip_member_handler(app, what, name, obj, skip, options):
    """Direct handler for autoapi skip member."""
    print(f"DEBUG: Direct handler called for {what} {name}")
    if name.startswith("_") and not name.startswith("__"):
        print(f"DEBUG: Direct handler skipping: {name}")
        return True
    return skip


# Assign the handler directly
autoapi_skip_member = autoapi_skip_member_handler

html_theme = "sphinx_rtd_theme"

# Support use of arbitrary section titles in docstrings
napoleon_custom_sections = ["Citations"]


def skip_private_members(app, what, name, obj, skip, options):
    """Skip private members during autoapi generation."""
    # Debug print to see what's being processed
    if name.startswith("_"):
        print(f"DEBUG: Considering {what} {name}, skip={skip}")

    # Skip private members (single underscore) but keep special methods (double underscore)
    if name.startswith("_") and not name.startswith("__"):
        print(f"DEBUG: Skipping private member: {name}")
        return True
    return skip


def setup(app):
    """Set up the Sphinx app with custom configurations."""
    print("DEBUG: Setup function called, trying different event names")

    # Try various possible event names for different autoapi versions
    event_names = [
        "autoapi-skip-member",
        "autodoc-skip-member",
        "autoapi-before-content",
        "autoapi-after-content",
    ]

    for event_name in event_names:
        try:
            app.connect(event_name, skip_private_members)
            print(f"DEBUG: Connected to event: {event_name}")
        except Exception as e:
            print(f"DEBUG: Failed to connect to {event_name}: {e}")

    # Also try connecting via autoapi's own configuration
    try:
        # Some versions of autoapi use a different callback mechanism
        app.add_config_value("autoapi_skip_members", skip_private_members, "env")
        print("DEBUG: Added autoapi_skip_members config value")
    except Exception as e:
        print(f"DEBUG: Failed to add config value: {e}")

    print("DEBUG: Setup complete")
