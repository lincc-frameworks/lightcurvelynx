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
    member_name = name.split(".")[-1] if "." in name else name
    print(f"DEBUG: Direct handler called for {what} {name}, member_name={member_name}")

    if member_name.startswith("_") and not member_name.startswith("__"):
        print(f"DEBUG: Direct handler FORCING SKIP: {member_name}")
        return True  # Force skip private members

    return skip


# Assign the handler directly - this is how some versions of autoapi expect it
autoapi_skip_member = autoapi_skip_member_handler


# Alternative direct assignment approach
def autoapi_skip_member_direct(app, what, name, obj, skip, options):
    """Module-level skip function that autoapi might find automatically."""
    member_name = name.split(".")[-1] if "." in name else name
    print(f"DEBUG: Module-level handler called for {what} {name}, member_name={member_name}")

    if member_name.startswith("_") and not member_name.startswith("__"):
        print(f"DEBUG: Module-level handler FORCING SKIP: {member_name}")
        return True  # Force skip private members

    return skip


html_theme = "sphinx_rtd_theme"

# Support use of arbitrary section titles in docstrings
napoleon_custom_sections = ["Citations"]


def skip_private_members(app, what, name, obj, skip, options):
    """Skip private members during autoapi generation."""
    # Get just the member name (without module path)
    member_name = name.split(".")[-1] if "." in name else name

    print(
        f"DEBUG: skip_private_members called with: what={what}, "
        "name={name}, member_name={member_name}, skip={skip}"
    )

    # Skip private members (single underscore) but keep special methods (double underscore)
    if member_name.startswith("_") and not member_name.startswith("__"):
        print(f"DEBUG: FORCING SKIP for private member: {member_name}")
        return True  # Force skip private members

    # For non-private members, use the default behavior
    return skip


def skip_member_new_signature(app, what, name, obj, skip, options):
    """Alternative signature for autoapi skip member."""
    member_name = name.split(".")[-1] if "." in name else name
    print(
        f"DEBUG: New signature called with: what={what}, name={name}, member_name={member_name}, skip={skip}"
    )

    if member_name.startswith("_") and not member_name.startswith("__"):
        print(f"DEBUG: New signature FORCING SKIP: {member_name}")
        return True  # Force skip private members

    return skip


def setup(app):
    """Set up the Sphinx app with custom configurations."""
    print("DEBUG: Setup function called, trying different approaches")

    # Connect to autoapi-skip-member with different function signatures
    try:
        app.connect("autoapi-skip-member", skip_private_members)
        print("DEBUG: Connected skip_private_members to autoapi-skip-member")
    except Exception as e:
        print(f"DEBUG: Failed to connect skip_private_members: {e}")

    # Try with different signature
    try:
        app.connect("autoapi-skip-member", skip_member_new_signature)
        print("DEBUG: Connected skip_member_new_signature to autoapi-skip-member")
    except Exception as e:
        print(f"DEBUG: Failed to connect skip_member_new_signature: {e}")

    # Try autodoc event too
    try:
        app.connect("autodoc-skip-member", skip_private_members)
        print("DEBUG: Connected to autodoc-skip-member")
    except Exception as e:
        print(f"DEBUG: Failed to connect to autodoc-skip-member: {e}")

    # Try the most direct approach: configure autoapi directly
    try:
        # Set the autoapi skip function directly on the app
        if hasattr(app, "config"):
            app.config.autoapi_skip_member = skip_private_members
            print("DEBUG: Set autoapi_skip_member on app.config")
    except Exception as e:
        print(f"DEBUG: Failed to set autoapi_skip_member on config: {e}")

    print("DEBUG: Setup complete")
