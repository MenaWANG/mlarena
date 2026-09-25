"""Sphinx configuration for MLArena."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "pyproject.toml").open("rb") as metadata_file:
    release = tomllib.load(metadata_file)["tool"]["poetry"]["version"]

project = "MLArena"
author = "Mena Wang"
copyright = "2026, Mena Wang"
version = release
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]
root_doc = "index"
myst_heading_anchors = 3
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_title = f"MLArena {release} documentation"
html_theme_options = {"navigation_depth": 3}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_numpy_docstring = True
pygments_style = "sphinx"
