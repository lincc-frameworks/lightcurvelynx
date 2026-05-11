import os
from pathlib import Path

from ._version import __version__ as __version__  # noqa

# Define some global directory paths to use for testing, notebooks, etc.
_LIGHTCURVELYNX_BASE_DIR = Path(__file__).parent.parent.parent
_LIGHTCURVELYNX_BASE_DATA_DIR = _LIGHTCURVELYNX_BASE_DIR / "data"


def _get_download_data_dir() -> Path:
    """Return the directory for downloaded data, following the XDG Base Directory Specification.

    The directory is resolved in priority order:
    1. ``LIGHTCURVELYNX_DATA_DIR`` environment variable (package-specific override)
    2. ``$XDG_CACHE_HOME/lightcurvelynx`` (XDG standard)
    3. ``~/.cache/lightcurvelynx`` (default)
    """
    if data_dir := os.environ.get("LIGHTCURVELYNX_DATA_DIR"):
        return Path(data_dir)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "lightcurvelynx"


_LIGHTCURVELYNX_DOWNLOAD_DATA_DIR = _get_download_data_dir()

_LIGHTCURVELYNX_TEST_DIR = _LIGHTCURVELYNX_BASE_DIR / "tests" / "lightcurvelynx"
_LIGHTCURVELYNX_TEST_DATA_DIR = _LIGHTCURVELYNX_TEST_DIR / "data"
