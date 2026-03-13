from pathlib import Path

from ._version import __version__ as __version__  # noqa

# Define some global directory paths to use for testing, notebooks, etc.
_LIGHTCURVELYNX_BASE_DIR = Path(__file__).parent.parent.parent
_LIGHTCURVELYNX_BASE_DATA_DIR = _LIGHTCURVELYNX_BASE_DIR / "data"

_LIGHTCURVELYNX_TEST_DIR = _LIGHTCURVELYNX_BASE_DIR / "tests" / "lightcurvelynx"
_LIGHTCURVELYNX_TEST_DATA_DIR = _LIGHTCURVELYNX_TEST_DIR / "data"
