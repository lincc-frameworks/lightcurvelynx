"""Basic functions for downloading data using pooch."""

import logging
from pathlib import Path

import pooch

from lightcurvelynx.utils.io_utils import SquashLogging

logger = logging.getLogger(__name__)


def download_data_file_if_needed(
    data_path,
    data_url,
    force_download=False,
    silent=False,
):
    """Download a data file from a URL and save it to a specified path.

    Parameters
    ----------
    data_path : str or Path
        The path to the data file. This is where the downloaded file will be written.
    data_url : str
        The URL to download the data file.
    force_download : bool, optional
        If True, the file will be downloaded even if it already exists. Default is False.
    silent : bool, optional
        If True, suppress print statements from the download process. Default is False.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    # Start by checking if the file already exists and if we are not forcing a download.
    data_path = Path(data_path)
    if not force_download and data_path.exists():
        logger.info(f"Data file {data_path} already exists. Skipping download.")
        return True

    # Check that there is a valid URL for the download.
    if data_url is None or len(data_url) == 0:
        raise ValueError("No URL given for table download.")
    if not silent:
        print(f"Downloading data file from {data_url} to {data_path}")

    # Create the directory in which to save the file if it does not already exist.
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Use pooch to download the data files and extract them to the data directory.
    # We use SquashLogging to suppress the INFO output from pooch, which is not
    # often useful to the user.
    with SquashLogging(logger=pooch.utils.get_logger(), level=logging.WARNING):
        full_path = pooch.retrieve(
            url=data_url,
            known_hash=None,
            fname=data_path.name,
            path=data_path.parent,
        )

    if full_path is None or not Path(full_path).exists():
        logger.error(f"Data file not downloaded from {data_url}.")
        return False
    return True
