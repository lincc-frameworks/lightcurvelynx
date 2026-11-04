"""SN Ia models with ejecta-companion interaction from Kasen (2010).

The spectral time series were computed by Dan Kasen for single-degenerate SN Ia
progenitor systems in which the supernova ejecta collide with a non-degenerate
companion star. The collision produces excess emission in the first days after
explosion whose strength depends on the companion (and thus the binary separation)
and on the viewing angle. The spectra include both the interaction and the
underlying supernova emission.

The data are distributed via the `prestocolor <https://github.com/fedhere/prestocolor>`_
repository. On first use the repository's archive is downloaded once, the spectra are
converted into a single ``.npz`` file per companion and stored in the LightCurveLynx
data cache directory, and the archive is removed.

This model is based on the `original notebook
<https://colab.research.google.com/drive/1uSY8vTTeYLaTkDIKMSkLtSHDAQsoam5p>`_.
"""

import logging
import os
import re
import tempfile
import zipfile
from pathlib import Path

import astropy.units as u
import numpy as np
from citation_compass import CiteClass, cite_function
from scipy.interpolate import interp1d

from lightcurvelynx import _LIGHTCURVELYNX_DOWNLOAD_DATA_DIR
from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.consts import PARSEC_TO_CM
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.utils.data_download import download_data_file_if_needed

logger = logging.getLogger(__name__)

KASEN_COMPANIONS = {
    "a5e11": "2 M_sun main sequence star (separation a = 5e11 cm)",
    "a2e12": "6 M_sun main sequence star (separation a = 2e12 cm)",
    "a2e13": "1 M_sun red giant (separation a = 2e13 cm)",
}
"""The available companion models, keyed by the binary separation used in the file names."""

_PRESTOCOLOR_COMMIT = "d368fe67924812723b9d94999040ec0d59813380"
_PRESTOCOLOR_ZIP_URL = f"https://github.com/fedhere/prestocolor/archive/{_PRESTOCOLOR_COMMIT}.zip"
"""The URL of the prestocolor repository archive, pinned to a specific commit."""

_KASEN_DATA_DIR = _LIGHTCURVELYNX_DOWNLOAD_DATA_DIR / "kasen2010_companion"
"""The default directory for the downloaded and converted Kasen (2010) spectra."""

_KASEN_ZIP_NAME = "prestocolor.zip"

_KASEN_FILE_LUM_UNIT = 1e36
"""The unit of the luminosity column in the per-angle spectrum files (erg/s/Angstrom)."""

# Matches the per-angle spectrum files, for example:
# companion_a5e11/optical_t002.35_I3.spec.91.4326
_KASEN_SPEC_FILE_RE = re.compile(
    r"/companion_(?P<companion>a5e11|a2e12|a2e13)/"
    r"optical_t(?P<time>\d{3}\.\d{2})_I3\.spec\.(?P<angle>\d+\.\d{4})$"
)


def _kasen_grid_path(data_dir, companion):
    """The path of the converted spectra file for a given companion."""
    return Path(data_dir) / f"{companion}.npz"


def convert_kasen_archive(zip_path, data_dir):
    """Convert the per-angle Kasen (2010) spectrum files of the prestocolor archive
    into a ``<companion>.npz`` file per companion.

    Each file contains the arrays:

    * ``times`` - A length T array of the times since explosion (in days).
    * ``wavelengths`` - A length W array of the wavelengths (in Angstroms).
    * ``angles`` - A length A array of the viewing angles (in degrees), sorted
      in increasing order.
    * ``luminosity`` - An (A, T, W) float32 array of the specific luminosity
      L_lambda in units of ``_KASEN_FILE_LUM_UNIT`` (1e36 erg/s/Angstrom), as in
      the original files.

    Parameters
    ----------
    zip_path : str or Path
        The path to the zip archive of the prestocolor repository.
    data_dir : str or Path
        The directory in which to create the companion files.

    Returns
    -------
    paths : dict
        A dictionary mapping each companion name to the path of its file.
    """
    # Collect the luminosity column of every file keyed by companion and (angle, time).
    spectra = {companion: {} for companion in KASEN_COMPANIONS}
    wavelengths = None
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            match = _KASEN_SPEC_FILE_RE.search(name)
            if match is None:
                continue
            with zf.open(name) as f:
                data = np.loadtxt(f)
            if wavelengths is None:
                wavelengths = data[:, 0]
            elif not np.array_equal(wavelengths, data[:, 0]):
                raise ValueError(f"Inconsistent wavelength grid in {name}.")
            key = (float(match["angle"]), float(match["time"]))
            spectra[match["companion"]][key] = data[:, 1].astype(np.float32)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for companion, comp_spectra in spectra.items():
        if len(comp_spectra) == 0:
            raise ValueError(f"No spectra found for companion {companion} in {zip_path}.")

        # Sorted (angle, time) keys of a complete grid are exactly the order of the
        # (angle, time) axes, so the stacked spectra reshape straight into the grid.
        keys = sorted(comp_spectra)
        angles = np.unique([angle for angle, _ in keys])
        times = np.unique([time for _, time in keys])
        if len(keys) != len(angles) * len(times):
            raise ValueError(
                f"Incomplete grid for companion {companion}: {len(keys)} spectra "
                f"for {len(angles)} angles and {len(times)} times."
            )
        luminosity = np.stack([comp_spectra[key] for key in keys])
        luminosity = luminosity.reshape(len(angles), len(times), len(wavelengths))

        # Write into a temporary file first and move it into place, so an interrupted
        # conversion never leaves a partial file that would be mistaken for a complete one.
        # The temporary file is created next to the final one so the move stays atomic.
        grid_path = _kasen_grid_path(data_dir, companion)
        with tempfile.TemporaryDirectory(dir=data_dir, prefix=f".{companion}-") as tmp_dir:
            tmp_path = Path(tmp_dir) / grid_path.name
            np.savez(tmp_path, times=times, wavelengths=wavelengths, angles=angles, luminosity=luminosity)
            os.replace(tmp_path, grid_path)
        paths[companion] = grid_path
    return paths


@cite_function
def load_kasen_companion_grid(companion, *, data_dir=None, force_download=False):
    """Load the Kasen (2010) spectral grid for a given companion, downloading and
    converting the data if needed.

    References
    ----------
    * Data: Bianco F. B. et al., 2019, PASP, 131, 068002 - https://doi.org/10.1088/1538-3873/ab121a
    * Repository: https://github.com/fedhere/prestocolor

    Parameters
    ----------
    companion : str
        The companion model. One of the keys of ``KASEN_COMPANIONS``:
        "a5e11", "a2e12", or "a2e13".
    data_dir : str or Path, optional
        The directory in which to store the data. Defaults to the
        "kasen2010_companion" subdirectory of the LightCurveLynx data cache directory.
    force_download : bool, optional
        If True, the data are downloaded and converted even if they already exist.
        Default: False

    Returns
    -------
    grid : dict
        A dictionary of arrays: "times" (days since explosion),
        "wavelengths" (Angstroms), "angles" (degrees), and "luminosity" (A x T x W float32
        array of L_lambda in units of 1e36 erg/s/Angstrom; cast to float64 before scaling
        to erg/s/Angstrom to avoid float32 overflow).
    """
    if companion not in KASEN_COMPANIONS:
        raise ValueError(f"Unknown companion '{companion}'. Must be one of {list(KASEN_COMPANIONS)}.")
    data_dir = Path(_KASEN_DATA_DIR if data_dir is None else data_dir)
    grid_path = _kasen_grid_path(data_dir, companion)

    if force_download or not grid_path.exists():
        zip_path = data_dir / _KASEN_ZIP_NAME
        if not download_data_file_if_needed(zip_path, _PRESTOCOLOR_ZIP_URL, force_download=force_download):
            raise RuntimeError(f"Failed to download the Kasen (2010) spectra from {_PRESTOCOLOR_ZIP_URL}.")
        logger.info(f"Converting the Kasen (2010) spectra from {zip_path} into {data_dir}.")
        convert_kasen_archive(zip_path, data_dir)
        # We keep only the converted files that we need.
        zip_path.unlink()

    with np.load(grid_path) as data:
        return dict(data)


def _linear_interp(grid, values, x, axis):
    """Linearly interpolate the values along one axis of the grid, with zeros outside of it."""
    interp = interp1d(
        grid, values, axis=axis, bounds_error=False, fill_value=0.0, copy=False, assume_sorted=True
    )
    return interp(x)


class KasenIaCompanionModel(SEDModel, CiteClass):
    """A model of the early (first ~10 days) SN Ia emission, including the interaction
    of the ejecta with a non-degenerate companion star, from the Kasen (2010) simulations.

    The companion is assumed to fill its Roche lobe, so a single number (the binary
    separation) describes both its mass and distance. Three companions are available
    (see ``KASEN_COMPANIONS``):

    * "a5e11" - a 2 M_sun main sequence star at a separation of 5e11 cm.
    * "a2e12" - a 6 M_sun main sequence star at a separation of 2e12 cm.
    * "a2e13" - a 1 M_sun red giant at a separation of 2e13 cm.

    The spectra are given on a grid of 100 epochs (0.05 to 9.95 days since explosion),
    149 wavelengths (51 to 14851 Angstroms), and 40 viewing angles (12.84 to 167.16
    degrees, equally spaced in cosine). A viewing angle of 0 degrees places the companion
    between the observer and the exploding white dwarf, which gives the strongest
    interaction signal. The model linearly interpolates the specific luminosity in
    time, wavelength, and viewing angle (in degrees), and it is zero outside of the grid
    in any of the three dimensions.

    The model's t0 is the time of explosion.

    Parameterized values include:

    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The time of explosion, MJD. [from BasePhysicalModel]
    * viewing_angle - The viewing angle in degrees. The grid covers 12.84 to 167.16 degrees,
      and the flux is zero outside of this range.

    References
    ----------
    * Model: Kasen D., 2010, ApJ, 708, 1025 - https://doi.org/10.1088/0004-637X/708/2/1025

    Attributes
    ----------
    companion : str
        The companion model.
    times : numpy.ndarray
        A length T array of the grid times since explosion (in days).
    wavelengths : numpy.ndarray
        A length W array of the grid wavelengths (in Angstroms).
    angles : numpy.ndarray
        A length A array of the grid viewing angles (in degrees).

    Parameters
    ----------
    companion : str
        The companion model. One of "a5e11", "a2e12", or "a2e13".
    viewing_angle : parameter, optional
        The viewing angle (in degrees). If not provided, the angle is sampled isotropically
        (uniformly in cosine) within the range covered by the grid.
    data_dir : str or Path, optional
        The directory in which to store the data. Defaults to the
        "kasen2010_companion" subdirectory of the LightCurveLynx data cache directory.
    force_download : bool, optional
        If True, the data are downloaded and converted even if they already exist.
        Default: False
    **kwargs : dict, optional
        Any additional keyword arguments, such as t0, redshift, distance, ra, and dec.
    """

    def __init__(self, companion, viewing_angle=None, *, data_dir=None, force_download=False, **kwargs):
        if kwargs.get("t0") is None:
            raise ValueError("KasenIaCompanionModel requires a t0 parameter (the time of explosion).")
        super().__init__(**kwargs)

        self.companion = companion
        grid = load_kasen_companion_grid(companion, data_dir=data_dir, force_download=force_download)
        self.times = grid["times"]
        self.wavelengths = grid["wavelengths"]
        self.angles = grid["angles"]
        # An (A, T, W) array in erg/s/Angstrom, converted to float64 first to avoid overflow.
        self._luminosity = grid["luminosity"].astype(np.float64) * _KASEN_FILE_LUM_UNIT

        if viewing_angle is None:
            cos_angle = NumpyRandomFunc(
                "uniform",
                low=np.cos(np.radians(self.angles[-1])),
                high=np.cos(np.radians(self.angles[0])),
            )
            viewing_angle = BasicMathNode("degrees(acos(cos_angle))", cos_angle=cos_angle, backend="numpy")
        self.add_parameter("viewing_angle", viewing_angle, description="The viewing angle in degrees.")

    def minwave(self, **kwargs):
        """Get the minimum supported wavelength of the model (in Angstroms).

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minwave : float
            The minimum wavelength of the model (in Angstroms).
        """
        return float(self.wavelengths[0])

    def maxwave(self, **kwargs):
        """Get the maximum supported wavelength of the model (in Angstroms).

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maxwave : float
            The maximum wavelength of the model (in Angstroms).
        """
        return float(self.wavelengths[-1])

    def minphase(self, **kwargs):
        """Get the minimum supported phase (time since explosion) of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minphase : float
            The minimum phase of the model (in days).
        """
        return float(self.times[0])

    def maxphase(self, **kwargs):
        """Get the maximum supported phase (time since explosion) of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maxphase : float
            The maximum phase of the model (in days).
        """
        return float(self.times[-1])

    def compute_luminosity(self, phases, wavelengths, viewing_angle):
        """Compute the specific luminosity by linear interpolation of the grid.

        The interpolation is separable: first in viewing angle, then in time, then in
        wavelength. This is mathematically identical to trilinear interpolation over
        (time, wavelength, angle) and to scipy's ``RegularGridInterpolator`` with
        ``method="linear"``, ``bounds_error=False``, and ``fill_value=0.0``.

        Parameters
        ----------
        phases : numpy.ndarray
            A length T array of rest frame times since explosion (in days).
        wavelengths : numpy.ndarray
            A length N array of rest frame wavelengths (in Angstroms).
        viewing_angle : float
            The viewing angle (in degrees).

        Returns
        -------
        luminosity : numpy.ndarray
            A T x N matrix of the specific luminosity L_lambda (in erg/s/Angstrom).
        """
        lum = _linear_interp(self.angles, self._luminosity, viewing_angle, axis=0)
        lum = _linear_interp(self.times, lum, phases, axis=0)
        return _linear_interp(self.wavelengths, lum, wavelengths, axis=1)

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free rest frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps (in MJD).
        wavelengths : numpy.ndarray
            A length N array of rest frame wavelengths (in Angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A T x N matrix of rest frame SED values (in nJy).
        """
        params = self.get_local_params(graph_state)
        if params["distance"] is None:
            raise ValueError(
                "KasenIaCompanionModel requires a distance. Provide the distance parameter "
                "or both the redshift and the cosmology."
            )

        lum = self.compute_luminosity(np.asarray(times) - params["t0"], wavelengths, params["viewing_angle"])
        distance_cm = params["distance"] * PARSEC_TO_CM
        flux_flam = lum / (4.0 * np.pi * distance_cm**2)
        return flam_to_fnu(
            flux_flam,
            wavelengths,
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
