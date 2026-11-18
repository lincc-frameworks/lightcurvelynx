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

The spectra cover only the first 10 days after explosion, so the model blends them into
the Arnett (1982) model of the supernova, computed with redback, which continues the light
curve to later times.

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
from scipy.special import ndtr

from lightcurvelynx import _LIGHTCURVELYNX_DOWNLOAD_DATA_DIR
from lightcurvelynx.astro_utils.black_body import black_body_luminosity_density_per_solid
from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.consts import ANGSTROM_TO_CM, CGS_FNU_UNIT_TO_NJY, M_SUN_G, PARSEC_TO_CM
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.models.redback_models import try_import_redback
from lightcurvelynx.utils.data_download import download_data_file_if_needed
from lightcurvelynx.utils.extrapolate import ZeroPadding

logger = logging.getLogger(__name__)

KASEN_COMPANIONS = {
    "a5e11": "2 M_sun main sequence star (separation a = 5e11 cm)",
    "a2e12": "6 M_sun main sequence star (separation a = 2e12 cm)",
    "a2e13": "1 M_sun red giant (separation a = 2e13 cm)",
}
"""The available companion models, keyed by the binary separation used in the file names."""

_PRESTOCOLOR_COMMIT = "d368fe67924812723b9d94999040ec0d59813380"
_PRESTOCOLOR_ZIP_URL = f"https://github.com/fedhere/prestocolor/archive/{_PRESTOCOLOR_COMMIT}.zip"
"""The URL of the prestocolor repository archive."""

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


def _convert_kasen_archive(zip_path, data_dir):
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
def _load_kasen_companion_grid(companion, *, data_dir=None, force_download=False):
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
        _convert_kasen_archive(zip_path, data_dir)
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


def _bin_edges(centers):
    """The edges of the bins around the given centers: midpoints, extended by half a bin at the ends."""
    mid = 0.5 * (centers[1:] + centers[:-1])
    return np.concatenate([[2.0 * centers[0] - mid[0]], mid, [2.0 * centers[-1] - mid[-1]]])


def _smoothing_matrix(centers, edges, sigma):
    """The matrix of a Gaussian smoothing of binned energies, which conserves the energy."""

    # [i, j] is the part of the Gaussian centered on bin j that falls between lo[i] and hi[i].
    def gaussian_fraction(lo, hi):
        return ndtr((hi[:, None] - centers) / sigma) - ndtr((lo[:, None] - centers) / sigma)

    lo, hi = edges[:-1], edges[1:]
    matrix = gaussian_fraction(lo, hi)
    # Reflect the parts outside of the grid back into it.
    for edge in (edges[0], edges[-1]):
        # Infinite edges (lg t = -inf at t = 0) have nothing beyond them.
        if np.isfinite(edge):
            matrix += gaussian_fraction(2.0 * edge - hi, 2.0 * edge - lo)
    # Normalize to account for multiple reflections, if the kernel is as wide as the grid.
    return matrix / matrix.sum(axis=0)


def _smooth_luminosity(luminosity, times, wavelengths, wave_smoothing_aa, lg_time_smoothing):
    """Smooth an (A, T, W) luminosity grid in wavelength and in lg(time), conserving energy.

    Parameters
    ----------
    luminosity : numpy.ndarray
        An A x T x W array of the specific luminosity.
    times : numpy.ndarray
        A length T array of the grid times since explosion (in days).
    wavelengths : numpy.ndarray
        A length W array of the grid wavelengths (in Angstroms).
    wave_smoothing_aa : float or None
        The standard deviation of the Gaussian in wavelength (in Angstroms), or None
        for no smoothing in wavelength.
    lg_time_smoothing : float or None
        The standard deviation of the Gaussian in the decimal logarithm of the time since
        explosion, lg(t / 1 day), in dex, or None for no smoothing in time.

    Returns
    -------
    luminosity : numpy.ndarray
        The smoothed A x T x W array.
    """
    if wave_smoothing_aa is not None:
        edges = _bin_edges(wavelengths)
        widths = np.diff(edges)
        matrix = _smoothing_matrix(wavelengths, edges, wave_smoothing_aa)
        luminosity = (luminosity * widths) @ matrix.T / widths
    if lg_time_smoothing is not None:
        # Nothing is emitted before the explosion, so the first bin starts at t = 0 (lg t = -inf).
        edges = _bin_edges(times)
        edges[0] = 0.0
        widths = np.diff(edges)[:, None]
        log_edges = np.concatenate([[-np.inf], np.log10(edges[1:])])
        matrix = _smoothing_matrix(np.log10(times), log_edges, lg_time_smoothing)
        luminosity = matrix @ (luminosity * widths) / widths
    return luminosity


def _smoothstep(x):
    """A smooth step from 0 at x <= 0 to 1 at x >= 1, with zero slope at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class KasenIaCompanionModel(SEDModel, CiteClass):
    """A model of SN Ia emission with the interaction of the ejecta with a non-degenerate
    companion star in the first ~10 days, from the Kasen (2010) simulations.

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
    time, wavelength, and the cosine of the viewing angle, and it is zero outside of the
    grid in any of the three dimensions.

    By default the grid is smoothed with a Gaussian in the decimal logarithm of time
    (sigma = 0.075 dex), and optionally with a Gaussian in wavelength. The lg(time) kernel is
    narrower than the grid step in the first hours, so it keeps the sharp early interaction
    flash while smoothing the later, slowly evolving epochs. The smoothing redistributes
    energy between the grid bins, reflecting it at the grid edges, so it conserves the total
    radiated energy.

    The model continues past the end of the grid as the Arnett (1982) model of the
    supernova, computed with redback. The model is (1 - alpha(t)) K(t) + alpha(t) A(t), where
    K is the Kasen model, A is the Arnett model, t is the time since explosion, and
    alpha(t) = 3 x^2 - 2 x^3 with x = min(max(t / t_end, 0), 1), where t_end = 9.95 days is
    the last epoch of the grid. The default Arnett parameters follow the Kasen (2010) ejecta:
    0.6 M_sun of 56Ni in a Chandrasekhar mass ejecta with a kinetic energy of 1e51 erg. The
    default optical opacity is that of the Doppler-broadened iron group lines that dominate
    SNe Ia (Kasen & Woosley 2007), and the default gamma-ray opacity is 0.06 Y_e cm^2/g with
    Y_e = 0.5 (Swartz et al. 1995). The spectrum is a blackbody at the photosphere, which
    expands with the ejecta, R = v t, where E = 3/10 M v^2, until its temperature reaches the
    floor.

    The model's t0 is the time of explosion.

    Parameterized values include:

    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The time of explosion, MJD. [from BasePhysicalModel]
    * viewing_angle - The viewing angle in degrees. The grid covers 12.84 to 167.16 degrees,
      and the Kasen (2010) part of the flux is zero outside of this range.
    * ejecta_mass - The ejecta mass of the Arnett model in M_sun.
    * nickel_mass - The 56Ni mass of the Arnett model in M_sun.
    * kinetic_energy - The kinetic energy of the ejecta of the Arnett model in erg.
    * kappa - The optical opacity of the Arnett model in cm^2/g.
    * kappa_gamma - The gamma-ray opacity of the Arnett model in cm^2/g.
    * temperature_floor - The photospheric temperature floor of the Arnett model in K.

    References
    ----------
    * Model: Kasen D., 2010, ApJ, 708, 1025 - https://doi.org/10.1088/0004-637X/708/2/1025
    * Arnett model: Arnett W. D., 1982, ApJ, 253, 785 - https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A
    * 56Ni and 56Co heating: Nadyozhin D. K., 1994, ApJS, 92, 527 -
      https://ui.adsabs.harvard.edu/abs/1994ApJS...92..527N
    * Optical opacity: Kasen D. & Woosley S. E., 2007, ApJ, 656, 661 - https://arxiv.org/abs/astro-ph/0609540
    * Gamma-ray opacity: Swartz D. A., Sutherland P. G. & Harkness R. P., 1995, ApJ, 446, 766 -
      https://arxiv.org/abs/astro-ph/9501005
    * redback: Sarin N. et al., 2024, MNRAS, 531, 1203 - https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.1203S

    Attributes
    ----------
    companion : str
        The companion model.

    Parameters
    ----------
    companion : str
        The companion model. One of "a5e11", "a2e12", or "a2e13".
    viewing_angle : parameter, optional
        The viewing angle (in degrees). If not provided, the angle is sampled isotropically
        (uniformly in cosine) within the range covered by the grid.
    wave_smoothing_aa : float or None, optional
        The standard deviation of the Gaussian smoothing in wavelength (in Angstroms),
        or None to disable it. The smoothing barely changes the passband fluxes, so it is
        needed only for smoother spectra, for which we recommend a value of about 200 Angstroms.
        Default: None
    lg_time_smoothing : float or None, optional
        The standard deviation of the Gaussian smoothing in the decimal logarithm of the
        time since explosion, lg(t / 1 day), in dex, or None to disable it. Default: 0.075
    ejecta_mass : parameter, optional
        The ejecta mass of the Arnett model (in M_sun). Default: 1.4, the Chandrasekhar mass
        that the Kasen (2010) ejecta are scaled to: the velocity scale of the ejecta,
        6e8 cm/s for E = 1e51 erg and M = M_ch, is (1e51 erg / 1.4 M_sun)^(1/2).
    nickel_mass : parameter, optional
        The 56Ni mass of the Arnett model (in M_sun). Default: 0.6, as in Kasen (2010).
    kinetic_energy : parameter, optional
        The kinetic energy of the ejecta of the Arnett model (in erg). Default: 1e51, the
        energy that the Kasen (2010) ejecta are scaled to.
    kappa : parameter, optional
        The optical opacity of the Arnett model (in cm^2/g). Default: 0.1, the opacity of the
        Doppler-broadened iron group lines that dominate SNe Ia (Kasen & Woosley 2007).
    kappa_gamma : parameter, optional
        The gamma-ray opacity of the Arnett model (in cm^2/g). Default: 0.03, which is
        0.06 Y_e cm^2/g with Y_e = 0.5 (Swartz et al. 1995).
    temperature_floor : parameter, optional
        The photospheric temperature floor of the Arnett model (in K). Default: 500.0
    time_extrapolation : FluxExtrapolationModel or tuple, optional
        The extrapolation model(s) to use for times outside of the model's phase range.
        Default: zero padding, so the flux is zero before the first epoch of the grid and
        after the maximum time of the redback Arnett model.
    data_dir : str or Path, optional
        The directory in which to store the data. Defaults to the
        "kasen2010_companion" subdirectory of the LightCurveLynx data cache directory.
    force_download : bool, optional
        If True, the data are downloaded and converted even if they already exist.
        Default: False
    **kwargs : dict, optional
        Any additional keyword arguments, such as t0, redshift, distance, ra, and dec.
    """

    def __init__(
        self,
        companion,
        viewing_angle=None,
        *,
        wave_smoothing_aa=None,
        lg_time_smoothing=0.075,
        ejecta_mass=1.4,
        nickel_mass=0.6,
        kinetic_energy=1e51,
        kappa=0.1,
        kappa_gamma=0.03,
        temperature_floor=500.0,
        time_extrapolation=None,
        data_dir=None,
        force_download=False,
        **kwargs,
    ):
        if kwargs.get("t0") is None:
            raise ValueError("KasenIaCompanionModel requires a t0 parameter (the time of explosion).")
        if time_extrapolation is None:
            time_extrapolation = ZeroPadding()
        super().__init__(time_extrapolation=time_extrapolation, **kwargs)

        redback = try_import_redback()
        # The maximum time of the redback Arnett model (in days).
        self._arnett_max_time = redback.model_library.get_model_metadata("arnett").max_time_days

        self.add_parameter("ejecta_mass", ejecta_mass, description="The ejecta mass in M_sun.")
        self.add_parameter("nickel_mass", nickel_mass, description="The 56Ni mass in M_sun.")
        self.add_parameter("kinetic_energy", kinetic_energy, description="The kinetic energy in erg.")
        self.add_parameter("kappa", kappa, description="The optical opacity in cm^2/g.")
        self.add_parameter("kappa_gamma", kappa_gamma, description="The gamma-ray opacity in cm^2/g.")
        self.add_parameter(
            "temperature_floor", temperature_floor, description="The photospheric temperature floor in K."
        )
        self.companion = companion
        grid = _load_kasen_companion_grid(companion, data_dir=data_dir, force_download=force_download)
        self._times = grid["times"]
        self._wavelengths = grid["wavelengths"]
        # The angles are sorted in increasing order, so their cosines are reversed to be increasing.
        self._cos_angles = np.cos(np.radians(grid["angles"][::-1]))
        # An (A, T, W) array in erg/s/Angstrom, converted to float64 first to avoid overflow.
        self._luminosity = _smooth_luminosity(
            grid["luminosity"][::-1].astype(np.float64) * _KASEN_FILE_LUM_UNIT,
            self._times,
            self._wavelengths,
            wave_smoothing_aa,
            lg_time_smoothing,
        )

        if viewing_angle is None:
            cos_angle = NumpyRandomFunc(
                "uniform",
                low=self._cos_angles[0],
                high=self._cos_angles[-1],
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
        return float(self._wavelengths[0])

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
        return float(self._wavelengths[-1])

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
        return float(self._times[0])

    def maxphase(self, **kwargs):
        """Get the maximum supported phase (time since explosion) of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maxphase : float
            The maximum phase of the model (in days), the maximum time of the redback
            Arnett model.
        """
        return self._arnett_max_time

    def _compute_kasen_luminosity(self, phases, wavelengths, viewing_angle):
        """Compute the specific luminosity of the Kasen (2010) grid by linear interpolation.

        The interpolation is separable: first in the cosine of the viewing angle, then in
        time, then in wavelength. This is mathematically identical to trilinear interpolation
        over (time, wavelength, cos(angle)) and to scipy's ``RegularGridInterpolator`` with
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
        cos_angle = np.cos(np.radians(viewing_angle))
        lum = _linear_interp(self._cos_angles, self._luminosity, cos_angle, axis=0)
        lum = _linear_interp(self._times, lum, phases, axis=0)
        return _linear_interp(self._wavelengths, lum, wavelengths, axis=1)

    def _compute_arnett_sed(self, phases, wavelengths, graph_state):
        """Compute the rest frame flux density of the redback Arnett (1982) model.

        The bolometric luminosity is powered by the 56Ni and 56Co decay and diffuses out
        of the ejecta. The spectrum is a blackbody at the photosphere, which expands with
        the ejecta, R = v t, where E = 3/10 M v^2, until its temperature reaches the floor.

        Parameters
        ----------
        phases : numpy.ndarray
            A length T array of sorted rest frame times since explosion (in days).
        wavelengths : numpy.ndarray
            A length N array of rest frame wavelengths (in Angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A T x N matrix of rest frame SED values (in nJy), zero before the explosion.
        """
        redback = try_import_redback()

        params = self.get_local_params(graph_state)
        phases = np.asarray(phases, dtype=float)
        flux_density = np.zeros((len(phases), len(wavelengths)))
        after = phases > 0.0
        if not np.any(after):
            return flux_density

        # The ejecta velocity in km/s.
        velocity = np.sqrt(10.0 * params["kinetic_energy"] / (3.0 * params["ejecta_mass"] * M_SUN_G)) / 1e5
        lbol = redback.transient_models.supernova_models.arnett_bolometric(
            time=phases[after],
            f_nickel=params["nickel_mass"] / params["ejecta_mass"],
            mej=params["ejecta_mass"],
            vej=velocity,
            kappa=params["kappa"],
            kappa_gamma=params["kappa_gamma"],
        )
        photosphere = redback.photosphere.TemperatureFloor(
            time=phases[after],
            luminosity=lbol,
            vej=velocity,
            temperature_floor=params["temperature_floor"],
        )
        # The luminosity density per solid angle (erg/s/Hz/sr) divided by the distance squared.
        lum_per_solid = black_body_luminosity_density_per_solid(
            photosphere.photosphere_temperature[:, None],
            photosphere.r_photosphere[:, None],
            np.asarray(wavelengths)[None, :] * ANGSTROM_TO_CM,
        )
        distance_cm = params["distance"] * PARSEC_TO_CM
        flux_density[after] = lum_per_solid / distance_cm**2 * CGS_FNU_UNIT_TO_NJY
        return flux_density

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

        phases = np.asarray(times) - params["t0"]
        lum = self._compute_kasen_luminosity(phases, wavelengths, params["viewing_angle"])
        distance_cm = params["distance"] * PARSEC_TO_CM
        flux_density = flam_to_fnu(
            lum / (4.0 * np.pi * distance_cm**2),
            wavelengths,
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        alpha = _smoothstep(phases / self._times[-1])[:, None]
        arnett = self._compute_arnett_sed(phases, wavelengths, graph_state)
        return (1.0 - alpha) * flux_density + alpha * arnett
