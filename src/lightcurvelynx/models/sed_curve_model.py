"""Model that generate the SED or bandflux of a source based on given observer frame
SED curves at given wavelengths.

Note: If you are interested in generating light curves from band-level curves, use
the LightcurveTemplateModel in src/lightcurvelynx/models/lightcurve_template_model.py
instead.
"""

import logging
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from scipy.interpolate import RectBivariateSpline

from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.math_nodes.given_sampler import GivenValueSampler
from lightcurvelynx.models.physical_model import SEDModel

logger = logging.getLogger(__name__)


class LightcurveSEDData:
    """A class to hold a grid of SED data over time and wavelength, and provide
    interpolation capabilities. The quantities can use whichever units are desired.

    Attributes
    ----------
    wavelengths : np.ndarray
        A length W array of the wavelengths for the SED.
    times : np.ndarray
        A length T array of the times for the SED relative to the reference epoch.
    interp : scipy.interpolate object
        The type of interpolation to use. One of 'linear' or 'spline'.
    period : float or None
        The period of the light curve, if it is periodic. Default is None.

    Parameters
    ----------
    lightcurves : np.ndarray
        A 2D array of shape (N x 3) containing phases, wavelengths, and fluxes.
    lc_data_t0 : float, optional
        The reference epoch of the input light curve. This is the time stamp of the input
        array that will correspond to t0 in the model. Default is 0.0.
    interpolation_type : str, optional
        The type of interpolation to use. One of 'linear' or 'cubic'. Default is 'linear'.
    periodic : bool, optional
        Whether the light curve is periodic. Default is False.
    baseline : np.ndarray or None, optional
        A length W array of baseline SED values for each wavelength. This is only used
        for non-periodic light curves when they are not active. Default is None.
    """

    def __init__(
        self,
        lightcurves,
        *,
        lc_data_t0=0.0,
        interpolation_type="linear",
        periodic=False,
        baseline=None,
    ):
        lightcurves = np.asarray(lightcurves)
        if lightcurves.ndim != 2 or lightcurves.shape[1] != 3:
            raise ValueError(
                f"lightcurves must be a 2D array with shape (N x 3). Got {lightcurves.shape} instead."
            )
        self.phases, self.wavelengths, sed_values = LightcurveSEDData._three_column_to_matrix(lightcurves)

        # Apply the lc_data_t0 offset to the phases to get the times.
        self.times = self.phases - lc_data_t0

        # Validate the baseline data.
        if baseline is not None:
            baseline = np.asarray(baseline)
            if baseline.shape != (len(self.wavelengths),):
                raise ValueError(
                    f"baseline shape {baseline.shape} must match wavelengths shape {self.wavelengths.shape}."
                )
        self.baseline = baseline

        # If the light curve is periodic, validate the input data and compute the period.
        if periodic:
            if len(self.times) < 2:
                raise ValueError("At least two time points are required for periodic light curves.")
            if not np.isclose(self.times[0], 0.0):
                self.times = self.times - self.times[0]
            if not np.allclose(sed_values[0, :], sed_values[-1, :]):
                raise ValueError(
                    "For periodic light curves, the first and last SED values for each wavelength must match."
                )
            self.period = self.times[-1] - self.times[0]
        else:
            self.period = None

        # Set up the interpolation object for this SED.
        interp_degree = 3 if interpolation_type == "cubic" else 1
        self.interp = RectBivariateSpline(
            self.times,
            self.wavelengths,
            sed_values,
            kx=interp_degree,
            ky=interp_degree,
        )

    def evaluate_sed(self, times, wavelengths):
        """Evaluate the SED at the given times and wavelengths.

        Parameters
        ----------
        times : np.ndarray
            A length T array of times (in the given units) at which to evaluate the SED
            (relative to t0).
        wavelengths : np.ndarray
            A length W array of wavelengths (in the given units) at which to evaluate the SED.

        Returns
        -------
        sed_values : np.ndarray
            A (T x W) matrix of SED values (in the given units) at the given times and wavelengths.
        """
        if self.period is None:
            sed_values = np.zeros((len(times), len(wavelengths)))

            in_range = (times >= self.times[0]) & (times <= self.times[-1])
            sed_values[in_range, :] = self.interp(times[in_range], wavelengths, grid=True)

            if self.baseline is not None:
                sed_values[~in_range, :] = self.baseline[np.newaxis, :]
        else:
            # Create the modulo times for periodic evaluation and an inverse mapping to original order.
            times = np.mod(times, self.period)
            argsort_idx = np.argsort(times)
            inv_idx = np.zeros_like(argsort_idx)
            inv_idx[argsort_idx] = np.arange(len(times))

            sed_values = self.interp(times[argsort_idx], wavelengths, grid=True)
            sed_values = sed_values[inv_idx, :]
        return sed_values

    @staticmethod
    def _three_column_to_matrix(data):
        """Convert 3-column SED data to a matrix form.

        Parameters
        ----------
        data : np.ndarray
            A 2D array of shape (N x 3) containing phases, wavelengths, and fluxes.

        Returns
        -------
        unique_phases : np.ndarray
            A length T array of unique phases.
        unique_wavelengths : np.ndarray
            A length W array of unique wavelengths.
        lightcurve_matrix : np.ndarray
            A 2D array of shape (T x W) with fluxes.
        """
        phases = data[:, 0]
        wavelengths = data[:, 1]
        fluxes = data[:, 2]

        unique_wavelengths = np.unique(wavelengths)
        unique_phases = np.unique(phases)

        lightcurve_matrix = np.zeros((len(unique_phases), len(unique_wavelengths)))
        for wave, phase, flux_val in zip(wavelengths, phases, fluxes, strict=False):
            wave_idx = np.where(unique_wavelengths == wave)[0][0]
            phase_idx = np.where(unique_phases == phase)[0][0]
            lightcurve_matrix[phase_idx, wave_idx] = flux_val

        return unique_phases, unique_wavelengths, lightcurve_matrix


class SEDCurveModel(SEDModel):
    """A model that generates either the SED or bandflux of a source based on
    SED values at given times and wavelengths.

    SEDCurveModel supports both periodic and non-periodic light curves. If the
    light curve is not periodic then each light curve's given values will be interpolated
    during the time range of the light curve. Values outside the time range (before and
    after) will be set to the baseline value for that wavelength (0.0 by default).

    Parameterized values include:
      * dec - The object's declination in degrees.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Notes
    -----
    If you are interested in generating light curves from band-level curves, use
    the LightcurveTemplateModel in src/lightcurvelynx/models/lightcurve_template_model.py
    instead.

    Attributes
    ----------
    lightcurves : LightcurveSEDData
        The data for the light curves, such as the times and bandfluxes in each filter.

    Parameters
    ----------
    lightcurves : numpy.ndarray or LightcurveSEDData
        The light curves can be passed as either:
        1) a LightcurveSEDData instance, or
        2) a numpy array of shape (T, 3) array where the first column is phase (in days), the
        second column is wavelength (in Angstroms), and the third column is the SED value (in nJy).
    lc_data_t0 : float
        The reference epoch of the input light curve. This is the time stamp of the input
        array that will correspond to t0 in the model. For periodic light curves, this either
        must be set to the first time of the light curve or set as 0.0 to automatically
        derive the lc_data_t0 from the light curve.
    interpolation_type : str, optional
        The type of interpolation to use. One of 'linear' or 'cubic'. Default is 'linear'.
    periodic : bool, optional
        Whether the light curve is periodic. Default is False.
    baseline : np.ndarray or None, optional
        A length W array of baseline SED values for each wavelength. This is only used
        for non-periodic light curves when they are not active. Default is None.
    """

    def __init__(
        self,
        lightcurves,
        lc_data_t0,
        *,
        interpolation_type="linear",
        periodic=False,
        baseline=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store the light curve data, parsing out different formats if needed.
        if isinstance(lightcurves, LightcurveSEDData):
            self.lightcurves = lightcurves
        else:
            self.lightcurves = LightcurveSEDData(
                lightcurves,
                lc_data_t0=lc_data_t0,
                periodic=periodic,
                baseline=baseline,
                interpolation_type=interpolation_type,
            )

        # Check that t0 is set.
        if "t0" not in kwargs or kwargs["t0"] is None:
            raise ValueError("SED curve models require a t0 parameter.")

    @property
    def times(self):
        """The times of the light curve data (in days)."""
        return self.lightcurves.times

    @property
    def wavelengths(self):
        """The wavelengths of the light curve data (in Angstroms)."""
        return self.lightcurves.wavelengths

    def compute_sed(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy). These are generated
            from non-overlapping box-shaped SED basis functions for each filter and
            scaled by the light curve values.
        """
        shifted_times = times - self.get_param(graph_state, "t0")
        return self.lightcurves.evaluate_sed(shifted_times, wavelengths)


class MultiSEDCurveModel(SEDModel):
    """A MultiSEDCurveModel randomly selects a SED-based light curve at each evaluation
    computes the flux from that source at given times and wavelengths.

    MultiSEDCurveModel supports both periodic and non-periodic light curves. See the
    LightcurveSEDData documentation for details on how each light curve is handled.

    Parameterized values include:
      * dec - The object's declination in degrees.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    lightcurves : list of LightcurveSEDData
        The data for the light curves, such as the times and bandfluxes in each filter.

    Parameters
    ----------
    lightcurves : list of LightcurveSEDData
        The data for the light curves, such as the times and bandfluxes in each filter.
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        a light curve at random. If None, all light curves will be weighted equally.
    """

    def __init__(
        self,
        lightcurves,
        *,
        weights=None,
        **kwargs,
    ):
        # Validate the light curve input.
        for lc in lightcurves:
            if not isinstance(lc, LightcurveSEDData):
                raise TypeError("Each light curve must be an instance of LightcurveSEDData.")
        self.lightcurves = lightcurves

        super().__init__(**kwargs)

        all_inds = [i for i in range(len(lightcurves))]
        self._sampler_node = GivenValueSampler(all_inds, weights=weights)
        self.add_parameter("selected_lightcurve", value=self._sampler_node, allow_gradient=False)

    def __len__(self):
        """Get the number of light curves."""
        return len(self.lightcurves)

    def compute_sed(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy). These are generated
            from non-overlapping box-shaped SED basis functions for each filter and
            scaled by the light curve values.
        """
        # Use the light curve selected by the sampler node to compute the flux density.
        model_ind = self.get_param(graph_state, "selected_lightcurve")
        shifted_times = times - self.get_param(graph_state, "t0")
        return self.lightcurves[model_ind].evaluate_sed(shifted_times, wavelengths)


class SIMSEDModel(MultiSEDCurveModel):
    """Generate lightcurves from SIMSED-formated data.

    Parameterized values include:
      * dec - The object's declination in degrees.
      * distance - The object's luminosity distance in pc.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    lightcurves : list of LightcurveSEDData
        The data for the light curves, such as the times and bandfluxes in each filter.
    flux_scale : float
        A scale factor to apply to all fluxes read from the SIMSED data files.
    """

    def __init__(self, lightcurves, flux_scale=1.0, **kwargs):
        self.flux_scale = flux_scale
        super().__init__(lightcurves=lightcurves, **kwargs)

    @classmethod
    def from_dir(cls, simsed_dir, **kwargs):
        """Read SIMSED-formatted data from a directory and create a SIMSEDModel.

        Parameters
        ----------
        simsed_dir : str or Path
            The directory containing the SIMSED-formatted data files.
        **kwargs : dict
            Additional keyword arguments to pass to the SIMSEDModel constructor.

        Returns
        -------
        SIMSEDModel
            The created SIMSEDModel instance.
        """
        simsed_dir = Path(simsed_dir)
        logger.debug(f"Reading SIMSED data from {simsed_dir}.")
        lc_files, simsed_params = SIMSEDModel._read_simsed_info_file(simsed_dir)

        # Extract the parameters that we need.
        if "FLUX_SCALE" not in simsed_params:
            warnings.warn("SIMSED SED.INFO file does not contain a FLUX_SCALE parameter. Using 1.0.")
            flux_scale = 1.0
        else:
            flux_scale = float(simsed_params["FLUX_SCALE"])

        # Read in each light curve file.
        lightcurves = [SIMSEDModel._read_simsed_data_file(simsed_dir / lc_file) for lc_file in lc_files]
        return cls(lightcurves, flux_scale=flux_scale, **kwargs)

    @staticmethod
    def _read_simsed_info_file(simsed_dir):
        """Read the SED.INFO file to get the list of light curve files and their properties.

        Parameters
        ----------
        simsed_dir : Path
            The directory containing the SIMSED-formatted data files.

        Returns
        -------
        lc_files : list of Path
            A list of paths to the light curve files.
        parameters : dict
            A dictionary of parameters read from the SED.INFO file.
        """
        info_file = simsed_dir / "SED.INFO"
        if not info_file.exists():
            raise FileNotFoundError(f"SED.INFO file not found in {simsed_dir}.")
        logger.debug(f"Reading SIMSED data from {info_file}.")

        parameters = {}
        lc_files = []
        with open(info_file, "r") as f:
            for line in f:
                # Remove blank space and comments.
                line = line.strip()
                comment_start = line.find("#")
                if comment_start != -1:
                    line = line[:comment_start].strip()

                key_end = line.find(":")
                if key_end != -1:
                    key = line[:key_end].strip().upper()
                    value = line[key_end + 1 :].strip()
                    if key == "SED":
                        lc_files.append(simsed_dir / value.split()[0])
                    else:
                        parameters[key] = value
        return lc_files, parameters

    @staticmethod
    def _read_simsed_data_file(file_path):
        """Read a simsed data file to get an individual light curve.

        Parameters
        ----------
        file_path : Path
            The path to the SIMSED-formatted data file.

        Returns
        -------
        LightcurveSEDData
            The light curve data extracted from the file.
        """
        if not file_path.exists() and file_path.suffix != ".gz":
            # If the file is not found, check for a .gz version.
            file_path = file_path.with_suffix(file_path.suffix + ".gz")
        if not file_path.exists():
            raise FileNotFoundError(f"SIMSED data file not found: {file_path}.")

        # Read in the data file.
        data = np.loadtxt(file_path, comments="#")
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"SIMSED data file {file_path} must have three columns (phase, wavelength, flux)."
            )

        sed_data = LightcurveSEDData(
            data,
            lc_data_t0=0.0,
            interpolation_type="linear",
            periodic=False,
        )
        return sed_data

    def compute_sed(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy). These are generated
            from non-overlapping box-shaped SED basis functions for each filter and
            scaled by the light curve values.
        """
        sed_values_flam = super().compute_sed(times, wavelengths, graph_state)

        # SIMSED data files provide flux in units of erg/cm²/s/Å at 10 pc (with a scale factor), so we
        # apply the scale factor, account for the actual distance, and convert it to fnu in nJy.
        luminosity_distance_pc = self.get_param(graph_state, "distance")
        if luminosity_distance_pc is None or luminosity_distance_pc <= 0:
            raise ValueError(
                f"Received invalid luminosity distance (pc) in SIMSED model {luminosity_distance_pc}."
            )
        sed_values_fnu = flam_to_fnu(
            flux_flam=sed_values_flam * self.flux_scale * (10 / luminosity_distance_pc) ** 2,
            wavelengths=wavelengths,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        return sed_values_fnu
