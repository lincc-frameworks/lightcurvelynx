"""Model that generate the SED or bandflux of a source based on given predefined observer frame
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
import yaml
from citation_compass import cite_inline
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.math_nodes.given_sampler import GivenValueSampler
from lightcurvelynx.models.physical_model import SEDModel

logger = logging.getLogger(__name__)


class SEDTemplate:
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
        The period of this data, if it is periodic. Default is None.

    Parameters
    ----------
    grid_data : np.ndarray
        A 2D array of shape (N x 3) containing phases, wavelengths, and fluxes.
    sed_data_t0 : float, optional
        The reference epoch of the input SED template. This is the time stamp of the input
        array that will correspond to t0 in the model. Default is 0.0.
    interpolation_type : str, optional
        The type of interpolation to use. One of 'linear' or 'cubic'. Default is 'linear'.
    periodic : bool, optional
        Whether the SED template is periodic. Default is False.
    baseline : np.ndarray or None, optional
        A length W array of baseline SED values for each wavelength. This is only used
        for non-periodic SED templates when they are not active. Default is None.
    **kwargs : dict
        Additional keyword arguments that are ignored.
    """

    def __init__(
        self,
        grid_data,
        *,
        sed_data_t0=0.0,
        interpolation_type="linear",
        periodic=False,
        baseline=None,
        **kwargs,
    ):
        grid_data = np.asarray(grid_data)
        if grid_data.ndim != 2 or grid_data.shape[1] != 3:
            raise ValueError(
                f"grid_data must be a 2D array with shape (N x 3). Got {grid_data.shape} instead."
            )
        self.phases, self.wavelengths, sed_values = SEDTemplate._three_column_to_matrix(grid_data)

        # Apply the sed_data_t0 offset to the phases to get the times.
        self.times = self.phases - sed_data_t0

        # Validate the baseline data.
        if baseline is not None:
            baseline = np.asarray(baseline)
            if baseline.shape != (len(self.wavelengths),):
                raise ValueError(
                    f"baseline shape {baseline.shape} must match wavelengths shape {self.wavelengths.shape}."
                )
        self.baseline = baseline

        # If the SED template is periodic, validate the input data and compute the period.
        if periodic:
            if len(self.times) < 2:
                raise ValueError("At least two time points are required for periodic SED templates.")
            if not np.isclose(self.times[0], 0.0):
                self.times = self.times - self.times[0]
            if not np.allclose(sed_values[0, :], sed_values[-1, :]):
                raise ValueError(
                    "For periodic SED templates, the first and last SED values for "
                    "each wavelength must match."
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

    @property
    def is_periodic(self):
        """Whether this SED template is periodic."""
        return self.period is not None

    @classmethod
    def from_file(cls, file_path, **kwargs):
        """Create a SEDTemplate from a file containing three-column data.

        Parameters
        ----------
        file_path : str or Path
            The path to the file containing the SED data.
        **kwargs : dict
            Additional keyword arguments to pass to the SEDTemplate constructor.

        Returns
        -------
        SEDTemplate
            The created SEDTemplate instance.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"SED data file not found: {file_path}.")

        data = np.loadtxt(file_path, comments="#")
        return cls(data, **kwargs)

    @classmethod
    def from_components(cls, times, wavelengths, sed_values, **kwargs):
        """Create a SEDTemplate from separate time, wavelength, and SED value arrays. This is
        a convenience method that packs the data into (and then later unpacks the data from) the
        required three-column format.

        Parameters
        ----------
        times : np.ndarray
            A length T array of times.
        wavelengths : np.ndarray
            A length W array of wavelengths.
        sed_values : np.ndarray
            A 2D array of shape (T x W) containing the SED values.
        **kwargs : dict
            Additional keyword arguments to pass to the SEDTemplate constructor.
        """
        times = np.asarray(times)
        wavelengths = np.asarray(wavelengths)
        sed_values = np.asarray(sed_values)

        if sed_values.shape != (len(times), len(wavelengths)):
            raise ValueError(
                f"sed_values shape {sed_values.shape} must match (len(times), len(wavelengths)) "
                f"= ({len(times)}, {len(wavelengths)})."
            )

        # Pack the data into three-column format.
        grid_data = np.zeros((len(times) * len(wavelengths), 3))
        row = 0
        for t_idx, time in enumerate(times):
            for w_idx, wavelength in enumerate(wavelengths):
                grid_data[row, 0] = time
                grid_data[row, 1] = wavelength
                grid_data[row, 2] = sed_values[t_idx, w_idx]
                row += 1

        return cls(grid_data, **kwargs)

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
        sed_matrix : np.ndarray
            A 2D array of shape (T x W) with fluxes.
        """
        phases = data[:, 0]
        wavelengths = data[:, 1]
        fluxes = data[:, 2]

        unique_wavelengths = np.unique(wavelengths)
        unique_phases = np.unique(phases)

        sed_matrix = np.zeros((len(unique_phases), len(unique_wavelengths)))
        for wave, phase, flux_val in zip(wavelengths, phases, fluxes, strict=False):
            wave_idx = np.where(unique_wavelengths == wave)[0][0]
            phase_idx = np.where(unique_phases == phase)[0][0]
            sed_matrix[phase_idx, wave_idx] = flux_val

        return unique_phases, unique_wavelengths, sed_matrix


class SEDTemplateModel(SEDModel):
    """A model that generates either the SED or bandflux of a source based on
    SED values at given times and wavelengths.

    SEDTemplateModel supports both periodic and non-periodic data. If the template
    is not periodic then the given values will be interpolated during the time range
    of the template. Values outside the time range (before and after) will be set to
    the baseline value for that wavelength (0.0 by default).

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
    template : SEDTemplate
        The data for the SED, including the times and bandfluxes in each filter.

    Parameters
    ----------
    template : numpy.ndarray or SEDTemplate
        The SED template information can be passed as either:
        1) a SEDTemplate instance, or
        2) a numpy array of shape (T, 3) array where the first column is phase (in days), the
        second column is wavelength (in Angstroms), and the third column is the SED value (in nJy).
    sed_data_t0 : float or None, optional
        The reference epoch of the input template. This is the time stamp of the input
        array that will correspond to t0 in the model. This is only required if the template
        is passed as a numpy array. Default is None.
    interpolation_type : str, optional
        The type of interpolation to use. One of 'linear' or 'cubic'. Default is 'linear'.
    periodic : bool, optional
        Whether the template is periodic. Default is False.
    baseline : np.ndarray or None, optional
        A length W array of baseline SED values for each wavelength. This is only used
        for non-periodic templates when they are not active. Default is None.
    """

    def __init__(
        self,
        template,
        *,
        sed_data_t0=None,
        interpolation_type="linear",
        periodic=False,
        baseline=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store the template data, parsing out different formats if needed.
        if isinstance(template, SEDTemplate):
            self.template = template
        else:
            if sed_data_t0 is None:
                raise ValueError("sed_data_t0 must be provided when template is not a SEDTemplate instance.")
            self.template = SEDTemplate(
                template,
                sed_data_t0=sed_data_t0,
                periodic=periodic,
                baseline=baseline,
                interpolation_type=interpolation_type,
            )

        # Check that t0 is set.
        if "t0" not in kwargs or kwargs["t0"] is None:
            raise ValueError("SED template models require a t0 parameter.")

    @classmethod
    def from_file(cls, file_path, **kwargs):
        """Create a SEDTemplateModel from a file containing three-column data.

        Parameters
        ----------
        file_path : str or Path
            The path to the file containing the SED data.
        **kwargs : dict
            Additional keyword arguments to pass to the SEDTemplateModel constructor.

        Returns
        -------
        SEDTemplateModel
            The created SEDTemplateModel instance.
        """
        template = SEDTemplate.from_file(file_path, **kwargs)
        return cls(template, **kwargs)

    @property
    def times(self):
        """The times of the template data (in days)."""
        return self.template.times

    @property
    def wavelengths(self):
        """The wavelengths of the template data (in Angstroms)."""
        return self.template.wavelengths

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
            A length T x N matrix of observer frame SED values (in nJy).
        """
        shifted_times = times - self.get_param(graph_state, "t0")
        return self.template.evaluate_sed(shifted_times, wavelengths)


class MultiSEDTemplateModel(SEDModel):
    """A MultiSEDTemplateModel randomly selects a SED template at each evaluation
    computes the flux from that source at given times and wavelengths.

    MultiSEDTemplateModel supports both periodic and non-periodic templates. See the
    SEDTemplate documentation for details on how each template is handled.

    Parameterized values include:
      * dec - The object's declination in degrees.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    templates : list of SEDTemplate
        The data for the templates, such as the times and bandfluxes in each filter.

    Parameters
    ----------
    templates : list of SEDTemplate
        The data for the templates, such as the times and bandfluxes in each filter.
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        a template at random. If None, all templates will be weighted equally.
    """

    def __init__(
        self,
        templates,
        *,
        weights=None,
        **kwargs,
    ):
        # Validate the input templates.
        for tmp in templates:
            if not isinstance(tmp, SEDTemplate):
                raise TypeError("Each template must be an instance of SEDTemplate.")
        self.templates = templates

        super().__init__(**kwargs)

        all_inds = [i for i in range(len(templates))]
        self._sampler_node = GivenValueSampler(all_inds, weights=weights)
        self.add_parameter(
            "selected_template",
            value=self._sampler_node,
            allow_gradient=False,
            description="Index of the SED template selected for sampling.",
        )

    def __len__(self):
        """Get the number of SED templates."""
        return len(self.templates)

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
        model_ind = self.get_param(graph_state, "selected_template")
        shifted_times = times - self.get_param(graph_state, "t0")
        return self.templates[model_ind].evaluate_sed(shifted_times, wavelengths)


class SIMSEDModel(MultiSEDTemplateModel):
    """Generate fluxes from SIMSED-formated data.

    Parameterized values include:
      * dec - The object's declination in degrees.
      * distance - The object's luminosity distance in pc.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    templates : list of SEDTemplate
        The data for the templates, such as the times and bandfluxes in each filter.
    flux_scale : float
        A scale factor to apply to all fluxes read from the SIMSED data files.
    """

    def __init__(self, templates, flux_scale=1.0, **kwargs):
        self.flux_scale = flux_scale
        super().__init__(templates, **kwargs)
        if not self.has_valid_param("distance"):
            raise ValueError(
                "SIMSEDModel requires a valid 'distance' parameter representing luminosity distance in pc. "
                "This can be specified as 'distance' directly or derived by a combination of the 'redshift' "
                "and 'cosmology' parameters."
            )

    @classmethod
    def from_dir(cls, simsed_dir, **kwargs):
        """Read SNANA-formatted data from a directory and create a SIMSEDModel.

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
        file_names, simsed_params = SIMSEDModel._read_simsed_info_file(simsed_dir)

        # Extract the parameters that we need.
        if "FLUX_SCALE" not in simsed_params:  # pragma: no cover
            warnings.warn("SIMSED SED.INFO file does not contain a FLUX_SCALE parameter. Using 1.0.")
            flux_scale = 1.0
        else:
            flux_scale = float(simsed_params["FLUX_SCALE"])

        # Read in each template file.
        templates = []
        for file_name in tqdm(file_names, total=len(file_names), desc="Loading", unit="file"):
            templates.append(SIMSEDModel._read_simsed_data_file(simsed_dir / file_name))

        # Add a citation for this data file.
        cite_inline(
            "SIMSED Data",
            f"SIMSED data files from {simsed_dir}. Check the SED.INFO file for citation information.",
        )

        return cls(templates, flux_scale=flux_scale, **kwargs)

    @staticmethod
    def _read_simsed_info_file(simsed_dir):
        """Read the SED.INFO file to get the list of template files and their properties.

        Parameters
        ----------
        simsed_dir : Path
            The directory containing the SIMSED-formatted data files.

        Returns
        -------
        file_names : list of Path
            A list of paths to the template files.
        parameters : dict
            A dictionary of parameters read from the SED.INFO file.
        """
        info_file = simsed_dir / "SED.INFO"
        if not info_file.exists():  # pragma: no cover
            raise FileNotFoundError(f"SED.INFO file not found in {simsed_dir}.")
        logger.debug(f"Reading SIMSED data from {info_file}.")

        parameters = {}
        file_names = []
        with open(info_file, "r") as f:
            # Read the header as YAML data.
            parameters = yaml.safe_load(f)

            # Reset the file pointer to read line by line for the file names,
            # because the YAML parser will only return the first one.
            f.seek(0)
            for line in f:
                line = line.strip()
                if line and line.upper().startswith("SED:"):
                    tokens = line.split()
                    file_names.append(simsed_dir / tokens[1].strip())
        return file_names, parameters

    @staticmethod
    def _read_simsed_data_file(file_path):
        """Read a simsed data file to get an individual template.

        Parameters
        ----------
        file_path : Path
            The path to the SIMSED-formatted data file.

        Returns
        -------
        SEDTemplate
            The SED template data extracted from the file.
        """
        if not file_path.exists() and file_path.suffix != ".gz":
            # If the file is not found, check for a .gz version.
            file_path = file_path.with_suffix(file_path.suffix + ".gz")
        if not file_path.exists():  # pragma: no cover
            raise FileNotFoundError(f"SIMSED data file not found: {file_path}.")

        # Read in the data file.
        data = np.loadtxt(file_path, comments="#")
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"SIMSED data file {file_path} must have three columns (phase, wavelength, flux)."
            )

        sed_data = SEDTemplate(
            data,
            sed_data_t0=0.0,
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
            A length T x N matrix of observer frame SED values (in nJy).
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
