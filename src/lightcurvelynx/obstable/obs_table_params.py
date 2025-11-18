"""This is a file with helper functions for deriving different parameters from an ObsTable."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.astro_utils.zeropoint import (
    calculate_zp_from_maglim,
    sky_bg_adu_to_electrons,
)
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ

logger = logging.getLogger(__name__)


class _ParamFormula:
    """Class to formulas for deriving parameters.

    Attributes
    ----------
    parameter : str
        The name of the parameter.
    inputs : list of str
        The names of the input parameters required for the formula.
    formula : function
        The function that computes the parameter from the inputs.
    """

    def __init__(self, parameter, inputs, formula):
        self.parameter = parameter
        self.inputs = inputs
        self.formula = formula

    def can_solve(self, params):
        """Check if the formula can be solved with the given parameters.

        Parameters
        ----------
        params : dict
            A dictionary mapping parameter names to their values.

        Returns
        -------
        bool
            True if the formula can be solved, False otherwise.
        """
        return all(param in params and params[param] is not None for param in self.inputs)

    def solve(self, params):
        """Call the formula with the given input parameters.

        Parameters
        ----------
        params : dict
            A dictionary mapping parameter names to their values.

        Returns
        -------
        The value of the derived parameter.
        """
        logger.debug(f"Deriving parameter {self.parameter} from {self.inputs}")
        return self.formula(**{k: params[k] for k in self.inputs})


class ParamDeriver(ABC):
    """Base class to derive parameters from an ObsTable using predefined formulas.

    Attributes
    ----------
    parameters : dict
        A dictionary mapping parameter names to their values. With None for parameters
        that do not have a valid value yet.
    formulas : list of _ParamFormula
        A list of _ParamFormula instances for deriving parameters.
    org_parameters : set of str
        A set of parameter names that are original parameters from the ObsTable.

    Parameters
    ----------
    parameter_list : iterable, optional
        A list of the parameters supported by this deriver. If None, no parameters are supported.
    """

    _registered_derivers = {}  # A mapping of class name to class object for subclasses.

    def __init__(self, parameter_list=None):
        if parameter_list is None:
            parameter_list = []
        self.parameters = {param: None for param in parameter_list}

        self.formulas = []
        self.org_parameters = set()
        self._init_formulas()

    def __init_subclass__(cls, **kwargs):
        # Register all subclasses in a dictionary mapping class name to the
        # class object, so we can programmatically create objects from the names.
        super().__init_subclass__(**kwargs)
        cls._registered_derivers[cls.__name__] = cls

    @classmethod
    def create_deriver(cls, deriver_name):
        """Create a deriver instance from its class name.

        Parameters
        ----------
        deriver_name : str
            The name of the deriver class to instantiate.

        Returns
        -------
        ParamDeriver
            An instance of the specified deriver class.

        Raises
        ------
        ValueError
            If the deriver name is not recognized.
        """
        if deriver_name not in cls._registered_derivers:
            raise ValueError(f"Unknown deriver name: {deriver_name}")
        return cls._registered_derivers[deriver_name]()

    def __str__(self):
        return (
            f"{self.__class__.__name__} with parameters: {list(self.parameters.keys())} "
            f"and {len(self.formulas)} formulas."
        )

    def add_formula(self, parameter, inputs, formula):
        """Add a new formula to the list of formulas.

        Parameters
        ----------
        parameter : str
            The name of the parameter to be derived.
        inputs : list of str
            The names of the input parameters required for the formula.
        formula : function
            The function that computes the parameter from the inputs.
        """
        # Make sure all the input and output parameters are in the parameters dict.
        if parameter not in self.parameters:
            raise KeyError(f"Parameter {parameter} is not a supported parameter.")
        for input_param in inputs:
            if input_param not in self.parameters:
                raise KeyError(f"Parameter {input_param} is not a supported parameter.")

        # Append the formula itself.
        self.formulas.append(_ParamFormula(parameter, inputs, formula))

    def init_from_obs_table(self, obs_table):
        """Initialize parameters from an ObsTable, retrieving values from either the
        columns or survey constants (survey_values) for every parameter it can find.

        Parameters
        ----------
        obs_table : ObsTable
            The observation table from which to initialize parameters.
        """
        # Get the filter row so we can unpack dictionaries if needed.
        filters = obs_table["filter"]

        # Load each parameter from the ObsTable if it exists and is valid.
        for param in self.parameters:
            if param in obs_table.columns:
                values = obs_table[param].to_numpy()
            elif param in obs_table.survey_values:
                values = obs_table.survey_values[param]
                if type(values) is dict:
                    values = np.array([values.get(band, None) for band in filters])
            else:
                values = None

            # Only use the values if they are valid (not None or NaN).
            if values is not None and not np.any(values == None):  # noqa: E711
                self.parameters[param] = values
                self.org_parameters.add(param)

    def derive_parameters(self, obs_table):
        """Derive parameters from an ObsTable using the predefined formulas, and
        update the ObsTable with the derived parameters.

        Parameters
        ----------
        obs_table : ObsTable
            The observation table from which to derive parameters.
        """
        logger.debug(f"Deriving parameters using {self.__class__.__name__}.")
        self.init_from_obs_table(obs_table)
        logger.debug(f"Starting parameter derivation with parameters: {self.parameters}")

        # Keep iterating through the formulas, trying to solve them, until we do a full iteration
        # through all of them without making any new progress.
        made_progress = True
        while made_progress:
            made_progress = False
            for formula in self.formulas:
                if self.parameters[formula.parameter] is None and formula.can_solve(self.parameters):
                    self.parameters[formula.parameter] = formula.solve(self.parameters)
                    made_progress = True
        logger.debug(f"After derivation, parameters = {self.parameters}")

        # Update the ObsTable with the derived parameters. Do not overrwrite anything already
        # there. Always prefer to add scalars to the survey values over adding entire columns.
        for param, value in self.parameters.items():
            if param not in self.org_parameters and value is not None:
                if np.isscalar(value):
                    obs_table.survey_values[param] = value
                elif np.max(np.abs(value - value[0])) < 1e-8:  # all values are effectively the same
                    obs_table.survey_values[param] = value[0]
                else:
                    obs_table.add_column(param, value)

    @abstractmethod
    def _init_formulas(self):
        """Initialize the formulas for deriving parameters. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _init_formulas().")  # pragma: no cover


class NoopParamDeriver(ParamDeriver):
    """A ParamDeriver that does nothing. Used as a placeholder."""

    def __init__(self):
        super().__init__(parameter_list=[])

    def _init_formulas(self):
        """Initialize the formulas for deriving parameters. To be implemented by subclasses."""
        pass


class FullParamDeriver(ParamDeriver):
    """Class to derive all supported parameters from an ObsTable using predefined formulas.

    Supported parameters (and their units) include:
    - adu_bias: Bias level in ADU
    - dark_current: Dark current in electrons / second / pixel
    - exptime: Exposure time in seconds
    - filter: Photometric filter (e.g., g, r, i)
    - fwhm_px: Full-width at half-maximum of the PSF in pixels
    - gain: CCD gain in electrons / ADU
    - maglim: Limiting magnitude (5-sigma) in mag
    - nexposure: Number of exposures per observation (unitless)
    - pixel_scale: Pixel scale in arcseconds per pixel
    - psf_footprint: Effective footprint of the PSF in pixels^2
    - read_noise: Read noise in electrons
    - seeing: Seeing in arcseconds
    - sky_bg_adu: Sky background in ADU / pixel
    - sky_bg_electrons: Sky background in electrons / pixel^2
    - skybrightness: Sky brightness in mag / arcsec^2
    - zp: Instrumental zero point (nJy per electron)
    - zp_per_band: Instrumental zero point per band (nJy per electron)

    Attributes
    ----------
    parameters : dict
        A dictionary mapping parameter names to their values. With None for parameters
        that do not have a valid value yet.
    formulas : list of _ParamFormula
        A list of _ParamFormula instances for deriving parameters.
    org_parameters : set of str
        A set of parameter names that are original parameters from the ObsTable.
    """

    def __init__(self):
        parameter_list = [
            "adu_bias",  # Bias level in ADU
            "dark_current",  # Dark current in electrons / second / pixel
            "exptime",  # Exposure time in seconds
            "filter",  # Photometric filter (e.g., g, r, i)
            "fwhm_px",  # Full-width at half-maximum of the PSF in pixels
            "gain",  # CCD gain in electrons / ADU
            "maglim",  # Limiting magnitude (5-sigma) in mag
            "nexposure",  # Number of exposures per observation (unitless)
            "pixel_scale",  # Pixel scale in arcseconds per pixel
            "psf_footprint",  # Effective footprint of the PSF in pixels
            "read_noise",  # Read noise in electrons
            "seeing",  # Seeing in arcseconds
            "sky_bg_adu",  # Sky background in ADU / pixel
            "sky_bg_electrons",  # Sky background in electrons / pixel^2
            "skybrightness",  # Sky brightness in mag / arcsec^2
            "zp",  # Instrumental zero point (nJy per electron)
            "zp_per_band",  # Instrumental zero point per band (nJy per electron)
        ]
        super().__init__(parameter_list=parameter_list)

    def _init_formulas(self):
        """Initialize the standard formulas for deriving parameters."""

        # Renaming transformations (makes a copy with a new name). The per-band information
        # has already been expanded out in init_from_obs_table().
        self.add_formula(parameter="zp", inputs=["zp_per_band"], formula=lambda zp_per_band: zp_per_band)

        # Formulas for deriving the psf_footprint (in pixels^2) and related information.
        self.add_formula(
            parameter="psf_footprint",
            inputs=["fwhm_px"],
            formula=lambda fwhm_px: GAUSS_EFF_AREA2FWHM_SQ * fwhm_px**2,
        )
        self.add_formula(
            parameter="fwhm_px",
            inputs=["pixel_scale", "seeing"],
            formula=lambda pixel_scale, seeing: seeing / pixel_scale,
        )
        self.add_formula(
            parameter="fwhm_px",
            inputs=["psf_footprint"],
            formula=lambda psf_footprint: np.sqrt(psf_footprint / GAUSS_EFF_AREA2FWHM_SQ),
        )
        self.add_formula(
            parameter="seeing",
            inputs=["pixel_scale", "fwhm_px"],
            formula=lambda pixel_scale, fwhm_px: fwhm_px * pixel_scale,
        )

        # Formulas for deriving the sky background (in electrons / pixel^2) and related information.
        self.add_formula(
            parameter="sky_bg_electrons",
            inputs=["skybrightness", "pixel_scale", "zp"],
            formula=lambda skybrightness, pixel_scale, zp: (mag2flux(skybrightness) * pixel_scale**2 / zp),
        )
        self.add_formula(
            parameter="sky_bg_electrons",
            inputs=["sky_bg_adu", "gain"],
            formula=sky_bg_adu_to_electrons,
        )
        self.add_formula(
            parameter="sky_bg_adu",
            inputs=["sky_bg_electrons", "gain"],
            formula=lambda sky_electrons, gain: sky_electrons / gain,
        )

        # Formulas for deriving the zero point (in nJy per electron) and related information.
        self.add_formula(
            parameter="zp",
            inputs=[
                "maglim",
                "sky_bg_electrons",
                "fwhm_px",
                "read_noise",
                "dark_current",
                "exptime",
                "nexposure",
            ],
            formula=calculate_zp_from_maglim,
        )


class FiveSigmaDepthDeriver(ParamDeriver):
    """Class to derive the noise parameters from only the five-sigma depth information.

    Supported parameters (and their units) include:
    - bandflux_error: The error associated with the computed bandflux.
    - bandflux_ref: The total flux that would be transmitted through the given bandfilter.
    - five_sigma_depth: Five-sigma depth in magnitudes

    Attributes
    ----------
    parameters : dict
        A dictionary mapping parameter names to their values. With None for parameters
        that do not have a valid value yet.
    formulas : list of _ParamFormula
        A list of _ParamFormula instances for deriving parameters.
    org_parameters : set of str
        A set of parameter names that are original parameters from the ObsTable.
    """

    def __init__(self):
        parameter_list = [
            "bandflux_error",  # The error associated with the computed bandflux.
            "bandflux_ref",  # The total flux that would be transmitted through the given bandfilter.
            "five_sigma_depth",  # Five-sigma depth in magnitudes
        ]
        super().__init__(parameter_list=parameter_list)

    @staticmethod
    def _bandflux_error_from_five_sigma_depth(five_sigma_depth, bandflux_ref):
        """Derive the bandflux error from the five-sigma depth and bandflux reference.

        Based on redback's bandflux_error_from_limiting_mag()
        https://github.com/nikhil-sarin/redback

        Parameters
        ----------
        five_sigma_depth : float or np.ndarray
            The five-sigma depth in magnitudes.
        bandflux_ref : float or np.ndarray
            The reference bandflux.

        Returns
        -------
        bandflux_error : float or np.ndarray
            The error associated with the computed bandflux.
        """
        flux_five_sigma = bandflux_ref * np.power(10.0, -0.4 * five_sigma_depth)
        bandflux_error = flux_five_sigma / 5.0
        return bandflux_error

    def _init_formulas(self):
        self.add_formula(
            parameter="bandflux_error",
            inputs=["five_sigma_depth", "bandflux_ref"],
            formula=FiveSigmaDepthDeriver._bandflux_error_from_five_sigma_depth,
        )
