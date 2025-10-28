import logging

import numpy as np

from lightcurvelynx.astro_utils.noise_model import poisson_bandflux_std
from lightcurvelynx.obstable.obs_table import ObsTable
from lightcurvelynx.obstable.obs_table_params import ParamDeriver

logger = logging.getLogger(__name__)


class FakeObsTable(ObsTable):
    """A subclass for a (simplified) fake survey. The user must provide a constant
    flux error to use or enough information to compute the poisson_bandflux_std noise model.

    The class uses a flexible deriver to try to compute any missing parameters needed from
    what is provided.

    Defaults are set for other parameters (e.g. exptime, nexposure, read_noise, dark_current), which
    the user can override with keyword arguments to the constructor.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the ObsTable information.  Must have columns
        "time", "ra", "dec", and "filter".
    colmap : dict, optional
        A mapping of standard column names to their names in the input table.
    zp_per_band : dict, optional
        A dictionary mapping filter names to their instrumental zero points (flux in nJy
        corresponding to 1 electron per exposure). The filters provided must match those
        in the table. This is required if the table does not have a zero point column.
    const_flux_error : float or dict, optional
        If provided, use this constant flux error (in nJy) for all observations (overriding
        the normal noise compuation). A value of 0.0 will produce a noise-free simulation.
        If a dictionary is provided, it should map filter names to constant flux errors per-band.
        This setting should primarily be used for testing purposes.
    radius : float, optional
        The angular radius of the field of view of the observations in degrees (default=None).
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, saturation effects will not be applied.
    noise_strategy : str, optional
        The name of the strategy to use to derive any missing table parameters needed to compute the noise.
        This is used if the table does not provide all the necessary parameters for the given noise model.
        Should be one of:
        - "given_only" : Use only the parameters already provided in the table and survey values.
        - "five_sigma_depth": Derive approximate noise from only the 5-sigma depth values if available
          (no survey values like PSF, sky background, etc. are used).
        - "exhaustive": Try all available derivation methods to fill in missing parameters.
        Default is "given_only" which does not attempt any derivation.
    **kwargs : dict
        Additional keyword arguments to pass to the ObsTable constructor. This includes overrides
        for survey parameters such as:
        - survey_name: The name of the survey (default="FAKE_SURVEY").
    """

    # Default survey values.
    _default_survey_values = {
        "dark_current": 0,
        "exptime": 30,  # seconds
        "fwhm_px": None,  # pixels
        "nexposure": 1,  # exposures
        "radius": None,  # degrees
        "read_noise": 0,  # electrons
        "sky_bg_electrons": None,  # electrons / pixel^2
        "survey_name": "FAKE_SURVEY",
    }

    def __init__(
        self,
        table,
        *,
        colmap=None,
        const_flux_error=None,
        noise_strategy="given_only",
        **kwargs,
    ):
        # Pass along all the survey parameters to the parent class.
        super().__init__(
            table,
            colmap=colmap,
            **kwargs,
        )

        # Derive any missing parameters needed for the flux error computation. We always create
        # a new ParamDeriver instance here, because they are stateful.
        if noise_strategy == "given_only":
            param_deriver = "NoopParamDeriver"
        elif noise_strategy == "five_sigma_depth":
            param_deriver = "FiveSigmaDepthParamDeriver"
        elif noise_strategy == "exhaustive":
            param_deriver = "FullParamDeriver"
        else:
            raise ValueError(
                f"Invalid noise_strategy '{noise_strategy}'. "
                "Should be one of: 'given_only', 'five_sigma_depth', 'exhaustive'."
            )
        param_deriver_obj = ParamDeriver.create_deriver(param_deriver)
        param_deriver_obj.derive_parameters(self)

        # If a constant flux error is provided, validate it.
        if const_flux_error is not None:
            # Convert a constant into a per-band dictionary.
            if isinstance(const_flux_error, int | float):
                const_flux_error = {fil: const_flux_error for fil in self.filters}

            # Check that every filter occurs in the dictionary.
            for fil in self.filters:
                if fil not in const_flux_error:
                    raise ValueError(
                        "`const_flux_error` must include all the filters in the table. Missing '{fil}'."
                    )

            # Translate the constant flux errors into a table column.
            bandflux_errors = np.array([const_flux_error[fil] for fil in self._table["filter"]])
            if np.any(bandflux_errors < 0):
                raise ValueError("Constant flux errors must be non-negative.")
            self._table["bandflux_error"] = bandflux_errors

        # Validate that we have enough information to compute flux errors.
        if "zp" not in self._table and "bandflux_error" not in self._table:
            column_names = self._table.columns.tolist()
            param_names = list(self.survey_values.keys())
            raise ValueError(
                f"Insufficient information to compute flux errors or zeropoints using {param_deriver}. "
                f"Table columns: {column_names}. Survey parameters: {param_names}."
            )

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the OpSim table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        # If we have a bandflux_error in the table (including from constant flux error), use that.
        if "bandflux_error" in self._table:
            return self.get_value_per_row("bandflux_error", indices=index)

        # Otherwise compute the flux error using the poisson_bandflux_std noise model.
        return poisson_bandflux_std(
            bandflux,
            total_exposure_time=self.get_value_per_row("exptime", indices=index),
            exposure_count=self.get_value_per_row("nexposure", indices=index),
            psf_footprint=self.get_value_per_row("psf_footprint", indices=index),
            sky=self.get_value_per_row("sky_bg_electrons", indices=index),
            zp=self.get_value_per_row("zp", indices=index),
            readout_noise=self.safe_get_survey_value("read_noise"),
            dark_current=self.safe_get_survey_value("dark_current"),
        )
