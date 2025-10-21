import numpy as np

from lightcurvelynx.astro_utils.noise_model import poisson_bandflux_std
from lightcurvelynx.obstable.obs_table import ObsTable
from lightcurvelynx.obstable.obs_table_params import _ParamDeriver


class FakeObsTable(ObsTable):
    """A subclass for a (simplified) fake survey. The user must provide a constant
    flux error to use or enough information to compute the poisson_bandflux_std noise model.

    The class uses a flexible deriver to try to compute any missing parameters needed from
    what is provided. Suppported parameters include:
    - adu_bias: Bias level in ADU
    - dark_current: Dark current in electrons / second / pixel
    - exptime: Exposure time in seconds
    - filter: Photometric filter (e.g., g, r, i)
    - fwhm_px: Full-width at half-maximum of the PSF in pixels
    - gain: CCD gain in electrons / ADU
    - instr_zp_mag: Instrumental zero point magnitude (mag)
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
    These can be provided either as columns in the input table or as keyword arguments to the constructor.

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
        **kwargs,
    ):
        self.const_flux_error = const_flux_error

        # Pass along all the survey parameters to the parent class.
        super().__init__(
            table,
            colmap=colmap,
            **kwargs,
        )

        # Derive any missing parameters needed for the flux error computation.
        deriver = _ParamDeriver()
        deriver.derive_parameters(self)

        # If a constant flux error is provided, validate it.
        if const_flux_error is not None:
            # Convert a constant into a per-band dictionary.
            if isinstance(const_flux_error, int | float):
                self.const_flux_error = {fil: const_flux_error for fil in self.filters}

            # Check that every filter occurs in the dictionary with a non-negative value.
            for fil in self.filters:
                if fil not in self.const_flux_error:
                    raise ValueError(
                        "`const_flux_error` must include all the filters in the table. Missing '{fil}'."
                    )
            for fil, val in self.const_flux_error.items():
                if val < 0:
                    raise ValueError(f"Constant flux error for band {fil} must be non-negative. Got {val}.")
        elif "zp" not in self._table.columns:
            raise ValueError("Insufficient information to compute flux errors or zeropoints.")

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
        # If we have a constant flux error, use that.
        if self.const_flux_error is not None:
            filters = self._table["filter"].iloc[index]
            return np.array([self.const_flux_error[fil] for fil in filters])

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
