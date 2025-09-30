import numpy as np

from lightcurvelynx.astro_utils.noise_model import poisson_bandflux_std
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ
from lightcurvelynx.obstable.obs_table import ObsTable


class FakeObsTable(ObsTable):
    """A subclass for a (simplified) fake survey. The user must provide a constant
    flux error to use or enough information to compute the poisson_bandflux_std noise model.
    To compute the flux error, the user must provide the following values
    either in the table or as keyword arguments to the constructor 1) sky, 2) zp_per_band,
    and 3) either a) fwhm_px or b) psf_footprint.

    Defaults are set for other parameters (e.g. exptime, nexposure, read_noise, dark_current), which
    the user can override with keyword arguments to the constructor.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the ObsTable information.  Must have columns
        "time", "ra", "dec", and "filter".
    colmap : dict, optional
        A mapping of standard column names to their names in the input table.
    zp_per_band : dict
        A dictionary mapping filter names to their instrumental zero points (flux in nJy
        corresponding to 1 electron per exposure). The filters provided must match those
        in the table. This is required if the table does not have a zero point column.
    const_flux_error : float or dict, optional
        If provided, use this constant flux error (in nJy) for all observations (overriding
        the normal noise compuation). If a dictionary is provided, it should map filter names
        to constant flux errors per-band. This should only be used for testing purposes.
    dark_current : float, optional
        The dark current for the camera in electrons per second per pixel (default=0.0).
    exptime : float, optional
        The exposure time for the camera in seconds, used for dark current calculation only
        (default=30).
    psf_footprint : float, optional
        The effective psf_footprint of the PSF in pixels.
    fwhm_px : float or dict, optional
        The full-width at half-maximum of the PSF in pixels. If a dictionary is provided,
        it should map filter names to fwhm values. This is only needed if `psf_footprint` is not provided
        and `const_flux_error` is not provided (default=None).
    nexposure : int, optional
        The number of exposures per observation (default=1).
    radius : float, optional
        The angular radius of the field of view of the observations in degrees (default=None).
    read_noise : float, optional
        The read noise for the camera in electrons (default=0.0).
    sky : float or dict, optional
        The sky background in the units of electrons / pixel^2. If a dictionary is provided,
        it should map filter names to sky values. This is only needed if `const_flux_error`
        is not provided (default=None).
    saturation_thresholds : dict, optional
        A dictionary mapping filter names to their saturation thresholds in nJy. The filters provided
        must match those in the table. If not provided, saturation effects will not be applied.
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
        "sky": None,  # electrons / pixel^2
        "survey_name": "FAKE_SURVEY",
    }

    def __init__(
        self,
        table,
        *,
        colmap=None,
        zp_per_band=None,
        const_flux_error=None,
        dark_current=0,
        exptime=30,
        psf_footprint=None,
        fwhm_px=None,
        nexposure=1,
        radius=None,
        read_noise=0,
        sky=None,
        saturation_thresholds=None,
        **kwargs,
    ):
        self.zp_per_band = zp_per_band
        self.const_flux_error = const_flux_error

        # Pass along all the survey parameters to the parent class.
        super().__init__(
            table,
            colmap=colmap,
            dark_current=dark_current,
            exptime=exptime,
            psf_footprint=psf_footprint,
            fwhm_px=fwhm_px,
            nexposure=nexposure,
            radius=radius,
            read_noise=read_noise,
            sky=sky,
            saturation_thresholds=saturation_thresholds,
            **kwargs,
        )

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
        else:
            # Make sure we have the required columns (fwhm_px, sky, exptime, nexposure) to
            # compute the flux error. If any are missing, assign a constant column from the survey values.
            if not self._assign_constant_if_needed("exptime", check_positive=True):
                raise ValueError("Must provide `exptime` to FakeSurveyTable.")
            if not self._assign_constant_if_needed("nexposure", check_positive=True):
                raise ValueError("Must provide `nexposure` to FakeSurveyTable.")
            if not self._assign_constant_if_needed("psf_footprint", check_positive=True):
                if not self._assign_constant_if_needed("fwhm_px", check_positive=True):
                    raise ValueError("Must provide either `psf_footprint` or `fwhm_px` to FakeSurveyTable.")
                psf_footprint = GAUSS_EFF_AREA2FWHM_SQ * (self._table["fwhm_px"]) ** 2
                self.add_column("psf_footprint", psf_footprint)
            if not self._assign_constant_if_needed("sky", check_positive=False):
                raise ValueError("Must provide `sky` to FakeSurveyTable.")

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy to the ObsTable. In this fake
        survey, we use a constant zero point per band.
        """
        # Check that we either have previously assigned zero points or have been given
        # a dictionary of zero points per band.
        if "zp" in self._table.columns:
            return  # Already have a column of zero points.
        if self.zp_per_band is None:
            raise ValueError("Must provide `zp_per_band` to FakeSurveyTable without a column of zero points.")
        if "filter" not in self._table.columns:
            raise ValueError(
                "Must provide a `filter` column to FakeSurveyTable without a column of zero points."
            )

        # Check that we have a zero point for every filter in the table.
        for fil in self.filters:
            if fil not in self.zp_per_band:
                raise ValueError(f"Must provide a zero point for filter {fil} in `zp_per_band`.")

        # Create a column of zero points, setting all values based on the filter column.
        zp_col = np.zeros(len(self._table), dtype=float)
        for key, val in self.zp_per_band.items():
            if val <= 0:
                raise ValueError(f"Zero point for band {key} must be positive. Got {val}.")
            zp_col[self._table["filter"] == key] = val
        self.add_column("zp", zp_col, overwrite=True)

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
        # We insert most the needed columns during construction, so we
        # can look up most of the values needed for the noise model.
        observations = self._table.iloc[index]
        psf_footprint = observations["psf_footprint"]

        return poisson_bandflux_std(
            bandflux,
            total_exposure_time=observations["exptime"],
            exposure_count=observations["nexposure"],
            psf_footprint=psf_footprint,
            sky=observations["sky"],
            zp=observations["zp"],
            readout_noise=self.safe_get_survey_value("read_noise"),
            dark_current=self.safe_get_survey_value("dark_current"),
        )
