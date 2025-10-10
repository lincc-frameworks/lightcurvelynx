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
    Users can create a completely noise-free survey by providing a constant flux error of 0.

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
    dark_current : float, optional
        The dark current for the camera in electrons per second per pixel (default=0.0).
    exptime : float, optional
        The exposure time for the camera in seconds, used for dark current calculation only
        (default=30).
    psf_footprint : float, optional
        The effective psf_footprint of the PSF in pixels^2.
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
        saturation_mags=None,
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
            saturation_mags=saturation_mags,
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
            exptime = self.get_value_per_row("exptime")
            if np.any(exptime == None) or np.any(exptime <= 0):  # noqa: E711
                raise ValueError("Must provide a positive `exptime` to FakeSurveyTable.")

            nexposure = self.get_value_per_row("nexposure")
            if np.any(nexposure == None) or np.any(nexposure <= 0):  # noqa: E711
                raise ValueError("Must provide a positive `nexposure` to FakeSurveyTable.")

            psf_footprint = self.get_value_per_row("psf_footprint")
            if np.any(psf_footprint == None) or np.any(psf_footprint <= 0):  # noqa: E711
                fwhm_px = self.get_value_per_row("fwhm_px")
                if np.any(fwhm_px == None) or np.any(fwhm_px <= 0):  # noqa: E711
                    raise ValueError(
                        "Must provide a positive `psf_footprint` or `fwhm_px` to FakeSurveyTable."
                    )
                psf_footprint = GAUSS_EFF_AREA2FWHM_SQ * (fwhm_px) ** 2
                self.add_column("psf_footprint", psf_footprint)

            sky = self.get_value_per_row("sky")
            if np.any(sky == None) or np.any(sky < 0):  # noqa: E711
                raise ValueError("Must provide `sky` to FakeSurveyTable.")

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy to the ObsTable. In this fake
        survey, we use a constant zero point per band.
        """
        # Check that we either have previously assigned zero points, do not need zero points,
        # or have been given a dictionary of zero points per band.
        if "zp" in self._table.columns:
            return  # Already have a column of zero points.
        if self.const_flux_error is not None:
            return  # Do not need zero points if using a constant flux error.
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
        return poisson_bandflux_std(
            bandflux,
            total_exposure_time=self.get_value_per_row("exptime", indices=index),
            exposure_count=self.get_value_per_row("nexposure", indices=index),
            psf_footprint=self.get_value_per_row("psf_footprint", indices=index),
            sky=self.get_value_per_row("sky", indices=index),
            zp=self.get_value_per_row("zp", indices=index),
            readout_noise=self.safe_get_survey_value("read_noise"),
            dark_current=self.safe_get_survey_value("dark_current"),
        )
