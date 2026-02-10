"""The ArgusObsTable stores observation information from the Argus survey."""


from lightcurvelynx.obstable.obs_table import ObsTable

_argus_view_radius = 52.0
"""The angular radius of the observation field (in degrees):
https://argus.unc.edu/about
"""

_argus_pixel_scale = 1.0
"""The pixel scale for the Argus survey in arcseconds per pixel:
https://argus.unc.edu/specifications
"""


class ArgusObsTable(ObsTable):
    """An ObsTable for observations from the Argus survey.

    Unlike other ObsTable classes, the ArgusObsTable does not consist of a table of
    pointings, but rather is organized at the healpix level. Each row corresponds
    to a healpix pixel and time.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the survey information.
    colmap : dict
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, Argus-specific defaults will be
        used.
    **kwargs : dict
        Additional keyword arguments to pass to the constructor. This includes overrides
        for survey parameters such as:
        - dark_electrons : The dark current for the camera in electrons per second per pixel.
        - gain: The gain for the camera in electrons per ADU.
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The readout noise for the camera in electrons per pixel.
    """

    # Column names from the Argus simulation files.
    _argus_sim_colmap = {
        "dark_electrons": "dark_electrons",  # ?
        "dec": "dec",  # degrees
        "exptime": "expTime",  # seconds
        "maglim": "magLim",  # magnitudes
        "ra": "ra",  # degrees
        "seeing": "seeing",  # arcseconds
        "skybrightness": "sky_brightness",  # mag/arcsec^2
        "sky_electrons": "sky_electrons",  # ?
        "time": "epoch",  # MJD
    }
    _default_colnames = _argus_sim_colmap

    # Default survey values: https://argus.unc.edu/specifications
    _default_survey_values = {
        "pixel_scale": _argus_pixel_scale,
        "radius": _argus_view_radius,
        "read_noise": 1.4,  # e-/pixel
        "zp_err_mag": None,
        "survey_name": "Argus",
    }

    # Class constants for the column names.
    def __init__(
        self,
        table,
        colmap=None,
        saturation_mags=None,
        **kwargs,
    ):
        pass

    def uses_footprint(self):
        """Return whether the ObsTable uses a detector footprint for filtering."""
        return False

    def clear_detector_footprint(self):
        """Clear the detector footprint, so no footprint filtering is done."""
        self._detector_footprint = None

    def set_detector_footprint(self, detector_footprint, wcs=None):
        """Set the detector footprint, so footprint filtering is done.

        Parameters
        ----------
        detector_footprint : astropy.regions.SkyRegion, Astropy.regions.PixelRegion, or
            DetectorFootprint
            The footprint object for the instrument's detector.
        wcs : astropy.wcs.WCS, optional
            The WCS for the footprint. Either this or pixel_scale must be provided if
            a footprint is provided as a Astropy region.
        """
        raise NotImplementedError("ArgusObsTable does not support detector footprints.")

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy (which produces 1 e-) to the LSSTObsTable tables."""
        raise NotImplementedError("ArgusObsTable does not have a noise model implemented yet.")

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the LSSTObsTable table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        raise NotImplementedError("ArgusObsTable does not have a noise model implemented yet.")
