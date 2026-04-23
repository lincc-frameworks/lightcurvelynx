"""A class for storing and working with SkyMapper data."""

import logging

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.coordinate_utils import build_moc_from_coords
from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ
from lightcurvelynx.noise_models.noise_utils import poisson_bandflux_std
from lightcurvelynx.obstable.obs_table import ObsTable

SKYMAPPER_PIXEL_SCALE = 0.497
"""The pixel scale for the SkyMapper's camera in arcseconds per pixel.
From https://arxiv.org/pdf/2402.02015
"""

_skymapper_readout_noise = 10.0
"""The standard deviation of the count of readout electrons per pixel for the camera.

From personal communication with the SkyMapper team.
"""

_skymapper_gain = 0.75
"""The gain for the SkyMapper camera in ADU per electron.
From https://arxiv.org/pdf/2402.02015
"""

_skymapper_ccd_radius = (
    0.5 * np.sqrt((4096 * SKYMAPPER_PIXEL_SCALE) ** 2 + (2048 * SKYMAPPER_PIXEL_SCALE) ** 2) / 3600.0
)
"""The approximate radius of the SkyMapper CCDs in degrees, computed from the pixel scale
and the CCD dimensions (2048 x 4096). All numbers from https://arxiv.org/pdf/2402.02015"""


_skymapper_dark_current = 0.0
"""The dark current for the SkyMapper camera in electrons per second per pixel. We assume it
is negligible for the purposes of this class, based on personal communication with the
SkyMapper team.
"""


class SkyMapperObsTable(ObsTable, CiteClass):
    """An ObsTable for observations from the SkyMapper survey.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the SkyMapper survey information.
    colmap : dict
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
        Defaults to the SkyMapper CCDVisit column names, stored in _default_colnames.
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, SkyMapper-specific defaults will be
        used.
    make_detector_footprint : bool, optional
        If True, the detector footprint will be created based on the xSize and ySize survey
        parameters. This can not be used if a detect footprint is already provided in the input table.
    **kwargs : dict
        Additional keyword arguments to pass to the constructor. This includes overrides
        for survey parameters such as:

        - dark_current : The dark current for the camera in electrons per second per pixel.
        - gain: The gain for the camera in electrons per ADU.
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The readout noise for the camera in electrons per pixel.

    References
    ----------
    SkyMapper Southern Survey: Data Release 4 (Onken et al. 2023)
    https://arxiv.org/pdf/2402.02015
    """

    # Column names for the SkyMapper visit table (provided by the SkyMapper team)
    _default_colnames = {
        "dec": "dec_deg",  # degrees
        "exptime": "exp_time",  # seconds
        "ra": "ra_deg",  # degrees
        "rotation": "pa_deg",  # degrees
        "skybrightness": "sb_mag",  # mag/arcsec^2
        "time": "mjd_midpt",  # days
        "zp_mag_adu": "zeropoint",  # magnitudes to produce 1 count (ADU)
    }

    # Default survey values (SkyMapper).
    _default_survey_values = {
        "ccd_pixel_width": 2048,  # Detector width in pixels
        "ccd_pixel_height": 4096,  # Detector height in pixels
        "dark_current": _skymapper_dark_current,
        "gain": _skymapper_gain,
        "pixel_scale": SKYMAPPER_PIXEL_SCALE,
        "radius": _skymapper_ccd_radius,
        "read_noise": _skymapper_readout_noise,
        "survey_name": "SkyMapper",
    }

    # Default SkyMapper saturation thresholds in magnitudes for the main
    # survey. From https://skymapper.anu.edu.au/surveys/
    _default_saturation_mags = {
        "u": 10.0,
        "v": 10.5,
        "g": 13.0,
        "r": 13.0,
        "i": 11.0,
        "z": 10.5,
    }

    # Default PSF FWHM values in arcseconds for each filter, from
    # https://arxiv.org/pdf/2402.02015, Table 1 (median values)
    _default_psf_fwhm = {
        "u": 3.15,  # arcseconds
        "v": 3.00,  # arcseconds
        "g": 2.80,  # arcseconds
        "r": 2.63,  # arcseconds
        "i": 2.54,  # arcseconds
        "z": 2.49,  # arcseconds
    }

    # Class constants for the column names.
    def __init__(
        self,
        table,
        *,
        colmap=None,
        saturation_mags=None,
        make_detector_footprint=False,
        **kwargs,
    ):
        colmap = self._default_colnames if colmap is None else colmap

        # If saturation thresholds are not provided, then set to the defaults.
        if saturation_mags is None:
            saturation_mags = self._default_saturation_mags

        super().__init__(table, colmap=colmap, saturation_mags=saturation_mags, **kwargs)

        # Construct a default detector footprint if requested. We use the same (average) footprint for
        #  all CCDs based on the survey parameters for pixel scale and CCD size.
        if make_detector_footprint:
            if "detector_footprint" in kwargs:
                raise ValueError("Cannot provide a detector footprint if make_detector_footprint is True.")
            pixel_scale = self.survey_values.get("pixel_scale")
            width_px = self.survey_values.get("ccd_pixel_width")
            height_px = self.survey_values.get("ccd_pixel_height")
            detect_fp = DetectorFootprint.from_pixel_rect(width_px, height_px, pixel_scale=pixel_scale)
            self.set_detector_footprint(detect_fp)

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy (which produces 1 e-) to the SkyMapperObsTable tables."""
        cols = self._table.columns.to_list()

        if "zp" in cols:
            return  # Nothing to do

        # If the zero point column is already present (as a magnitude),
        # we convert it to nJy (per electron).
        if "zp_mag_adu" in cols:
            zp_values = mag2flux(self._table["zp_mag_adu"]) / self.safe_get_survey_value("gain")
            self.add_column("zp", zp_values, overwrite=True)
            return
        if "zp_mag_e" in cols:
            zp_values = mag2flux(self._table["zp_mag_e"])
            self.add_column("zp", zp_values, overwrite=True)
            return

        raise ValueError("Not enough information to compute the zero points.")

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the SkyMapperObsTable table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        observations = self._table.iloc[index]

        # By the effective FWHM definition, see
        # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
        # We need it in pixel^2
        pixel_scale = self.safe_get_survey_value("pixel_scale")
        if "fwhm" in observations:
            seeing = observations["fwhm"].values
        elif "seeing" in observations:
            seeing = observations["seeing"].values
        else:
            # Use the median seeing per-filter if the seeing is not provided for each observation.
            seeing = np.zeros(len(observations))
            for filter, fwhm_arcsec in self._default_psf_fwhm.items():
                filter_mask = observations["filter"] == filter
                seeing[filter_mask] = fwhm_arcsec
        psf_footprint = GAUSS_EFF_AREA2FWHM_SQ * (seeing / pixel_scale) ** 2

        # Get the computed zeropoints in nJy/electron.
        zp = observations["zp"]

        # Convert skybrightness (mag/arcsec^2) -> nJy/arcsec^2 -> nJy/pixel^2,
        # then divide by zp (nJy/electron) to get electrons/pixel^2.
        sky = mag2flux(observations["skybrightness"]) * pixel_scale**2 / zp

        return poisson_bandflux_std(
            bandflux,
            total_exposure_time=observations["exptime"],
            exposure_count=1,
            psf_footprint=psf_footprint,
            sky=sky,
            zp=zp,
            readout_noise=self.safe_get_survey_value("read_noise"),
            dark_current=self.safe_get_survey_value("dark_current"),
            zp_err_mag=0.0,
        )

    def build_moc(
        self,
        *,
        max_depth=10,
        **kwargs,
    ):
        """Build a Multi-Order Coverage Map from the regions in the data set.

        Because SkyMapper data is given at the CCD-level, we use a sampling
        based approach for MOC construction.

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of the MOC. Default is 10.
        kwargs : dict
            Additional keyword that are passed to other build_moc implementations, but are
            not used for this class.

        Returns
        -------
        MOC
            The Multi-Order Coverage Map constructed from the data set.
        """
        logging.getLogger(__name__).debug(f"Building MOC from SkyMapperObsTable data: Depth={max_depth}.")

        # Deduplicate near-matching pointings to save computation time.
        ra = self._table["ra"].to_numpy()
        dec = self._table["dec"].to_numpy()
        moc = build_moc_from_coords(ra, dec, depth=max_depth)
        return moc
