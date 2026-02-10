"""The top-level module for survey related data, such as pointing and noise
information. By default the module uses the Rubin OpSim data, but it can be
extended to other survey data as well.
"""

from __future__ import annotations  # "type1 | type2" syntax in Python <3.10

import numpy as np

from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.astro_utils.noise_model import poisson_bandflux_std
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ
from lightcurvelynx.obstable.obs_table import ObsTable

LSSTCAM_PIXEL_SCALE = 0.2
"""The pixel scale for the LSST camera in arcseconds per pixel."""

_lsstcam_readout_noise = 5.82
"""The standard deviation of the count of readout electrons per pixel for the LSST
camera. This is the average value from the two CCD types used (e2v=5.40 and ITL=6.21)
in the LSST camera.

Averaging as sqrt((5.40^2 + 6.21^2) / 2) = 5.82

The value is from https://lsstcam.lsst.io/index.html
"""

_lsstcam_dark_current = 0.022
"""The dark current for the LSST camera in electrons per second per pixel.
This is the average value from the two CCD types used (e2v=0.023 and ITL=0.021)
in the LSST camera.

The value is from https://lsstcam.lsst.io/index.html
"""

_lsstcam_gain = 1.595
"""The gain for the LSST camera in electrons per ADU.
This is the average value from the two CCD types used (e2v=1.51 and ITL=1.68)
in the LSST camera.

The value is from https://lsstcam.lsst.io/index.html
"""

_lsstcam_view_radius = 1.75
"""The angular radius of the observation field (in degrees). LSSTCam field of view is 9.6 square degree """

_lsstcam_ccd_radius = 0.1574
"""The approximate angular radius of a single LSST CCD (in degrees). Each CCD is 800*800 arcsec^2.
We approximate the radius as 800 arcsec/ sqrt(2). We overestimate slightly, because this value is
used in range searches. More exact filtering is done with the detector footprint.
"""

_lsst_zp_err_mag = 1.0e-4
"""The zero point error in magnitude.

We choose a very conservative noise flooring of 1e-4 mag.
This number will be updated when we have a better estimate from LSST.
"""


class LSSTObsTable(ObsTable):
    """An ObsTable for observations from the Rubin Observatory data releases.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the OpSim information.
    colmap : dict
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
        Defaults to the Rubin column names (OpSim, DP1, etc.), stored in _default_colnames.
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, OpSim-specific defaults will be
        used.
    **kwargs : dict
        Additional keyword arguments to pass to the constructor. This includes overrides
        for survey parameters such as:
        - dark_current : The dark current for the camera in electrons per second per pixel.
        - gain: The gain for the camera in electrons per ADU.
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The readout noise for the camera in electrons per pixel.
    """

    _required_names = ["ra", "dec", "time"]

    # Column names for the Rubin CCDVisit table from schemas including
    #   * DP1 (https://sdm-schemas.lsst.io/dp1.html#CcdVisit)
    #   * DP2+ (https://sdm-schemas.lsst.io/lsstcam.html#CcdVisit)
    _ccdvisit_colmap = {
        "dec": "dec",  # degrees
        "exptime": "expTime",  # seconds
        "filter": "band",
        "maglim": "magLim",  # magnitudes
        "pixel_scale": "pixelScale",  # arcseconds per pixel
        "ra": "ra",  # degrees
        "rotation": "skyRotation",  # degrees
        "seeing": "seeing",  # arcseconds
        "sky_bg_adu": "skyBg",  # Averge sky background in ADU
        "time": ["expMidptMJD", "obsStartMJD"],  # days
        "zp_mag_adu": "zeroPoint",  # magnitudes to produce 1 count (ADU)
    }

    # Column names for the Rubin Science Validation visit data release, which has
    # a different schema than the DP1 and DP2+ CCDVisit tables (mostly similar to OpSim):
    # https://survey-strategy.lsst.io/progress/sv_status/sv_20250930.html
    # See also:
    # https://github.com/lsst/tutorial-notebooks/blob/main/Commissioning/101_lsstcam_visits_database.ipynb
    _sv_visits_colmap = {
        # Values defined in the OpSim schema:
        # https://rubin-scheduler.lsst.io/fbs-output-schema.html
        "dec": "fieldDec",  # degrees
        "exptime": "visitExposureTime",  # seconds
        "filter": "band",
        "maglim": "fiveSigmaDepth",  # magnitudes
        "ra": "fieldRA",  # degrees
        "rotation": "rotSkyPos",  # degrees
        "seeing": "seeingFwhmEff",  # arcseconds
        "time": "exp_midpt_mjd",  # days
        # Some of the values are defined in the ConsDB schema:
        # https://sdm-schemas.lsst.io/cdb_lsstcam.html#exposure
        "sky_bg_e": "sky_bg_median",  # Averge sky background in electrons per pixel
        "zp_mag_adu": "zero_point_median",  # magnitudes to produce 1 count (ADU)
    }

    # For now use the CCDVisit column mapping as the default.
    _default_colnames = _ccdvisit_colmap

    # Default survey values (LSSTCam).
    _default_survey_values = {
        "ccd_pixel_width": 4000,
        "ccd_pixel_height": 4000,
        "dark_current": _lsstcam_dark_current,
        "gain": _lsstcam_gain,
        "pixel_scale": LSSTCAM_PIXEL_SCALE,
        "radius": _lsstcam_view_radius,
        "read_noise": _lsstcam_readout_noise,
        "zp_err_mag": _lsst_zp_err_mag,
        "survey_name": "LSST",
    }

    # Default LSST saturation thresholds in magnitudes.
    # https://www.lsst.org/sites/default/files/docs/sciencebook/SB_3.pdf
    _default_saturation_mags = {
        "u": 14.7,
        "g": 15.7,
        "r": 15.8,
        "i": 15.8,
        "z": 15.3,
        "y": 13.9,
    }

    # Class constants for the column names.
    def __init__(
        self,
        table,
        colmap=None,
        saturation_mags=None,
        **kwargs,
    ):
        colmap = self._default_colnames if colmap is None else colmap

        # If saturation thresholds are not provided, then set to the
        # LSSTObsTable defaults.
        if saturation_mags is None:
            saturation_mags = self._default_saturation_mags

        super().__init__(table, colmap=colmap, saturation_mags=saturation_mags, **kwargs)

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy (which produces 1 e-) to the LSSTObsTable tables."""
        cols = self._table.columns.to_list()

        if "zp" in cols:
            return  # Nothing to do

        # If the zero point column is already present (as a magnitude),
        # we convert it to nJy.
        if "zp_mag_adu" in cols:
            zp_values = mag2flux(self._table["zp_mag_adu"]) * self.safe_get_survey_value("gain")
            self.add_column("zp", zp_values, overwrite=True)
            return

        raise ValueError("Not enough information to compute the zero points.")

    @classmethod
    def from_ccdvisit_table(cls, table, make_detector_footprint=False, **kwargs):
        """Construct an LSSTObsTable object from a CCDVisit table.

        As an example we could access the DP1 CCDVisit table from RSP as:
            from lsst.rsp import get_tap_service
            service = get_tap_service("tap")
            table = service.search("SELECT * FROM dp1.CcdVisit").to_table().to_pandas()

        Or you can read a table from a file (e.g. using the `read_sqlite_table` function).
            from lightcurvelynx.utils.io_utils import read_sqlite_table
            table = read_sqlite_table("path_to_file.db", sql_query="SELECT * FROM observations")

        Parameters
        ----------
        table : pandas.core.frame.DataFrame
            The CCDVisit table containing the LSSTObsTable data.
        make_detector_footprint : bool, optional
            If True, the detector footprint will be created based on the xSize and ySize columns
            in the table.
        **kwargs : dict
            Additional keyword arguments to pass to the LSSTObsTable constructor.

        Returns
        -------
        obstable : LSSTObsTable
            An LSSTObsTable object containing the data from the CCDVisit table.
        """
        table = table.copy()
        cols = table.columns.to_list()

        # Try to derive the viewing radius if we have the information to do so.
        if "xSize" in cols and "ySize" in cols and "pixel_scale" in cols:
            radius_px = np.sqrt((table["xSize"] / 2) ** 2 + (table["ySize"] / 2) ** 2)
            table["radius"] = (radius_px * table["pixel_scale"]) / 3600.0  # arcsec to degrees
        elif "radius" not in kwargs:
            # Use a single approximate average ccd radius.
            kwargs["radius"] = _lsstcam_ccd_radius

        # Create the ObsTable object. Use the default column mapping for LSST, which
        # supports the DP1 and DP2+ CCDVisit Table schemas.
        obstable = cls(table, **kwargs)

        # Create a detector footprint if requested. We use the same (average) footprint for
        #  all CCDs based on the survey parameters for pixel scale and CCD size.
        if make_detector_footprint:
            pixel_scale = obstable.survey_values.get("pixel_scale")
            width_px = obstable.survey_values.get("ccd_pixel_width")
            height_px = obstable.survey_values.get("ccd_pixel_height")
            detect_fp = DetectorFootprint.from_pixel_rect(width_px, height_px, pixel_scale=pixel_scale)
            obstable.set_detector_footprint(detect_fp)

        return obstable

    @classmethod
    def from_sv_visits_table(cls, table, **kwargs):
        """Construct an LSSTObsTable object from a science validation visits table.
        https://survey-strategy.lsst.io/progress/sv_status/sv_20250930.html
        Note this table uses a combination of the schemas (e.g. OpSim and ConsDB).

        As an example we can read a table from a file (e.g. using the `read_sqlite_table` function).
            from lightcurvelynx.utils.io_utils import read_sqlite_table
            table = read_sqlite_table("path_to_file.db", sql_query="SELECT * FROM observations")

        Parameters
        ----------
        table : pandas.core.frame.DataFrame
            The science validation visits table containing the LSSTObsTable data.
        **kwargs : dict
            Additional keyword arguments to pass to the LSSTObsTable constructor.

        Returns
        -------
        obstable : LSSTObsTable
            An LSSTObsTable object containing the data from the science validation visits table.
        """
        # Remove the rows with NaNs in sky_bg_median or zero_point_median.
        table = table.dropna(subset=["sky_bg_median", "zero_point_median"])

        # Set the radius as the view (not CCD) radius because we do not have
        # per-CCD information.
        if "radius" not in kwargs:
            kwargs["radius"] = _lsstcam_view_radius

        # Create the ObsTable object using the science validation visits column mapping
        # (if a custom column mapping is not provided).
        colmap = kwargs.pop("colmap", cls._sv_visits_colmap)
        obstable = cls(table, colmap=colmap, **kwargs)
        return obstable

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
        observations = self._table.iloc[index]

        # By the effective FWHM definition, see
        # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
        # We need it in pixel^2
        pixel_scale = self.safe_get_survey_value("pixel_scale")
        psf_footprint = GAUSS_EFF_AREA2FWHM_SQ * (observations["seeing"] / pixel_scale) ** 2
        zp = observations["zp"]

        # Compute sky background in e- from sky_bg_adu.
        if "sky_bg_e" in observations:
            sky = observations["sky_bg_e"]
        elif "sky_bg_adu" in observations:
            sky = observations["sky_bg_adu"] * self.safe_get_survey_value("gain")
        else:
            raise ValueError(
                "Some value of sky background (electrons or ADU) is required to compute "
                "the sky background noise."
            )

        return poisson_bandflux_std(
            bandflux,
            total_exposure_time=observations["exptime"],
            exposure_count=1,
            psf_footprint=psf_footprint,
            sky=sky,
            zp=zp,
            readout_noise=self.safe_get_survey_value("read_noise"),
            dark_current=self.safe_get_survey_value("dark_current"),
            zp_err_mag=self.safe_get_survey_value("zp_err_mag"),
        )
