"""The top-level module for survey related data, such as pointing and noise
information. By default the module uses the Rubin OpSim data, but it can be
extended to other survey data as well.
"""

from __future__ import annotations  # "type1 | type2" syntax in Python <3.10

import numpy as np
import pandas as pd

from lightcurvelynx import _LIGHTCURVELYNX_BASE_DATA_DIR
from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.astro_utils.noise_model import poisson_bandflux_std
from lightcurvelynx.astro_utils.zeropoint import flux_electron_zeropoint
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ
from lightcurvelynx.obstable.obs_table import ObsTable
from lightcurvelynx.utils.data_download import download_data_file_if_needed

LSSTCAM_PIXEL_SCALE = 0.2
"""The pixel scale for the LSST camera in arcseconds per pixel."""

_lsstcam_readout_noise = 8.8
"""The standard deviation of the count of readout electrons per pixel for the LSST camera.

The value is from https://smtn-002.lsst.io/v/OPSIM-1171/index.html
"""

_lsstcam_dark_current = 0.2
"""The dark current for the LSST camera in electrons per second per pixel.

The value is from https://smtn-002.lsst.io/v/OPSIM-1171/index.html
"""

_lsstcam_view_radius = 1.75
"""The angular radius of the observation field (in degrees)."""

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

_lsstcam_extinction_coeff = {
    "u": -0.458,
    "g": -0.208,
    "r": -0.122,
    "i": -0.074,
    "z": -0.057,
    "y": -0.095,
}
"""The extinction coefficients for the LSST filters.

Values are from
https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
Calculated with syseng_throughputs v1.9
"""

_lsstcam_zeropoint_per_sec_zenith = {
    "u": 26.524,
    "g": 28.508,
    "r": 28.361,
    "i": 28.171,
    "z": 27.782,
    "y": 26.818,
}
"""The zeropoints for the LSST filters at zenith

This is magnitude that produces 1 electron in a 1 second exposure,
see _assign_zero_points() docs for more details.

Values are from
https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
Calculated with syseng_throughputs v1.9
"""


class OpSim(ObsTable):
    """A wrapper class around the opsim table with cached data for efficiency.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the OpSim information.
    colmap : dict
        A mapping of short column names to their names in the underlying table.
        Defaults to the Rubin OpSim column names, stored in the class variable
        _opsim_colnames.
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, OpSim-specific defaults will be
        used.
    **kwargs : dict
        Additional keyword arguments to pass to the constructor. This includes overrides
        for survey parameters such as:
        - dark_current : The dark current for the camera in electrons per second per pixel.
        - ext_coeff: Mapping of filter names to extinction coefficients.
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The readout noise for the camera in electrons per pixel.
        - zp_per_sec: Mapping of filter names to zeropoints at zenith.
    """

    _required_names = ["ra", "dec", "time"]

    # Default column names for the Rubin OpSim.
    _default_colnames = {
        "airmass": "airmass",
        "dec": "fieldDec",
        "exptime": "visitExposureTime",
        "filter": "filter",
        "ra": "fieldRA",
        "time": "observationStartMJD",
        "zp": "zp_nJy",  # We add this column to the table
        "seeing": "seeingFwhmEff",
        "skybrightness": "skyBrightness",
        "nexposure": "numExposures",
    }

    # Default survey values.
    _default_survey_values = {
        "ccd_pixel_width": 4000,
        "ccd_pixel_height": 4000,
        "dark_current": _lsstcam_dark_current,
        "ext_coeff": _lsstcam_extinction_coeff,
        "pixel_scale": LSSTCAM_PIXEL_SCALE,
        "radius": _lsstcam_view_radius,
        "read_noise": _lsstcam_readout_noise,
        "zp_per_sec": _lsstcam_zeropoint_per_sec_zenith,
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

        # If saturation thresholds are not provided, then set to the OpSim defaults.
        if saturation_mags is None:
            saturation_mags = self._default_saturation_mags

        super().__init__(table, colmap=colmap, saturation_mags=saturation_mags, **kwargs)

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy to the OpSim tables."""
        cols = self._table.columns.to_list()
        if not ("filter" in cols and "airmass" in cols and "exptime" in cols):
            raise ValueError(
                "OpSim does not include the columns needed to derive zero point "
                "information. Required columns: filter, airmass, and exptime."
            )

        zp_values = flux_electron_zeropoint(
            ext_coeff=self.safe_get_survey_value("ext_coeff"),
            instr_zp_mag=self.safe_get_survey_value("zp_per_sec"),
            filter=self._table["filter"],
            airmass=self._table["airmass"],
            exptime=self._table["exptime"],
        )
        self.add_column("zp", zp_values, overwrite=True)

    @classmethod
    def from_url(cls, opsim_url, force_download=False):
        """Construct an OpSim object from a URL to a predefined opsim data file.

        For Rubin OpSim data, you will typically use the latest baseline data set in:
        https://s3df.slac.stanford.edu/data/rubin/sim-data/
        such as:
        https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs3.4/baseline/baseline_v3.4_10yrs.db

        Parameters
        ----------
        opsim_url : str
            The URL to the opsim data file.
        force_download : bool, optional
            If True, the OpSim data will be downloaded even if it already exists locally.
            Default is False.

        Returns
        -------
        opsim : OpSim
            An OpSim object containing the data from the specified URL.
        """
        data_file_name = opsim_url.split("/")[-1]
        data_path = _LIGHTCURVELYNX_BASE_DATA_DIR / "opsim" / data_file_name

        if not download_data_file_if_needed(data_path, opsim_url, force_download=force_download):
            raise RuntimeError(f"Failed to download opsim data from {opsim_url}.")
        return cls.from_db(data_path)

    @classmethod
    def from_ccdvisit_table(cls, table, make_detector_footprint=False, **kwargs):
        """Construct an OpSim object from a CCDVisit table.

        As an example we could access the DP1 CCDVisit table from RSP as:
            from lsst.rsp import get_tap_service
            service = get_tap_service("tap")
            table = service.search("SELECT * FROM dp1.CcdVisit").to_table().to_pandas()

        Parameters
        ----------
        table : pandas.core.frame.DataFrame
            The CCDVisit table containing the OpSim data.
        make_detector_footprint : bool, optional
            If True, the detector footprint will be created based on the xSize and ySize columns
            in the table.
        **kwargs : dict
            Additional keyword arguments to pass to the OpSim constructor.

        Returns
        -------
        opsim : OpSim
            An OpSim object containing the data from the CCDVisit table.
        """
        table = table.copy()

        # Bulk rename the columns to match the expected names. We also use this
        # dictionary to check that all of the expected columns as given by:
        # https://sdm-schemas.lsst.io/dp1.html#CcdVisit
        # are presents. We also annotate the expected units for each column.
        colmap = {
            "band": "filter",
            "ccdVisitId": "ccdVisitId",  # Integer ID
            "dec": "dec",  # Degrees
            "expMidptMJD": "time",  # MJD (days)
            "expTime": "exptime",  # Seconds
            "magLim": "fiveSigmaDepth",  # Magnitudes
            "ra": "ra",  # Degrees
            "seeing": "seeing",  # arcseconds
            "skyRotation": "rotation",  # Degrees
            "skyBg": "skybrightness",  # adu
            "skyNoise": "skynoise",  # adu
            "zeroPoint": "zp_mag",  # Magnitude (converted to flux below)
        }
        for c in colmap:
            if c not in table.columns:
                raise ValueError(f"Missing column '{c}' in the CCDVisit table.")
        table.rename(columns=colmap, inplace=True)
        cols = table.columns.to_list()

        # The CCDVisit table uses mag for zero point, we convert it to nJy.
        if "zp_mag" in cols:
            table["zp"] = mag2flux(table["zp_mag"])

        # Try to derive the viewing radius if we have the information to do so.
        if "xSize" in cols and "ySize" in cols and "pixel_scale" in cols:
            radius_px = np.sqrt((table["xSize"] / 2) ** 2 + (table["ySize"] / 2) ** 2)
            table["radius"] = (radius_px * table["pixel_scale"]) / 3600.0  # arcsec to degrees
        elif "radius" not in kwargs:
            # Use a single approximate average ccd radius.
            kwargs["radius"] = _lsstcam_ccd_radius

        # Create the OpSim object.
        opsim = cls(table, **kwargs)

        # Create a detector footprint if requested. We use the same (average) footprint for
        #  all CCDs based on the survey parameters for pixel scale and CCD size.
        if make_detector_footprint:
            pixel_scale = opsim.survey_values.get("pixel_scale")
            width_px = opsim.survey_values.get("ccd_pixel_width")
            height_px = opsim.survey_values.get("ccd_pixel_height")
            detect_fp = DetectorFootprint.from_pixel_rect(width_px, height_px, pixel_scale=pixel_scale)
            opsim.set_detector_footprint(detect_fp)

        return opsim

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
        observations = self._table.iloc[index]

        # By the effective FWHM definition, see
        # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
        # We need it in pixel^2
        pixel_scale = self.safe_get_survey_value("pixel_scale")
        psf_footprint = GAUSS_EFF_AREA2FWHM_SQ * (observations["seeing"] / pixel_scale) ** 2
        zp = observations["zp"]

        # Table value is in mag/arcsec^2
        sky_njy_angular = mag2flux(observations["skybrightness"])
        # We need electrons per pixel^2
        sky = sky_njy_angular * pixel_scale**2 / zp

        return poisson_bandflux_std(
            bandflux,
            total_exposure_time=observations["exptime"],
            exposure_count=observations["nexposure"],
            psf_footprint=psf_footprint,
            sky=sky,
            zp=zp,
            readout_noise=self.safe_get_survey_value("read_noise"),
            dark_current=self.safe_get_survey_value("dark_current"),
            zp_err_mag=self.safe_get_survey_value("zp_err_mag"),
        )


def create_random_opsim(num_obs, seed=None):
    """Create a random OpSim pointings drawn uniformly from (RA, dec).

    Parameters
    ----------
    num_obs : int
        The size of the OpSim to generate.
    seed : int
        The seed to used for random number generation. If None then
        uses a default random number generator.
        Default: None

    Returns
    -------
    opsim_data : OpSim
        The OpSim data structure.
    seed : int, optional
        The seed for the random number generator.
    """
    if num_obs <= 0:
        raise ValueError("Number of observations must be greater than zero.")

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)

    # Generate the (RA, dec) pairs uniformly on the surface of a sphere.
    ra = np.degrees(rng.uniform(0.0, 2.0 * np.pi, size=num_obs))
    dec = np.degrees(np.arccos(2.0 * rng.uniform(0.0, 1.0, size=num_obs) - 1.0) - (np.pi / 2.0))

    # Generate the information needed to compute zeropoint.
    airmass = rng.uniform(1.3, 1.7, size=num_obs)
    filter = rng.choice(["u", "g", "r", "i", "z", "y"], size=num_obs)

    input_data = {
        "observationStartMJD": 0.05 * np.arange(num_obs),
        "fieldRA": ra,
        "fieldDec": dec,
        "airmass": airmass,
        "filter": filter,
        "visitExposureTime": 29.0 * np.ones(num_obs),
    }

    opsim = OpSim(input_data)
    return opsim


def opsim_add_random_data(opsim_data, colname, min_val=0.0, max_val=1.0):
    """Add a column composed of random uniform data. Used for testing.

    Parameters
    ----------
    opsim_data : OpSim
        The OpSim data structure to modify.
    colname : str
        The name of the new column to add.
    min_val : float
        The minimum value of the uniform range.
        Default: 0.0
    max_val : float
        The maximum value of the uniform range.
        Default: 1.0
    """
    values = np.random.uniform(low=min_val, high=max_val, size=len(opsim_data))
    opsim_data.add_column(colname, values)


def oversample_opsim(
    opsim: OpSim,
    *,
    pointing: tuple[float, float] = (200, -50),
    search_radius: float = 1.75,
    delta_t: float = 0.01,
    time_range: tuple[float | None, float | None] = (None, None),
    bands: list[str] | None = None,
    strategy: str = "darkest_sky",
):
    """Single-pointing oversampled OpSim table.

    It includes observations for a single pointing only,
    but with very high time resolution. The observations
    would alternate between the bands.

    Parameters
    ----------
    opsim : OpSim
        The OpSim table to oversample.
    pointing : tuple of RA and Dec in degrees
        The pointing to use for the oversampled table.
    search_radius : float, optional
        The search radius for the oversampled table in degrees.
        The default is the half of the LSST's field of view.
    delta_t : float, optional
        The time between observations in days.
    time_range : tuple or floats or Nones, optional
        The start and end times of the observations in MJD.
        None means to use the minimum (maximum) time in
        all the observations found for the given pointing.
        Time is being samples as np.arange(*time_range, delta_t).
    bands : list of str or None, optional
        The list of bands to include in the oversampled table.
        The default is to include all bands found for the given pointing.
    strategy : str, optional
        The strategy to select prototype observations.
        - "darkest_sky" selects the observations with the minimal sky brightness
          (maximum "skyBrightness" value) in each band. This is the default.
        - "random" selects the observations randomly. Fixed seed is used.

    """
    ra, dec = pointing
    observations = opsim._table.iloc[opsim.range_search(ra, dec, radius=search_radius)]
    if len(observations) == 0:
        raise ValueError("No observations found for the given pointing.")

    time_min, time_max = time_range
    if time_min is None:
        time_min = np.min(observations["time"])
    if time_max is None:
        time_max = np.max(observations["time"])
    if time_min >= time_max:
        raise ValueError(f"Invalid time_range: start > end: {time_min} > {time_max}")

    uniq_bands = np.unique(observations["filter"])
    if bands is None:
        bands = uniq_bands
    elif not set(bands).issubset(uniq_bands):
        raise ValueError(f"Invalid bands: {bands}")

    new_times = np.arange(time_min, time_max, delta_t)
    n = len(new_times)
    if n < len(bands):
        raise ValueError("Not enough time points to cover all bands.")

    new_table = pd.DataFrame(
        {
            # Just in case, to not have confusion with the original table
            "observationId": opsim._table["observationId"].max() + 1 + np.arange(n),
            "time": new_times,
            "ra": ra,
            "dec": dec,
            "filter": np.tile(bands, n // len(bands)),
        }
    )
    other_columns = [column for column in observations.columns if column not in new_table.columns]

    if strategy == "darkest_sky":
        for band in bands:
            # MAXimum magnitude is MINimum brightness (darkest sky)
            idxmax = observations["skybrightness"][observations["filter"] == band].idxmax()
            idx = new_table.index[new_table["filter"] == band]
            darkest_sky_obs = pd.DataFrame.from_records([observations.loc[idxmax]] * idx.size, index=idx)
            new_table.loc[idx, other_columns] = darkest_sky_obs[other_columns]
    elif strategy == "random":
        rng = np.random.default_rng(0)
        for band in bands:
            single_band_obs = observations[observations["filter"] == band]
            idx = new_table.index[new_table["filter"] == band]
            random_obs = single_band_obs.sample(idx.size, replace=True, random_state=rng).set_index(idx)
            new_table.loc[idx, other_columns] = random_obs[other_columns]
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return OpSim(
        new_table,
        colmap=opsim._colmap,
        **opsim.survey_values,
    )
