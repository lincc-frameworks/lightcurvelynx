import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.astro_utils.zeropoint import calculate_zp_from_maglim, sky_bg_adu_to_electrons
from lightcurvelynx.consts import GAUSS_EFF_AREA2FWHM_SQ
from lightcurvelynx.noise_models.base_noise_models import PoissonFluxNoiseModel
from lightcurvelynx.noise_models.noise_utils import poisson_bandflux_std
from lightcurvelynx.obstable.obs_table import ObsTable

ZTFCAM_PIXEL_SCALE = 1.01
"""The pixel scale for the ZTF camera in arcseconds per pixel."""

_ztfcam_readout_noise = 8
"""The standard deviation of the count of readout electrons per pixel for the ZTF camera."""

_ztfcam_dark_current = 0.0
"""The dark current for the ZTF camera in electrons per second per pixel."""

_ztfcam_view_radius = 3.868
"""The angular radius of the observation field (in degrees). ZTF FOV is 47 deg^2 = pi * radius**2"""

_ztfcam_ccd_gain = 6.2
"""CCD gain (in e-/ADU)"""

_ztf_zp_error = 0.01
"""The zero point error in magnitude.
According to Masci et al. 2019, calibration error is around between 8 and 25 millimag.
https://ui.adsabs.harvard.edu/abs/2019PASP..131a8003M/abstract"""


class ZTFPoissonFluxNoiseModel(PoissonFluxNoiseModel):
    """A subclass of PoissonFluxNoiseModel for ZTF survey data."""

    def __init__(self):
        super().__init__()

    def compute_flux_error(self, bandflux, obs_table, indices):
        """Compute the flux error for the given bandflux and observation parameters.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in energy units, e.g. nJy.
        obs_table : ObsTable
            Table containing the observation parameters, including all
            parameters needed to compute the noise.
        indices : array_like of int
            Indices of the observations in the ObsTable for which to compute the noise.

        Returns
        -------
        flux_err : array_like
            The standard deviation of the bandflux measurement error, in the
            same units as the input bandflux.
        """
        # By the effective FWHM definition, see
        # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
        fwhm = obs_table["fwhm"].iloc[indices]
        footprint = GAUSS_EFF_AREA2FWHM_SQ * fwhm**2  # in pixels

        return poisson_bandflux_std(
            bandflux,  # nJy
            total_exposure_time=obs_table["exptime"].iloc[indices],
            exposure_count=1,
            psf_footprint=footprint,
            sky=obs_table["sky"].iloc[indices] * obs_table.safe_get_survey_value("gain"),  # e-/pixel^2
            zp=obs_table["zp"].iloc[indices],  # nJy
            readout_noise=obs_table.safe_get_survey_value("read_noise"),  # e-/pixel
            dark_current=obs_table.safe_get_survey_value("dark_current"),  # e-/second/pixel
            zp_err_mag=obs_table.safe_get_survey_value("zp_err_mag"),
        )


class ZTFObsTable(ObsTable):
    """A subclass for ZTF exposure table.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the observation information.
    colmap : dict
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
        Defaults to the ZTF column names, stored in _default_colnames.
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, ZTF-specific defaults will be used.
    noise_model : NoiseModel, optional
        The noise model to use for this ObsTable. If not provided, defaults to
        ZTFPoissonFluxNoiseModel.
    **kwargs : dict
        Additional keyword arguments to pass to the ObsTable constructor. This includes overrides
        for survey parameters such as:

        - dark_current : The dark current for the camera in electrons per second per pixel.
        - gain: The CCD gain (in e-/ADU).
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The standard deviation of the count of readout electrons per pixel.
    """

    # Default column names for the ZTF survey data.
    _default_colnames = {
        "maglim": "maglim",
        "sky": "scibckgnd",
        "fwhm": "fwhm",
        "dec": "dec",
        "exptime": "exptime",
        "filter": "filter",
        "ra": "ra",
        "time": "obsmjd",
        "zp": "zp_nJy",  # We add this column to the table
    }

    # Default survey values.
    _default_survey_values = {
        "dark_current": _ztfcam_dark_current,
        "gain": _ztfcam_ccd_gain,
        "pixel_scale": ZTFCAM_PIXEL_SCALE,
        "radius": _ztfcam_view_radius,
        "read_noise": _ztfcam_readout_noise,
        "zp_err_mag": _ztf_zp_error,
        "survey_name": "ZTF",
    }

    # Default saturation thresholds for ZTF, in magnitudes.
    # https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_extended_cautionary_notes.pdf
    # Using a naive value of 12.5 mag for the time being.
    _default_saturation_mags = {
        "g": 12.5,
        "r": 12.5,
        "i": 12.5,
    }

    def __init__(
        self,
        table,
        colmap=None,
        saturation_mags=None,
        noise_model=None,
        **kwargs,
    ):
        colmap = self._default_colnames if colmap is None else colmap

        # Make a copy of the table data with the obsdate converted to the MJD and
        # save in time.
        if "obsdate" in table and "obsmjd" not in table:
            table = table.copy()
            t = Time(list(table["obsdate"]), format="iso", scale="utc")
            table["obsmjd"] = t.mjd

        # If saturation thresholds are not provided, then set to the ZTF defaults.
        if saturation_mags is None:
            saturation_mags = self._default_saturation_mags

        # If noise model is not provided, then set to the ZTF default.
        if noise_model is None:
            noise_model = ZTFPoissonFluxNoiseModel()

        super().__init__(
            table,
            colmap=colmap,
            saturation_mags=saturation_mags,
            noise_model=noise_model,
            **kwargs,
        )

    def _assign_zero_points(self):
        """Assign instrumental zero points in ADU to the ObsTable."""
        cols = self._table.columns.tolist()
        if not ("maglim" in cols and "sky" in cols and "fwhm" in cols and "exptime" in cols):
            raise ValueError(
                "ObsTable does not include the columns needed to derive zero point "
                "information. Required columns: maglim, sky, fwhm and exptime."
            )

        # replace invalid values in table
        self._table = self._table.replace("", np.nan)
        self._table = self._table.dropna(subset=["fwhm"])

        # Compute the sky background in electrons/pixel. The sky column is in ADU/pixel,
        # so we need to multiply by the gain.
        sky_bg_electrons = sky_bg_adu_to_electrons(self._table["sky"], _ztfcam_ccd_gain)
        zp_values = calculate_zp_from_maglim(
            maglim=self._table["maglim"],
            sky_bg_electrons=sky_bg_electrons,
            fwhm_px=self._table["fwhm"],
            read_noise=_ztfcam_readout_noise,
            dark_current=_ztfcam_dark_current,
            exptime=self._table["exptime"],
            nexposure=1,
        )
        zp_nJy = mag2flux(zp_values)
        self.add_column("zp", zp_nJy, overwrite=True)

    @classmethod
    def from_db(cls, filename, sql_query="SELECT * from exposures", colmap=None):
        """Create an ObsTable object from the data in a db file.

        Parameters
        ----------
        filename : str
            The name of the db file.
        sql_query : str
            The SQL query to use when loading the table.
            Default: "SELECT * FROM observations"
        colmap : dict, optional
            A mapping of short column names to their names in the underlying table.
            If None then defaults to the ZTF column names.

        Returns
        -------
        obstable : ZTFObsTable
            A table with all of the pointing data.

        Raise
        -----
        FileNotFoundError
            if the file does not exist.
        ValueError
            if unable to load the table.
        """
        if colmap is None:
            colmap = cls._default_colnames

        if not Path(filename).is_file():
            raise FileNotFoundError(f"ObsTable file {filename} not found.")
        con = sqlite3.connect(f"file:{filename}?mode=ro", uri=True)

        # Read the table.
        try:
            obstable = pd.read_sql_query(sql_query, con)
        except Exception:
            raise ValueError("ObsTable database read failed.") from None

        # Close the connection.
        con.close()

        return ZTFObsTable(obstable, colmap=colmap)


def create_random_ztf_obs_data(num_obs, seed=None):
    """Create a random ObsTable pointings drawn uniformly from (RA, dec).

    Parameters
    ----------
    num_obs : int
        The size of the ObsTable to generate.
    seed : int
        The seed to used for random number generation. If None then
        uses a default random number generator.
        Default: None

    Returns
    -------
    obstable : pd.DataFrame
        The data for the ObsTable.
    """
    if num_obs <= 0:
        raise ValueError("Number of observations must be greater than zero.")

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)

    # Generate the (RA, dec) pairs uniformly on the surface of a sphere.
    ra = np.degrees(rng.uniform(0.0, 2.0 * np.pi, size=num_obs))
    dec = np.degrees(np.arccos(2.0 * rng.uniform(0.0, 1.0, size=num_obs) - 1.0) - (np.pi / 2.0))

    # Generate the information needed to compute zeropoint.
    maglim = rng.uniform(19.0, 21.0, size=num_obs)
    sky = rng.uniform(100.0, 200.0, size=num_obs)
    fwhm = rng.uniform(1.0, 3.0, size=num_obs)
    filter = rng.choice(["g", "r", "i"], size=num_obs)

    input_data = {
        "obsdate": ["2018-03-25 06:04:40.000"] * num_obs,
        "ra": ra,
        "dec": dec,
        "maglim": maglim,
        "scibckgnd": sky,
        "fwhm": fwhm,
        "filter": filter,
        "exptime": 30.0 * np.ones(num_obs),
    }
    return pd.DataFrame(input_data)
