import numpy as np
import pandas as pd
from astropy.table import Table
from citation_compass import cite_function

from lightcurvelynx import _LIGHTCURVELYNX_BASE_DATA_DIR
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.astro_utils.noise_model import poisson_bandflux_std
from lightcurvelynx.obstable.obs_table import ObsTable

ROMAN_PIXEL_SCALE = 0.11
"""The pixel scale for Roman in arcseconds per pixel.
https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/introduction-to-the-wfi/wfi-quick-reference
"""

_roman_dark_current = 0.0
"""The dark current for Roman in electrons per second per pixel.
< 0.005 e-/s/pix
https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/wfi-detectors
"""

_roman_fov = 0.28
"""The Roman field of view in square degree
https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/introduction-to-the-wfi/wfi-quick-reference
"""

_roman_zodi_level_factor = 1.5
"""
Factor for calculating Zodiacal light intensity.

According to the WFI technical documentation:
https://roman.gsfc.nasa.gov/science/WFI_technical.html

- At high galactic latitudes, the Zodi intensity is typically ~1.5× the
  minimum.
- For observation into the galactic bulge, the Zodi intensity is typically 2.5-7x the minimum.
"""


_psf_url = "https://raw.githubusercontent.com/RomanSpaceTelescope/roman-technical-information/refs/heads/main/data/WideFieldInstrument/Imaging/PointSpreadFunctions/SummaryPSFstats_center.ecsv"
_zp_url = "https://raw.githubusercontent.com/RomanSpaceTelescope/roman-technical-information/refs/heads/main/data/WideFieldInstrument/Imaging/ZeroPoints/Roman_zeropoints_20240301.ecsv"
_thermal_url = "https://raw.githubusercontent.com/RomanSpaceTelescope/roman-technical-information/refs/heads/main/data/WideFieldInstrument/Imaging/Backgrounds/internal_thermal_backgrounds.ecsv"
_zodiacal_url = "https://raw.githubusercontent.com/RomanSpaceTelescope/roman-technical-information/refs/heads/main/data/WideFieldInstrument/Imaging/ZodiacalLight/zodiacal_light.ecsv"


def _get_roman_char():
    psf_table = Table.read(_psf_url, format="csv", comment="#").to_pandas()
    zp_table = Table.read(_zp_url, format="csv", comment="#", delimiter=" ").to_pandas()
    thermal_table = Table.read(_thermal_url, format="csv", comment="#").to_pandas()
    zodiacal_table = Table.read(_zodiacal_url, format="csv", comment="#").to_pandas()

    return {
        "psf_table": psf_table,
        "zp_table": zp_table,
        "thermal_table": thermal_table,
        "zodiacal_min_table": zodiacal_table,
    }


def _get_ma_table(ma_table_path=None):
    if ma_table_path is None:
        ma_table_file = (
            _LIGHTCURVELYNX_BASE_DATA_DIR
            / "roman_characterization/roman_wfi_imaging_multiaccum_tables_with_exptime.csv"
        )
    ma_table = pd.read_csv(ma_table_file)

    return ma_table


class RomanObsTable(ObsTable):
    """A subclass for Roman exposure table.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the observation information.
    colmap : dict
        A mapping of short column names to their names in the underlying table.
        Defaults to the Roman APT column names, stored in _default_colnames.
    ma_table_path : str or pathlib.Path, optional
        The path to the Roman MultiAccum table CSV file. If not provided, a default table included
        with LightCurveLynx will be used.
    saturation_mags : dict, optional
        A dictionary mapping filter names to their saturation thresholds in magnitudes. The filters
        provided must match those in the table. If not provided, saturation effects will not be applied.
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
    _default_colnames = {"dec": "DEC", "filter": "BANDPASS", "ra": "RA", "zp": "zp_nJy", "time": "time"}

    # Default survey values.
    _default_survey_values = {
        "dark_current": _roman_dark_current,
        "pixel_scale": ROMAN_PIXEL_SCALE,
        "survey_name": "Roman",
        "radius": np.sqrt(_roman_fov / np.pi),
        "zodi_level": _roman_zodi_level_factor,
    }

    def __init__(self, table, colmap=None, ma_table_path=None, saturation_mags=None, **kwargs):
        colmap = self._default_colnames if colmap is None else colmap

        self.apt_table = table
        self.ma_table = _get_ma_table(ma_table_path)
        roman_char = _get_roman_char()
        self.zp_table = roman_char["zp_table"]
        self.psf_table = roman_char["psf_table"]
        self.thermal_table = roman_char["thermal_table"]
        self.zodiacal_table = roman_char["zodiacal_min_table"]
        self._append_apt_table()

        super().__init__(self.apt_table, colmap=colmap, saturation_mags=saturation_mags, **kwargs)

    def _append_apt_table(self):
        self.apt_table["zp_abmag"] = 0.0
        self.apt_table["N_Eff_Pix"] = 0.0
        self.apt_table["zodi_countrate_min"] = 0.0
        self.apt_table["thermal_countrate"] = 0.0
        self.apt_table["time"] = 0.0
        self.apt_table["exptime"] = 0.0

        for f in np.unique(self.apt_table.BANDPASS):
            zp_abmag = self.zp_table.loc[self.zp_table.element == f, "ABMag"].values[0]
            self.apt_table.loc[f == self.apt_table.BANDPASS, "zp_abmag"] = zp_abmag
            n_eff_pix = self.psf_table.loc[self.psf_table["Filter"] == f, "N_Eff_Pix"].values[0]
            self.apt_table.loc[f == self.apt_table.BANDPASS, "N_Eff_Pix"] = n_eff_pix
            sigma_zodi_min = self.zodiacal_table.loc[self.zodiacal_table["filter"] == f, "rate"].values[0]
            self.apt_table.loc[f == self.apt_table.BANDPASS, "zodi_countrate_min"] = sigma_zodi_min
            sigma_thermal = self.thermal_table.loc[self.thermal_table["filter"] == f, "rate"].values[0]
            self.apt_table.loc[f == self.apt_table.BANDPASS, "thermal_countrate"] = sigma_thermal

        for mat_number in np.unique(self.apt_table.MA_TABLE_NUMBER):
            exptime = self.ma_table.loc[self.ma_table["MATableNumber"] == mat_number, "Exptime"].values[0]
            self.apt_table.loc[mat_number == self.apt_table.MA_TABLE_NUMBER, "exptime"] = exptime

        npass = len(np.unique(self.apt_table.PASS))
        self.start_time = 61304.0
        self.cadence = 5.0
        times = self.start_time + self.cadence * np.arange(npass)
        for i, pass_num in enumerate(np.unique(self.apt_table.PASS)):
            self.apt_table.loc[pass_num == self.apt_table.PASS, "time"] = times[i]

        self.apt_table["zp_nJy"] = mag2flux(self.apt_table["zp_abmag"])

    @cite_function
    def readnoise_func(self, exptime):
        """
        Readout noise function for Roman.

        References
        ----------
        Eq. 9 of Rose et al 2025 - https://ui.adsabs.harvard.edu/abs/2025ApJ...988...65R/abstract

        Parameters
        ----------
        exptime: float or npt.ArrayLike
            Exposure time in seconds.

        Returns
        -------
        sigma_read: float or npt.ArrayLike
            Readout noise.
        """
        exptime = np.asarray(exptime)
        sigma_floor_sq = 25.0
        sigma_sq = 3072.0
        n = exptime / 3.04
        nfactor = (n - 1) / n / (n + 1)
        sigma_read = np.sqrt(sigma_floor_sq + sigma_sq * nfactor)

        return sigma_read

    @cite_function
    def calculate_skynoise(self, exptime, zodi_scale, zodi_countrate_min, thermal_countrate):
        """Calculate sky noise.

        Reference
        ---------
        Eq. 10 of Rose et al 2025 - https://ui.adsabs.harvard.edu/abs/2025ApJ...988...65R/abstract

        Parameters
        ----------
        exptime: float or npt.ArrayLike
            Exposure time.
        zodi_scale: float or npt.ArrayLike
            Zodiacal light scale. The zodiacal light amount is zodi_scale * zodi_countrate_min
        zodi_countrate_min: float or npt.ArrayLike
            Minimum zodiacal count rate (e-/s/pixel).
        thermal_countrate: float or npt.ArrayLike
            Thermal count rate (e-/s/pixel).

        Returns
        -------
        sky_variance: float or npt.ArrayLike
            Total sky variance (e-^2/pixel).
        """

        exptime = np.asarray(exptime)
        zodi_scale = np.asarray(zodi_scale)
        zodi_countrate_min = np.asarray(zodi_countrate_min)
        thermal_countrate = np.asarray(thermal_countrate)
        sky_variance = exptime * ((zodi_scale * zodi_countrate_min) + thermal_countrate)

        return sky_variance

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        observations = self._table.iloc[index]
        observations["sky"] = self.calculate_skynoise(
            observations["exptime"],
            self.safe_get_survey_value("zodi_level"),
            observations["zodi_countrate_min"],
            observations["thermal_countrate"],
        )

        return poisson_bandflux_std(
            bandflux,  # nJy
            total_exposure_time=observations["exptime"],
            exposure_count=1,
            psf_footprint=observations["N_Eff_Pix"],
            sky=observations["sky"],
            zp=observations["zp"] / observations["exptime"],  # (nJy/s * s)^-1
            readout_noise=self.readnoise_func,  # e-/pixel
            dark_current=self.safe_get_survey_value("dark_current"),  # e-/second/pixel
        )
