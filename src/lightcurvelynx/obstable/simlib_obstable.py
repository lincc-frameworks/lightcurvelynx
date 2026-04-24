"""A class for storing and working with survey data in simlib format."""

from astropy.table import vstack
from citation_compass import CiteClass, cite_function

from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.obstable.obs_table import ObsTable


class SIMLIBObsTable(ObsTable, CiteClass):
    """An ObsTable for observations from the SkyMapper survey.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the survey information.
    radius : float
        The angular radius of the observations (in degrees). This is required.
    colmap : dict
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
        Defaults to the SIMLIB column names, stored in _default_colnames.
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
    SNANA: A Public Software Package for Supernova Analysis
    https://arxiv.org/abs/0908.4280
    """

    # Column names for the SIMLIB from: https://github.com/RickKessler/SNANA
    _default_colnames = {
        "dec": "DECL",  # degrees
        "gain": "CCD_GAIN",  # electrons/ADU
        "pixel_scale": "PIXSIZE",  # arcseconds/pixel
        "psf_sigma_1": "PSF1",  # pixels
        "psf_sigma_2": "PSF2",  # pixels
        "filter": "FLT",  # filter name
        "ra": "RA",  # degrees
        "read_noise": "CCD_NOISE",  # electrons/pixel
        "sky": "SKYSIG",  # skynoise in ADU per pixel
        "time": "MJD",  # days
        "zp_mag_adu": "ZPTAVG",  # magnitudes to produce 1 count (ADU)
    }

    # Class constants for the column names.
    def __init__(
        self,
        table,
        *,
        radius=None,
        colmap=None,
        **kwargs,
    ):
        colmap = self._default_colnames if colmap is None else colmap

        if radius is None or radius <= 0:
            raise ValueError("Radius must be provided and positive for SIMLIBObsTable.")

        super().__init__(
            table,
            colmap=colmap,
            radius=radius,
            **kwargs,
        )

    @staticmethod
    def expand_simlib_obset(obs_set):
        """Expand a simlib obs_set into a table with the metadata repeated for each observation.

        Parameters
        ----------
        obs_set : astropy.table.table.Table
            A table representing an obs_set from a simlib file. Each row corresponds to an observation,
            and the table contains the standard columns for a SIMLIB observation, such as "SEARCH", "MJD",
            "IDEXPT", etc. as well as the standard metadata.

        Returns
        -------
        astropy.table.table.Table
            A table containing one row per observation, with the metadata columns repeated
            for each observation.
        """
        expanded = obs_set.copy()
        for meta_col, meta_value in obs_set.meta.items():
            expanded[meta_col] = [meta_value] * len(expanded)

        # Clear the metadata since we have moved it to columns.
        expanded.meta.clear()

        return expanded

    @staticmethod
    def combine_simlib_obs_sets(obs_sets):
        """Combine multiple simlib OBB sets into a single astropy table.

        Parameters
        ----------
        obs_sets : iterable of astropy.table.table.Table
            The list of tables representing the obs_sets from a simlib file. Each
            table contains the standard columns for a SIMLIB observation, such as
            "SEARCH", "MJD", "IDEXPT", etc. as well as the standard metadata.

        Returns
        -------
        astropy.table.table.Table
            A single table containing all the observations from the input obs_sets.
        """
        if len(obs_sets) == 0:
            raise ValueError("No obs_sets provided to combine.")

        all_tables = [SIMLIBObsTable.expand_simlib_obset(obs_set) for obs_set in obs_sets.values()]
        combined_data = vstack(all_tables)
        return combined_data

    @classmethod
    @cite_function
    def from_simlib_file(cls, filename, **kwargs):
        """Create a SIMLIBObsTable from a simlib file.

        Parameters
        ----------
        filename : str or pathlib.Path
            The path to the simlib file to read.
        **kwargs : dict
            Additional keyword arguments to pass to the constructor. This includes overrides
            for survey parameters such as:

            - dark_current : The dark current for the camera in electrons per second per pixel.
            - gain: The gain for the camera in electrons per ADU.
            - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
            - radius: The angular radius of the observations (in degrees).

        References
        ----------
        sncosmo - https://zenodo.org/records/14714968
        """
        try:
            from sncosmo import read_snana_simlib
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "sncosmo package is not installed be default. You can install it with "
                "sncosmo package is not installed by default. You can install it with "
                "`pip install sncosmo` or `conda install conda-forge::sncosmo`."
            ) from err

        _, obs_sets = read_snana_simlib(filename)
        combined_data = cls.combine_simlib_obs_sets(obs_sets)
        return cls(combined_data, **kwargs)

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy (which produces 1 e-) to the SkyMapperObsTable tables."""
        cols = set(self._table.columns.to_list())

        if "zp" in cols:
            return  # Nothing to do

        # If the zero point column is already present (as a magnitude),
        # we convert it to nJy (per electron).
        if "zp_mag_adu" in cols:
            if "gain" in cols:
                gain = self._table["gain"]
            elif "gain" in self.survey_values:
                gain = self.safe_get_survey_value("gain")
            else:
                raise ValueError("Cannot compute zero points from zp_mag_adu without gain information.")

            zp_values = mag2flux(self._table["zp_mag_adu"]) / gain
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
        raise NotImplementedError("bandflux_error_point_source is not yet implemented for SIMLIBObsTable.")
