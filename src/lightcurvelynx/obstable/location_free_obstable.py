"""The LocationFreeObsTable stores observation information from the entire sky."""

import logging

import numpy as np
import pandas as pd
from mocpy import MOC

from lightcurvelynx.obstable.obs_table import ObsTable


class LocationFreeObsTable(ObsTable):
    """An ObsTable for observations from the entire sky (no location information).

    This is used when you want to simulate observations on a regular cadence
    regardless of where they are in the sky, such as when comparing them to
    a survey that has a regular cadence (e.g. ZTF, ATLAS, etc.).

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the ObsTable information.  Must have columns
        "time" and "filter".
    colmap : dict, optional
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
    **kwargs : dict
        Additional keyword arguments to pass to the ObsTable constructor. This includes overrides
        for survey parameters such as:

        - survey_name: The name of the survey (default="FAKE_SURVEY").
        - dark_electrons : The dark current for the camera in electrons per second per pixel.
        - gain: The gain for the camera in electrons per ADU.
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The readout noise for the camera in electrons per pixel.
    """

    # Default column mapping and survey values (no noise by default).
    _required_columns = ["time", "filter"]
    _default_colnames = {}
    _default_survey_values = {
        "nexposure": 1,
        "radius": 180.0,  # degrees
        "survey_name": "location_free",
    }

    def __init__(self, table, *, colmap=None, **kwargs):
        # If the input is a dictionary, convert it to a DataFrame.
        if isinstance(table, dict):
            table = pd.DataFrame(table)

        # Use the default column mapping if one is not provided.
        if colmap is None:
            colmap = self._default_colnames

        # Check the unsupported terms in the kwargs and raise an error if they are provided.
        if kwargs.get("detector_footprint") is not None or kwargs.get("wcs") is not None:  # pragma: no cover
            raise ValueError("LocationFreeObsTable does not support detector footprints.")

        super().__init__(table=table, colmap=colmap, **kwargs)

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
        raise NotImplementedError("LocationFreeObsTable does not support detector footprints.")

    def _build_spatial_data(self):
        """Build the spatial data for the LocationFreeObsTable. This is a no-op since the LocationFreeObsTable
        covers the entire sky.
        """
        pass

    def build_moc(self, max_depth=10, **kwargs):
        """Build a Multi-Order Coverage Map from the regions in the data set. This will
        always be the whole sky.

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of the MOC.
            Default is 10.
        **kwargs : dict
            Additional keyword arguments to pass to the MOC construction. Not currently used,
            but accepted for consistency with the ObsTable interface.

        Returns
        -------
        MOC
            The Multi-Order Coverage Map constructed from the data set.
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"Building MOC from LocationFreeObsTable at depth={max_depth}.")
        moc = MOC.from_healpix_cells(
            np.arange(12, dtype=np.uint64),
            depth=0,
            max_depth=max_depth,
        )
        return moc

    def _derive_noise_columns(self):
        """Derive any missing noise-related columns (e.g. zero points) from the existing columns
        and survey values.
        """
        pass

    def range_search(self, query_ra, query_dec, *, radius=None, t_min=None, t_max=None):
        """Return the indices of the pointings that fall within the time range (t_min, t_max).
        Since this is an all-sky table, the spatial filtering is never used.

        Parameters
        ----------
        query_ra : float or numpy.ndarray
            The query right ascension (in degrees). Only used to determine the number of query points,
            not for filtering.
        query_dec : float or numpy.ndarray
            The query declination (in degrees). Only used to determine the number of query points,
            not for filtering.
        radius : float or None, optional
            Not used in this implementation.
        t_min : float, numpy.ndarray or None, optional
            The minimum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.
        t_max : float, numpy.ndarray or None, optional
            The maximum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.

        Returns
        -------
        inds : numpy.ndarray or list[numpy.ndarray]
            If the input is a single query point, this is a 1D array of row indices.
            If the input is an array of query points, this is a list of index arrays (one per query).
        """
        if query_ra is None or query_dec is None:
            raise ValueError("Query RA and dec must be provided for range search, but got None.")

        # If the query RA and Dec are scalars, convert them to 1D arrays for consistent processing.
        is_scalar = np.isscalar(query_ra) and np.isscalar(query_dec)
        try:
            query_ra = np.atleast_1d(query_ra).astype(float)
            query_dec = np.atleast_1d(query_dec).astype(float)
        except ValueError as err:
            raise ValueError("Query RA and Dec must be convertible to float.") from err

        if query_ra.ndim != 1 or query_dec.ndim != 1:
            raise ValueError("Query RA and Dec must be 1-dimensional arrays.")
        if len(query_ra) != len(query_dec):
            raise ValueError("Query RA and Dec must have the same length.")
        if np.any(np.isnan(query_ra)) or np.any(np.isnan(query_dec)):
            raise ValueError("Query RA and Dec cannot contain NaN.")

        num_queries = len(query_ra)

        # If t_min is a scalar, convert it to a 1D array for consistent processing.
        if t_min is None:
            t_min = np.full(num_queries, -np.inf)
        elif np.isscalar(t_min):
            t_min = np.full(num_queries, t_min)
        elif len(t_min) != num_queries:
            raise ValueError("t_min must be a scalar or have the same length as query_ra and query_dec.")

        # If t_max is a scalar, convert it to a 1D array for consistent processing.
        if t_max is None:
            t_max = np.full(num_queries, np.inf)
        elif np.isscalar(t_max):
            t_max = np.full(num_queries, t_max)
        elif len(t_max) != num_queries:
            raise ValueError("t_max must be a scalar or have the same length as query_ra and query_dec.")

        # For each query point, apply time filtering if specified.
        all_rows = np.arange(len(self._table))
        all_times = self._table["time"].to_numpy()

        # Fast path when all queries share the same time bounds (common for scalar/None t_min/t_max).
        if num_queries == 0:
            inds = []
        elif np.all(t_min == t_min[0]) and np.all(t_max == t_max[0]):
            curr_rows = all_rows[(all_times >= t_min[0]) & (all_times <= t_max[0])]
            inds = [curr_rows] * num_queries
        else:
            inds = []
            for t_min_val, t_max_val in zip(t_min, t_max, strict=False):
                curr_rows = all_rows[(all_times >= t_min_val) & (all_times <= t_max_val)]
                inds.append(curr_rows)

        # If the input was a single query point, return a single array of indices instead of a list of arrays.
        if is_scalar:
            inds = inds[0]
        return inds
