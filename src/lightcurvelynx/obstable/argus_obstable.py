"""The ArgusObsTable stores observation information from the Argus survey."""

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from mocpy import MOC

from lightcurvelynx.obstable.obs_table import ObsTable

_argus_view_radius = 52.0
"""The angular radius of the observation field (in degrees):
https://argus.unc.edu/about
"""

_argus_pixel_scale = 1.0
"""The pixel scale for the Argus survey in arcseconds per pixel:
https://argus.unc.edu/specifications
"""


class ArgusHealpixObsTable(ObsTable):
    """An ObsTable for observations from the Argus survey in healpix format.

    Unlike other ObsTable classes, the ArgusHealpixObsTable does not consist of a table of
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

    def __init__(
        self,
        table,
        *,
        colmap=None,
        apply_saturation=True,
        saturation_mags=None,
        nside=None,
        **kwargs,
    ):
        # Set some default values.
        self._spatial_data = None
        self._nside = nside
        self._depth = None

        # Check the unsupported terms in the kwargs and raise an error if they are provided.
        if "detector_footprint" in kwargs or "wcs" in kwargs:
            raise ValueError("ArgusObsTable does not support detector footprints.")

        super().__init__(
            table=table,
            colmap=colmap,
            apply_saturation=apply_saturation,
            saturation_mags=saturation_mags,
            **kwargs,
        )

    def uses_footprint(self):
        """Return whether the ObsTable uses a detector footprint for filtering."""
        return False

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

    def _build_spatial_data(self):
        """Construct a mapping of healpix id to row number from the ObsTable."""
        if self._nside is None:
            if "nside" in self._table.colnames:
                self._nside = self._table["nside"][0]
            else:
                raise ValueError(
                    "nside must be provided for ArgusHealpixObsTable construction or "
                    "as a column in the table."
                )

        # Check all nside values are the same.
        if "nside" in self._table.colnames:
            nside = self._table["nside"].to_numpy()
            if not np.all(nside == self._nside):
                raise ValueError(
                    "Inconsistent nside values found in the table. Expected all nside values"
                    f"to be {self._nside}, but found values: {np.unique(nside)}"
                )

        # Compute the depth.
        self._depth = int(np.log2(self._nside))

        # Create a healpix mapper for the given nside to convert between healpix ids and coordinates.
        self._healpix_mapper = HEALPix(nside=self._nside, order="nested", frame="icrs")

        self._spatial_data = {}
        if "healpix" in self._table.colnames:
            index = self._table["healpix"].to_numpy()
        elif "healpix_id" in self._table.colnames:
            index = self._table["healpix_id"].to_numpy()
        elif self._table.index.name == "healpix_id" or self._table.index.name == "healpix":
            index = self._table.index.to_numpy()
        else:
            raise ValueError(
                "No healpix id column found in the table. Expected one of 'healpix', "
                "'healpix_id', or index named 'healpix_id' or 'healpix'."
            )

        # Build a mapping of healpix id to row number.
        for idx in np.unique(index):
            self._spatial_data[idx] = np.where(index == idx)[0]

    def build_moc(self, max_depth=None, **kwargs):
        """Build a Multi-Order Coverage Map from the regions in the data set.

        These are built directly from the healpix pixels.

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth of the MOC. Default is the depth of the healpix pixels
            in the table.
        **kwargs : dict
            Additional keyword arguments to pass to the MOC construction. Not currently used,
            but accepted for consistency with the ObsTable interface.

        Returns
        -------
        MOC
            The Multi-Order Coverage Map constructed from the data set.
        """
        if max_depth is None:
            max_depth = self._depth

        logger = logging.getLogger(__name__)
        logger.debug(f"Building MOC from ArgusHealpixObsTable at depth={max_depth}.")
        moc = MOC.from_healpix_cells(
            healpix_cells=np.array(list(self._spatial_data.keys())),
            depth=self._depth,
            max_depth=max_depth,
        )
        return moc

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

    def range_search(self, query_ra, query_dec, *, radius=None, t_min=None, t_max=None):
        """Return the indices of the pointings that fall within the field
        of view of the query point(s).

        Note that radius is not used since everything is already stored at the healpix level,
        but it is accepted as a parameter for consistency with the ObsTable interface.

        Parameters
        ----------
        query_ra : float or numpy.ndarray
            The query right ascension (in degrees).
        query_dec : float or numpy.ndarray
            The query declination (in degrees).
        radius : float or None, optional
            Not used in this implementation since the data is already organized at the healpix level.
        t_min : float, numpy.ndarray or None, optional
            The minimum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.
        t_max : float, numpy.ndarray or None, optional
            The maximum time (in MJD) for the observations to consider.
            If None, no time filtering is applied.

        Returns
        -------
        inds : list[int] or list[numpy.ndarray]
            Depending on the input, this is either a list of indices for a single query point
            or a list of arrays (of indices) for an array of query points.
        """
        # If the query RA and Dec are scalars, convert them to 1D arrays for consistent processing.
        is_scalar = np.isscalar(query_ra) and np.isscalar(query_dec)
        query_ra = np.atleast_1d(query_ra)
        query_dec = np.atleast_1d(query_dec)
        if len(query_ra) != len(query_dec):
            raise ValueError("Query RA and Dec must have the same length.")

        # Bulk compute the healpix ids for all query points.
        coords = SkyCoord(query_ra * u.deg, query_dec * u.deg, frame="icrs")
        healpix = self._healpix_mapper.skycoord_to_healpix(coords)

        # For each query point, get the rows and apply time filtering if specified.
        inds = []
        for hp in healpix:
            if hp in self._spatial_data:
                row_inds = self._spatial_data[hp]

                # Apply time filtering if specified.
                if t_min is not None or t_max is not None:
                    times = self._table["time"][row_inds].to_numpy()
                    if t_min is not None:
                        row_inds = row_inds[times >= t_min]
                    if t_max is not None:
                        row_inds = row_inds[times <= t_max]

                inds.append(row_inds)
            else:
                inds.append(np.array([], dtype=int))

        # If the input was a single query point, return a single array of indices instead of a list of arrays.
        if is_scalar:
            inds = inds[0]
        return inds
