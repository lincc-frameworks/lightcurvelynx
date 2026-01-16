"""A wrapper for querying dust maps and returning E(B-V) values.

This module supports a variety of external dustmaps libaries, but
was primarily designed to work with the dustmaps package:
https://github.com/gregreen/dustmaps

Note that the dustmaps package is not included by default, because it uses
dependencies that may not be compatible with all systems. Users can manually
install dustmaps in their environment with “pip install dustmaps” or
“conda install conda-forge::dustmaps”.
"""

import importlib
import logging
from pathlib import Path

import numpy as np
import pooch
from astropy.coordinates import SkyCoord
from citation_compass import CiteClass

from lightcurvelynx import _LIGHTCURVELYNX_BASE_DATA_DIR
from lightcurvelynx.base_models import FunctionNode


class DustEBV(FunctionNode):
    """A wrapper that queries a dust map and returns the ebv for each location.

    This wrapper is designed to work with multiple dust map implementations
    by providing a standard interface.

    Attributes
    ----------
    query_fn : function
        The function to query the dust map ebv value given (RA, dec).

    Parameters
    ----------
    query_fn : function
        The function to query the dust map ebv value given (RA, dec).
    RA : parameter
        The object's right ascension (in degrees).
    dec : parameter
        The object's declination (in degrees).
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, query_fn, ra=None, dec=None, **kwargs):
        # RA and dec are required, but must be passed through as keyword
        # arguments to be used in the sampling function (hence the None default).
        if ra is None or dec is None:
            raise ValueError("RA and dec must be provided to query the dust map.")
        super().__init__(query_fn, ra=ra, dec=dec, outputs=["ebv"], **kwargs)


class DustmapWrapper(DustEBV, CiteClass):
    """A convenience wrapper for the dustmap package
    https://github.com/gregreen/dustmaps

    The DustmapWrapper is designed to take dust map objects from the dustmaps package,
    but can be used with any object that has a query(coords: SkyCoord) method.

    Parameters
    ----------
    dust_map : DustMap object
        The dust map. Since different dustmap's query function may produce different
        outputs, you should include the corresponding ebv_func to transform the result
        into ebv if needed.
    ebv_func : function, optional
        A function to translate the result of the dustmap query into an ebv. This is
        not needed for dustmaps that already produce ebv values.
    RA : parameter
        The object's right ascension (in degrees).
    dec : parameter
        The object's declination (in degrees).
    **kwargs : `dict`, optional
        Any additional keyword arguments.

    References
    ----------
    Green 2018, JOSS, 3(26), 695.
    https://github.com/gregreen/dustmaps
    """

    def __init__(self, dust_map, ra=None, dec=None, ebv_func=None, **kwargs):
        # RA and dec setters are passed through to the DustEBV init method.
        super().__init__(self.compute_ebv, ra=ra, dec=dec, **kwargs)
        self._ebv_func = ebv_func
        self._dust_map = dust_map

    def compute_ebv(self, ra, dec):
        """Compute the E(B-V) value for a given location.

        Parameters
        ----------
        ra : float or np.array
            The object's right ascension (in degrees).
        dec : float or np.array
            The object's declination (in degrees).

        Returns
        -------
        ebv : float or np.array
            The E(B-V) value or array of values.
        """
        if self._dust_map is None:
            raise ValueError("A dust map must be provided to compute E(B-V) values.")

        # Wrap the RA and dec in a SkyCoord object and call the dust map's query method.
        coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
        dustmap_value = self._dust_map.query(coord)

        # Only translate the value to ebv if the dust map does not provide the result in ebv.
        if self._ebv_func is not None:
            return self._ebv_func(dustmap_value)
        return dustmap_value


class ConstantHemisphereDustMap:
    """A DustMap with a constant value in each hemisphere for testing
    and debugging purposes.

    This class is designed to look like a DustMap object from the dustmaps package,
    so that it can be used to test DustmapWrapper.

    Attributes
    ----------
    north_ebv : `float`
        The DustMap's ebv value at all points in the Northern Hemisphere.
    south_ebv : `float`
        The DustMap's ebv value at all points in the Southern Hemisphere.
    """

    def __init__(self, north_ebv, south_ebv):
        self.north_ebv = north_ebv
        self.south_ebv = south_ebv

    def compute_ebv(self, ra, dec):
        """Compute the E(B-V) value for a given location.

        Parameters
        ----------
        ra : float or np.array
            The object's right ascension (in degrees).
        dec : float or np.array
            The object's declination (in degrees).

        Returns
        -------
        ebv : float or np.array
            The E(B-V) value or array of values.
        """
        if np.isscalar(ra):
            return self.north_ebv if dec >= 0 else self.south_ebv

        ebv = np.where(np.asarray(dec) < 0, self.south_ebv, self.north_ebv)
        return ebv

    def query(self, coords):
        """A query function to match the DustMap interface so that we can
        pass ConstantHemisphereDustMap into DustmapWrapper for testing purposes.

        Note
        ----
        This shouldn't be used in production code. Use the compute_ebv()
        function directly.

        Parameters
        ----------
        coords : SkyCoord
            The object's coordinates.

        Returns
        -------
        ebv : float or np.array
            The E(B-V) value or array of values.
        """
        return self.compute_ebv(coords.ra.deg, coords.dec.deg)


class SFDMap(DustEBV):
    """A dustmap using the sfdmap2 package.

    This does not need to be used with DustmapWrapper, but rather is
    a standalone ParameterizedNode.

    Note
    ----
    If the dustmap data is not present in "data/dustmaps/sfdmap2",
    it will automatically be downloaded there when the class is
    instantiated.

    Citations
    ---------
    Software https://github.com/AmpelAstro/sfdmap2
    Forked from https://github.com/kbarbary/sfdmap
    Dust map data from Schlegel, Finkbeiner and Davis (1998).

    Attributes
    ----------
    dustmap : sfdmap.SFDMap
        The dust map object.

    Parameters
    ----------
    data_dir : `str`, optional
        The directory containing the dust map data files.
        If None, the default directory will be used.
    RA : parameter
        The object's right ascension (in degrees).
    dec : parameter
        The object's declination (in degrees).
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    _default_map_dir = _LIGHTCURVELYNX_BASE_DATA_DIR / "dustmaps" / "sfdmap2"

    def __init__(self, data_dir=None, ra=None, dec=None, **kwargs):
        # Pass RA and dec through to DustEBV's init method.
        super().__init__(self.compute_ebv, ra=ra, dec=dec, **kwargs)
        self.dustmap = self._load_data(data_dir)

    def _data_files_exist(self, data_dir):
        """Check that the necessary data files exist in the given directory."""
        if not data_dir.exists() or not data_dir.is_dir():
            return False
        elif (data_dir / "SFD_dust_4096_ngp.fits").exists() is False:
            return False
        elif (data_dir / "SFD_dust_4096_sgp.fits").exists() is False:
            return False
        return True

    def _load_data(self, data_dir=None):
        """Load the dust map data files.

        Parameters
        ----------
        data_dir : `str`, optional
            The directory containing the dust map data files.
            If None, the default directory will be used.

        Returns
        -------
        dustmap : sfdmap.SFDMap
            The dust map object.
        """
        logger = logging.getLogger(__name__)

        # Check that we have the sfdmap2 package installed (since it is not installed by default).
        if importlib.util.find_spec("sfdmap2") is None:
            raise ImportError(
                "The sfdmap2 package is required to use the SFDMap effect and not installed by default. "
                "You can install sfdmap2 using `pip install sfdmap2`."
            )
        else:
            from sfdmap2 import sfdmap

        data_dir = Path(data_dir) if data_dir is not None else self._default_map_dir
        logger.debug(f"Loading SFD dust map data from {data_dir}")

        if not self._data_files_exist(data_dir):
            data_url = ("https://github.com/kbarbary/sfddata/archive/master.tar.gz",)
            logger.info(
                "SFD dust map data files not found.\n"
                f"Attempting to download from: {data_url}\n"
                f"to the directory {data_dir}"
            )

            # Create the data directory if it doesn't exist.
            if not data_dir.exists():
                data_dir.mkdir(parents=True)

            # Use pooch to download the data files and extract them to the data directory.
            pooch.retrieve(
                url="https://github.com/kbarbary/sfddata/archive/master.tar.gz",
                known_hash="95e8645dcdcbd4ad48398d44307741f550bdde95e8f438b2a3be0021723a4d7e",
                processor=pooch.Untar(extract_dir=data_dir),
            )
            data_dir = data_dir / "sfddata-master"

        # Check that the files were downloaded and extracted.
        if not self._data_files_exist(data_dir):
            raise ValueError(f"The SFD dust map data files are missing from {data_dir}.")

        return sfdmap.SFDMap(data_dir)

    def compute_ebv(self, ra, dec):
        """Compute the E(B-V) value for a given location.

        Parameters
        ----------
        ra : float or np.array
            The object's right ascension (in degrees).
        dec : float or np.array
            The object's declination (in degrees).

        Returns
        -------
        ebv : float or np.array
            The E(B-V) value or array of values.
        """
        ebv = self.dustmap.ebv(ra, dec)
        return ebv
