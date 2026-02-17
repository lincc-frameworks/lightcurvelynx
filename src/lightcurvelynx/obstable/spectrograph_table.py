"""A class for storing and working with simple spectrograph observation tables"""

from __future__ import annotations  # "type1 | type2" syntax in Python <3.10

import numpy as np
import pandas as pd

from lightcurvelynx.obstable.obs_table import ObsTable


class SpectrographObsTable(ObsTable):
    """An ObsTable for simple spectrograph observations.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the spectrograph observation information.
    colmap : dict
        A mapping of standard column names to a list of possible names in the input table.
        Each value in the dictionary can be a string or a list of strings.
        Defaults to the Rubin CCDVisit column names, stored in _default_colnames.
    **kwargs : dict
        Additional keyword arguments to pass to the constructor.
    """

    _required_names = ["ra", "dec", "time"]

    # By default we do not need any column mapping.
    _default_colnames = {}

    # By default there are no survey values.
    _default_survey_values = {
        "radius": 10.0 / 3600.0,  # degrees, corresponds to a 10 arcsec radius
    }

    # Class constants for the column names.
    def __init__(
        self,
        table,
        colmap=None,
        **kwargs,
    ):
        # If the input table is a dictionary, convert it to a DataFrame.
        if isinstance(table, dict):  # pragma: no cover
            table = pd.DataFrame(table)
        colmap = self._default_colnames if colmap is None else colmap

        # Fill in a dummy filter value if none is given. This will not be used, but
        # is required for the ObsTable constructor.
        if "filter" not in table.columns and (
            "filter" not in colmap or colmap["filter"] not in table.columns
        ):
            table["filter"] = "spectra"  # Name does not actually matter.

        super().__init__(table, colmap=colmap, **kwargs)

    def _assign_zero_points(self):
        """Assign instrumental zero points in nJy (which produces 1 e-) to the LSSTObsTable tables."""
        if "zp" in self._table.columns:
            return  # Nothing to do

        # Add a fake zero point column if we have no information to compute it. This column is not
        # used for spectrographs.
        self.add_column("zp", 0.05 * np.ones(len(self._table)), overwrite=True)
