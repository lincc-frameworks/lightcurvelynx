"""Dummy ObsTables that can be used for testing noise models."""

import numpy as np
import pandas as pd


class LookupOnlyObsTable:
    """A simple dummy class to simulate the ObsTable's basic getter methods and
    get_value_per_row method.

    Attributes
    ----------
    table_values : pandas.DataFrame
        A DataFrame containing values stored in the table part of the ObsTable.
    const_values : dict
        A dictionary containing constant values for the ObsTable (survey_values).
    """

    def __init__(self, table_values, const_values=None):
        self.table_values = pd.DataFrame(table_values)
        self.const_values = const_values or {}

    def __len__(self):
        return len(self.table_values)

    def __contains__(self, key):
        return key in self.table_values.columns or key in self.const_values

    def __getitem__(self, key):
        if key in self.table_values.columns:
            return self.table_values[key]
        if key in self.const_values:
            return self.const_values[key]
        raise KeyError(f"Key '{key}' not found in table values or constant values.")

    def get_value_per_row(self, key, *, indices=None, default=None):
        """Get the values for each row from the table or survey values (defaults).

        Parameters
        ----------
        key : str
            The name of the column to retrieve.
        indices : numpy.ndarray, optional
            The indices of the rows for which to retrieve values. If None, retrieve all rows.
            Default: None
        default : any, optional
            The default value to use if the key is not found in the table or survey values.
            This can be None to indicate missing values. Default: None

        Returns
        -------
        numpy.ndarray
            The values for each row in the table.
        """
        if indices is None:
            indices = slice(None)
            num_indices = len(self.table_values)
        else:
            num_indices = len(indices)

        # Prioritize columns that are in the table.
        if key in self.table_values.columns:
            return self.table_values[key][indices].to_numpy()

        # Otherwise fall back to the survey values if they are defined.
        value = self.const_values.get(key, None)
        if value is None:
            return np.full((num_indices,), default)
        if isinstance(value, float | int | np.float64 | np.int64):
            # Use the same value for all rows.
            return np.full((num_indices,), value)
        if isinstance(value, dict):
            # We have a dictionary mapping the values for each filter to the values for those rows.
            if "filter" not in self.table_values.columns:
                raise ValueError(
                    f"Cannot use a dictionary for '{key}' if there is no 'filter' column in the table."
                )

            # Map the values for each filter to the rows in the table.
            result = np.zeros(num_indices, dtype=float)
            for fil, val in value.items():
                result[self.table_values["filter"][indices] == fil] = val
            return result
        raise TypeError(f"Unsupported type for '{key}': {type(value)}")
