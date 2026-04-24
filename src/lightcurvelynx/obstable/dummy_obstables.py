"""Dummy ObsTable classes for testing and development."""

import numpy as np
import pandas as pd


class LookupOnlyObsTable:
    """A simple dummy class to simulate the ObsTable's lookup methods.

    Attributes
    ----------
    values : dict or pandas.core.frame.DataFrame
        The table with all the survey information, organized as a dictionary of column name
        to values.
    """

    def __init__(self, values, survey_values=None):
        if isinstance(values, dict):
            values = pd.DataFrame(values)
        self.values = values
        self.survey_values = survey_values if survey_values is not None else {}

    def __len__(self):
        """Return the number of rows in the table."""
        return len(self.values)

    def __getitem__(self, key):
        """Access the underlying observation table by column or parameter name. This will
        return either a full column from the table or a survey parameter value.
        """
        if key in self.values.columns:
            return self.values[key]
        if key in self.survey_values:
            return self.survey_values[key]
        raise KeyError(f"Column or parameter not found: {key}")

    def __contains__(self, key):
        """Check if a column exists in the survey table or a parameter in the parameter table."""
        return key in self.values.columns or key in self.survey_values

    def get_value_per_row(self, key, indices, default=None):
        """Simulate the ObsTable's get_value_per_row method."""
        if key in self.values.columns:
            return self.values[key].iloc[indices].values
        if key in self.survey_values:
            return np.full(len(indices), self.survey_values[key])
        if default is not None:
            return np.full(len(indices), default)
        raise KeyError(f"Missing required key: {key}")

    def safe_get_survey_value(self, key):
        """Get a survey value by key, checking that it is not None.

        Parameters
        ----------
        key : str
            The key of the survey value to retrieve.
        """
        value = self.survey_values.get(key, None)
        if value is None:
            raise ValueError(
                f"Survey value for {key} is not defined. This should be set when creating the object."
            )
        return value
