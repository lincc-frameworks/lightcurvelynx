"""
The ResultsBuilder is a helper class that builds simulation results incrementally,
keeping track of various meta-data.

The main goal of this function is to simplify the code in the _simulate_lightcurves_batch
function while adding significantly more validity checks and convenience for managing the results.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
from nested_pandas import NestedFrame

logger = logging.getLogger(__name__)


class SimulationResultBuilder:
    """
    A helper class to build simulation results incrementally, which keeping track
    of various meta-data. We break this into a separate class to similify the
    _simulate_lightcurves_batch function.

    Parameters
    ----------
    ra : array-like
        Right ascension of the objects that have realized samples.
    dec : array-like
        Declination of the objects that have realized samples.
    t0 : array-like
        Reference time of the objects that have realized samples.
    redshift : array-like
        Redshift of the objects that have realized samples.

    Attributes
    ----------
    results_dict : dict
        The dictionary storing the top-level results for each object. These will be
        the non-nested columns for each object.
    results_dtypes : dict
        The dictionary storing the data types for each top-level column.
    lightcurve_index : list
        The list storing the indices for the nested light curve observations.
    lightcurve_dict : dict
        The dictionary storing the nested light curve observations.
    lightcurve_dtypes : dict
        The dictionary storing the data types for each nested light curve column.
    spectra_index : list
        The list storing the indices for the nested spectra observations.
    spectra_dict : dict
        The dictionary storing the nested spectra observations.
    spectra_dtypes : dict
        The dictionary storing the data types for each nested spectra column.
    """

    def __init__(self, ra, dec, t0, redshift):
        self.realized_num_samples = len(ra)
        if len(dec) != self.realized_num_samples:
            raise ValueError("Length of dec must match length of ra.")
        if len(t0) != self.realized_num_samples:
            raise ValueError("Length of t0 must match length of ra.")
        if len(redshift) != self.realized_num_samples:
            raise ValueError("Length of redshift must match length of ra.")

        # Create the basic results dictionary and the initial column dtypes.
        self.results_dict = {
            "id": list(range(self.realized_num_samples)),
            "ra": ra.tolist(),
            "dec": dec.tolist(),
            "t0": t0.tolist(),
            "z": redshift.tolist(),
            "nobs": [0] * self.realized_num_samples,
        }
        self.results_dtypes = {
            "id": int,
            "ra": float,
            "dec": float,
            "t0": float,
            "z": float,
            "nobs": int,
        }

        # Create a nested array for light curves.
        self.lightcurve_index = []
        self.lightcurve_dict = {
            "mjd": [],
            "filter": [],
            "flux": [],
            "fluxerr": [],
            "flux_perfect": [],
            "survey_idx": [],  # The index of the survey
            "obs_idx": [],  # The index of the observation in the survey
            "is_saturated": [],
        }
        self.lightcurve_dtypes = {
            "mjd": float,
            "filter": str,
            "flux": float,
            "fluxerr": float,
            "flux_perfect": float,
            "survey_idx": int,
            "obs_idx": int,
            "is_saturated": bool,
        }
        self.current_index = None
        self.current_light_curve = None
        self.current_spectra = None
        self.current_num_obs = 0

        # Create a nested array for spectra. Most dtypes are objects, because
        # the are lists of values (one list per spectra observation).
        self.spectra_index = []
        self.spectra_dict = {
            "mjd": [],
            "waves": [],
            "measured_flux": [],
            "measured_flux_perfect": [],
            "measured_flux_error": [],
            "instrument": [],
        }

        spectrum_array_dtype = pd.ArrowDtype(pa.list_(pa.float64()))
        self.spectra_dtypes = {
            "mjd": float,
            "waves": spectrum_array_dtype,
            "measured_flux": spectrum_array_dtype,
            "measured_flux_perfect": spectrum_array_dtype,
            "measured_flux_error": spectrum_array_dtype,
            "instrument": pd.StringDtype(),
        }

    def add_result_column(self, column_name, column_values, dtype=None):
        """
        Add a new column to the top-level (non-nested) results.

        Parameters
        ----------
        column_name : str
            The name of the column to add.
        column_values : list
            The values for the new column. Must have the same length as the number of samples.
        dtype : type, optional
            The data type of the column. If not specified, it will be inferred from the first value.
        """
        # Check that the column does not exist yet.
        if column_name in self.results_dict:
            raise KeyError(f"Column {column_name} already exists in the results.")

        # Check that the column has the correct number of values.
        if len(column_values) != self.realized_num_samples:
            raise ValueError(
                f"Length of column_values ({len(column_values)}) does not match the "
                f"number of samples ({self.realized_num_samples})."
            )
        self.results_dict[column_name] = column_values

        # Use the given dtype if provided, otherwise infer it from the set of values.
        if dtype is not None:
            self.results_dtypes[column_name] = dtype
        else:
            # Check whether there is a single consistent type.
            all_types = set(type(val) for val in column_values)
            if len(all_types) == 1:
                inferred_type = all_types.pop()
            else:
                inferred_type = object
            self.results_dtypes[column_name] = inferred_type

    def add_param_column(self, param_name, sample_states):
        """
        Add a new column of sampled parameter values to the top-level (non-nested) results
        from the sample states.

        Parameters
        ----------
        param_name : str
            The name of the parameter to add, using the dot notation that includes
            the node name.
        sample_states : GraphState
            The ``GraphState`` object containing the parameter values.
        """
        if param_name not in sample_states:
            raise KeyError(
                f"Parameter column {param_name} not found in model parameters. "
                f"Available parameters are: {sample_states.get_all_params_names()}."
            )

        new_name = param_name.replace(".", "_")
        logger.debug(f"Adding parameter column {param_name} as {new_name}")
        self.add_result_column(new_name, np.atleast_1d(sample_states[param_name]).tolist())

    def add_light_curve_column(self, column_name, obs_table_list=None, dtype=None):
        """
        Add a new empty column to the light curve nested dictionary.

        Parameters
        ----------
        column_name : str
            The name of the column to add.
        obs_table_list : list, optional
            A list of observation tables corresponding to the surveys. If no dtype is provided,
            this is used to determine the dtype of the column.
        dtype : type, optional
            The data type of the column. If not specified, it will be inferred from the observation tables.
        """
        if len(self.lightcurve_dict["mjd"]) != 0:
            raise RuntimeError("Cannot add new light curve columns after observations have been added.")
        if column_name in self.lightcurve_dict:
            raise KeyError(f"Column {column_name} already exists in the light curve dictionary.")

        # Save the dtype of the column.
        if dtype is not None:
            self.lightcurve_dtypes[column_name] = dtype
        elif obs_table_list is None or len(obs_table_list) == 0:
            raise ValueError("Must either provide a dtype or a list of ObsTables.")
        elif np.all([column_name not in obs_table for obs_table in obs_table_list]):
            # Don't store the column if it will have no values. But produce a warning.
            warnings.warn(f"No observation tables contain column {column_name}.")
            return
        else:
            # Collect the dtypes of the column from each observation table. If the column is missing,
            # use object.
            all_dtypes = [
                (
                    obs_table[column_name].dtype
                    if hasattr(obs_table[column_name], "dtype")
                    else obs_table.get_value_per_row(column_name, indices=[]).dtype
                )
                if column_name in obs_table
                else object
                for obs_table in obs_table_list
            ]

            # Make sure we can determine a consistent dtype for the column.
            if len(set(all_dtypes)) == 1:
                self.lightcurve_dtypes[column_name] = all_dtypes[0]
            else:
                self.lightcurve_dtypes[column_name] = object

        # Add the empty column.
        self.lightcurve_dict[column_name] = []

    def start_new_object(self, index):
        """
        Prepare to start collecting information for a new object. This includes
        both filter and spectral observations for the object over multiple surveys.

        Parameters
        ----------
        index : int
            The index of the light curve to initialize.
        """
        if self.current_light_curve is not None:
            raise RuntimeError("Cannot start a new object before appending the current one.")

        if index < 0 or index >= self.realized_num_samples:
            raise ValueError(f"Index {index} is out of range for the light curve samples.")
        self.current_index = index

        # Create a temporary empty object for this index.
        self.current_light_curve = {key: [] for key in self.lightcurve_dict}
        self.current_spectra = {key: [] for key in self.spectra_dict}
        self.current_num_obs = 0

    def add_light_curve_observations(self, **kwargs):
        """
        Add light curve observations to the current object. The values are the lists of observations
        for each column are provided as lists in the keyword arguments. Values must be provided for
        all columns in the light curve dictionary, including: mjd, filter, flux, fluxerr, flux_perfect,
        survey_idx, obs_idx, is_saturated, and any obs_table columns.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments where the key is the column name and the value is a list of observations.
        """
        if self.current_light_curve is None:
            raise RuntimeError("start_new_object must be called before adding observations.")

        num_observations = len(kwargs["mjd"])
        for col in self.lightcurve_dict:
            if col not in kwargs:
                raise KeyError(f"Missing observations for column {col}.")

            # We keep the observations as a list of lists and use np.concatenate when
            # needed to flatten them.
            values = kwargs[col]
            if len(values) != num_observations:
                raise ValueError(f"Column {col} must have {num_observations} observations.")
            self.current_light_curve[col].append(values)
        self.current_num_obs += num_observations

    def add_spectra_observations(self, **kwargs):
        """
        Add observations corresponding to a single spectra entry. Each value is a list
        list of wavelengths corresponding to the spectral observation.
        Add observations to the spectra nested dictionary. The values are the lists of
        observations for each column are provided as lists in the keyword arguments. Values
        must be provided for all columns in the spectra dictionary, including: mjd, waves,
        measured_flux, measured_flux_perfect, measured_flux_error, and instrument.

        Parameters
        ----------
        index : int
            The index of the spectra to which the observations should be added.
        **kwargs : dict
            Keyword arguments where the key is the column name and the value is a list of observations.
        """
        if self.current_index is None:
            raise RuntimeError("start_new_object must be called before adding spectra observations.")

        for col in self.current_spectra:
            if col not in kwargs:
                raise KeyError(f"Missing observations for column {col}.")

            # We maintain the overall spectra dictionary as a list of lists, where each entry
            # corresponds to a single spectra observation.
            self.current_spectra[col].append(kwargs[col])
        self.current_num_obs += 1

    def append_current_object(self):
        """
        Finish and append the current object to the main light curve dictionary. We sort
        the observations by their MJD before appending.
        """
        if self.current_light_curve is None:
            raise RuntimeError("No current light curve to append.")

        # Nothing to append.
        if self.current_num_obs == 0:
            self.current_light_curve = None
            self.current_spectra = None
            self.current_index = None
            return

        # Sort the values of each column by increasing time (MJD) and then append
        # these values onto the global light curve table,
        if len(self.current_light_curve["mjd"]) > 0:
            all_mjd = np.concatenate(self.current_light_curve["mjd"])
            sorted_indices = np.argsort(all_mjd, kind="stable")

            for col in self.current_light_curve:
                time_sorted_values = np.concatenate(self.current_light_curve[col])[sorted_indices]
                self.lightcurve_dict[col].extend(time_sorted_values.tolist())

            # Add the object index so we can build the nested frame later.
            self.lightcurve_index.extend([self.current_index] * len(sorted_indices))

        # Append the spectra information onto the global spectra table.
        if len(self.current_spectra["mjd"]) > 0:
            for col in self.current_spectra:
                self.spectra_dict[col].extend(self.current_spectra[col])
            self.spectra_index.extend([self.current_index] * len(self.current_spectra["mjd"]))

        # Reset the data structure for the next light curve.
        self.results_dict["nobs"][self.current_index] += self.current_num_obs
        self.current_num_obs = 0
        self.current_light_curve = None
        self.current_spectra = None
        self.current_index = None

    def build_results(self, state=None):
        """Construct the nested results data frame.

        Parameters
        ----------
        state : GraphState, optional
            The GraphState object from which to extract the parameters for the nested frame.

        Returns
        -------
        nested_pandas.NestedFrame
            The constructed nested results data frame.
        """
        # Create the outer results data frame (with appropriate dtypes)
        results_data = {}
        for col in self.results_dict:
            results_data[col] = pd.Series(self.results_dict[col], dtype=self.results_dtypes[col])
        results_df = NestedFrame(
            data=results_data,
            index=list(range(self.realized_num_samples)),
        )

        # Create the light curve data frame (with appropriate dtypes) and merge it in.
        light_curve_data = {}
        for col in self.lightcurve_dict:
            light_curve_data[col] = pd.Series(
                self.lightcurve_dict[col],
                dtype="string" if self.lightcurve_dtypes[col] is str else self.lightcurve_dtypes[col],
                index=self.lightcurve_index,
            )
        lightcurve_frame = pd.DataFrame(data=light_curve_data)
        results_df = results_df.join_nested(lightcurve_frame, "lightcurve")

        # Create the spectra data frame (with appropriate dtypes) and merge it in.
        spectra_data = {}
        for col in self.spectra_dict:
            spectra_data[col] = pd.Series(
                self.spectra_dict[col],
                dtype=self.spectra_dtypes[col],
                index=self.spectra_index,
            )
        spectra_frame = pd.DataFrame(data=spectra_data)
        results_df = results_df.join_nested(spectra_frame, "spectra")

        # Add the parameters as a pyarrow column if provided.
        if state is not None:
            params = state.to_pyarrow_struct_array()
            results_df["params"] = pd.Series(params, dtype=pd.ArrowDtype(params.type))

        return results_df
