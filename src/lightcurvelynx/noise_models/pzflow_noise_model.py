"""A wrapper that trains and queries pzflow for sampling noise.

We strongly recommend using the ``learn_pzflow_noise_model`` function to train
a flow and create a PZFlowNoiseModel.

For the full pzflow package see:
https://github.com/jfcrenshaw/pzflow
"""

import pickle

import numpy as np
import pandas as pd
from citation_compass import CiteClass, cite_function

from lightcurvelynx.noise_models.base_noise_models import FluxNoiseModel


class _ColumnNormalizationData:
    """A class to hold the data needed for normalizing the data for training
    and prediction data for a PZFlowNoiseModel.

    Attributes
    ----------
    offset : float
        The offset value used for normalization, typically the minimum value of the data.
    scale : float
        The scale factor to be used for normalization, typically the range of the data.
    log_transform : bool
        Whether to apply a log transform to the data for normalization. If True,
        the data will be transformed using log10(x - offset + 1) before applying
        the scaling. This can be useful for data that is highly skewed or has a
        large dynamic range.

    Parameters
    ----------
    data : array_like
        The data to be used for computing the normalization parameters.
    log_transform : bool, optional
        Whether to apply a log transform to the data for normalization. If True,
        the data will be transformed using log10(x - offset + 1) before applying
        the scaling. This can be useful for data that is highly skewed or has a
        large dynamic range.
    """

    def __init__(self, data, log_transform=False):
        # Perform some basic checks on the data.
        data = np.asarray(data)
        if len(data) == 0:
            raise ValueError("Data is empty, cannot compute normalization parameters.")
        if np.all(np.isnan(data)):
            raise ValueError("Data contains only NaN values, cannot compute normalization parameters.")
        if log_transform and np.any(data <= 0):
            raise ValueError(
                "Data contains non-positive values, cannot apply log transform for normalization."
            )

        # Take the log of the data (if needed).
        self.log_transform = log_transform
        if log_transform:
            data = np.log10(data)

        # Compute the scale of the data such that it will have a range of width 1 after normalization.
        range = np.max(data) - np.min(data)
        if range <= 0:
            # Handle the case where all the data is the same.
            range = 1.0
        self.scale = 1.0 / range
        data = data * self.scale

        # Shift the data to lie between 0 and 1 after normalization.
        self.offset = -np.min(data)

    def normalize(self, data):
        """Normalize the data using the specified min, max, and log transform.

        Parameters
        ----------
        data : array_like
            The data to be normalized.

        Returns
        -------
        array_like
            The normalized data.
        """
        data = np.asarray(data)
        if self.log_transform:
            if np.any(data <= 0):
                raise ValueError(
                    "Data contains non-positive values, cannot apply log transform for normalization."
                )
            data = np.log10(data)
        data = data * self.scale + self.offset

        return data

    def denormalize(self, data):
        """Denormalize the data using the specified min, max, and log transform.

        Parameters
        ----------
        data : array_like
            The data to be denormalized.

        Returns
        -------
        array_like
            The denormalized data.
        """
        data = (data - self.offset) / self.scale
        if self.log_transform:
            data = 10**data
        return data


class PZFlowNoiseModel(FluxNoiseModel, CiteClass):
    """A noise model that uses pzflow to sample noise parameters (standard
    deviation of the noise) for bandflux measurements.

    The names of the pzflow input parameters should either match the column names
    or be mapped to column names in the ObsTable using the `input_col_map` when
    creating the PZFlowNoiseModel. The exception to this is `bandflux` input,
    which would be passed directly as an argument to the `apply_noise` method.

    We strongly recommend using the ``learn_pzflow_noise_model`` function to train
    a flow and create a PZFlowNoiseModel as this will correctly handle potentially error-prone
    aspects like normalization.

    References
    ----------
    * Paper: Crenshaw et. al. 2024 - https://ui.adsabs.harvard.edu/abs/2024AJ....168...80C
    * Zenodo: Crenshaw et. al. 2024 - https://doi.org/10.5281/zenodo.10710271

    Parameters
    ----------
    flow_obj : pzflow.flow.Flow or pzflow.flowEnsemble.FlowEnsemble
        The object from which to sample.
    input_col_map : dict, optional
        A dictionary where the keys are the input parameter names for the pzflow
        model and the values are the names of the columns in the ObsTable that
        should be used for those parameters.
    normalizer_data : dict, optional
        A dictionary where the keys are the column names of the data used for
        training the flow and the values are _ColumnNormalizationData objects that
        contain the information needed to normalize and denormalize the data for prediction.
    """

    def __init__(
        self,
        flow_obj,
        *,
        input_col_map=None,
        normalizer_data=None,
    ):
        # Validate the pzflow object has the expected output column
        # and save its name.
        self._flow = flow_obj
        if len(self._flow.data_columns) != 1:
            raise ValueError(
                "PZFlowNoiseModel currently only supports flows with a"
                "single data column (the standard deviation of the noise)."
            )
        self._output_column = self._flow.data_columns[0]

        # Save the meta data.
        self._input_col_map = input_col_map if input_col_map is not None else {}
        self._normalizer_data = normalizer_data if normalizer_data is not None else {}

    def add_column_mapping(self, flow_input_name, obs_table_col_name):
        """Add a mapping from a pzflow input parameter name to an ObsTable column name.

        This function is used when the model is created with on data with different
        column names than the ObsTable on which it will be applied.

        Parameters
        ----------
        flow_input_name : str
            The name of the input parameter for the pzflow model.
        obs_table_col_name : str
            The name of the column in the ObsTable that should be used for this parameter.
        """
        self._input_col_map[flow_input_name] = obs_table_col_name

    @classmethod
    def from_file(cls, filename, *, input_col_map=None):
        """Create a PZFlowNoiseModel from a saved file.

        Parameters
        ----------
        filename : str or Path
            The location of the saved flow.
        input_col_map : dict, optional
            A dictionary where the keys are the input parameter names for the pzflow
            model and the values are the names of the columns in the ObsTable that
            should be used for those parameters. If provided this overrides the input_col_map
            that was saved with the model.
        """
        with open(filename, "rb") as f:
            noise_model = pickle.load(f)
        assert isinstance(noise_model, PZFlowNoiseModel), "The loaded object is not a PZFlowNoiseModel."

        if input_col_map is not None:
            noise_model._input_col_map = input_col_map

        return noise_model

    def apply_noise(
        self,
        bandflux,
        *,
        obs_table=None,
        indices=None,
        rng=None,
        **kwargs,
    ):
        """Compute the noise parameters for given observations in
        an ObsTable and apply noise to the input bandflux.

        Parameters
        ----------
        bandflux : array_like of float
            Source bandflux in energy units, e.g. nJy.
        obs_table : ObsTable, optional
            Table containing the observation parameters, including all
            parameters needed to compute the noise.
        indices : array_like of int, optional
            Indices of the observations in the ObsTable to which noise should
            be applied.
        rng : np.random.Generator, optional
            The random number generator to use for applying noise. If None,
            a default generator will be used.
        **kwargs
            Additional parameters for the noise model.

        Returns
        -------
        flux : array_like
            The updated flux measurements after applying noise, in the same
            units as the input bandflux.
        flux_err : array_like
            The bandflux measurement error used for applying noise, in the
            same units as the input bandflux.
        """
        if obs_table is None:
            raise ValueError("ObsTable must be provided for PZFlowNoiseModel.")
        if indices is None:
            raise ValueError("Indices must be provided for PZFlowNoiseModel.")
        num_samples = len(bandflux)
        if len(indices) != num_samples:
            raise ValueError("Length of indices must match length of bandflux.")

        # Get the input parameters for the flow (if there are any).
        if self._flow.conditional_columns is not None and len(self._flow.conditional_columns) > 0:
            input_params = {}
            for col in self._flow.conditional_columns:
                if col == "bandflux" and bandflux is not None:
                    values = bandflux
                else:
                    key = self._input_col_map.get(col, col)
                    values = obs_table.get_value_per_row(key, indices=indices)

                # Normalize the input parameters if needed. We can lookup by col (instead of key),
                # because the normalization is based on the pzflow column names.
                if self._normalizer_data.get(col) is not None:
                    normalizer = self._normalizer_data[col]
                    values = normalizer.normalize(values)

                # Record the input parameters for this column.
                input_params[col] = values

            input_df = pd.DataFrame(input_params)
        else:
            input_df = None

        # Sample from the flow to get the noise parameters.
        rng = np.random.default_rng(rng)
        pzflow_seed = rng.integers(0, 1e9)
        samples = self._flow.sample(nsamples=1, conditions=input_df, seed=pzflow_seed)
        flux_err = np.clip(samples[self._output_column].values, a_min=0, a_max=None)

        # If we have normalization data for the output column, denormalize the output.
        if self._normalizer_data.get(self._output_column) is not None:
            normalizer = self._normalizer_data[self._output_column]
            flux_err = normalizer.denormalize(flux_err)

        # Apply noise to the input bandflux using the sampled noise parameters.
        noisy_bandflux = rng.normal(loc=bandflux, scale=flux_err)
        return noisy_bandflux, flux_err

    def save_to_file(self, filename):
        """Save the PZFlowNoiseModel to a file.

        Parameters
        ----------
        filename : str or Path
            The location where the flow should be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)


@cite_function
def learn_pzflow_noise_model(
    data,
    *,
    noise_column=None,
    normalize=True,
    **kwargs,
):
    """Train a pzflow model to predict noise parameters (standard deviation of the noise) for
    bandflux measurements.

    Parameters
    ----------
    data : dict or pd.DataFrame
        The data to be used for training the flow. This should include all conditional
        parameters listed in the ``conditional_columns`` parameter as well as the
        ``noise_column`` parameter.
    noise_column : str or array_like, optional
        The name of the column in ``data`` that contains the noise values to predict. All
        other columns are treated as input.
    normalize : bool, optional
        Whether to normalize the data for training the flow. This can help the
        flow learn the distribution more effectively, especially if the data has
        a large dynamic range or is highly skewed.
    **kwargs
        Additional parameters for training the flow.

    References
    ----------
    * Paper: Crenshaw et. al. 2024 - https://ui.adsabs.harvard.edu/abs/2024AJ....168...80C
    * Zenodo: Crenshaw et. al. 2024 - https://doi.org/10.5281/zenodo.10710271
    """
    # Check that we can load the pzflow package before doing any other work.
    try:
        from pzflow import Flow
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "pzflow package is not installed by default. You can install it with "
            "`pip install pzflow` or `conda install conda-forge::pzflow`."
        ) from err

    # Check that we have training data with all the columns we need. Make a copy
    # of the data so we can normalize it safely.
    local_data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if noise_column is None or not isinstance(noise_column, str):
        raise ValueError("A noise_column string must be specified to train the flow.")
    if noise_column not in local_data.columns:
        raise ValueError(f"noise_column '{noise_column}' not found in data.")

    # Normalize each column of the data and save the normalization information.
    normalizer_data = {}
    for col in local_data.columns:
        if normalize:
            log_transform = (col == noise_column) or (col == "bandflux")
            normalizer = _ColumnNormalizationData(local_data[col], log_transform=log_transform)
            normalizer_data[col] = normalizer
            local_data[col] = normalizer.normalize(local_data[col])
        else:
            normalizer_data[col] = None

    # Train the actual flow using the normalized data.
    cond_columns = [col for col in local_data.columns if col != noise_column]
    flow = Flow(data_columns=[noise_column], conditional_columns=cond_columns)
    _ = flow.train(pd.DataFrame(local_data), verbose=False, **kwargs)

    return PZFlowNoiseModel(
        flow_obj=flow,
        normalizer_data=normalizer_data,
    )
