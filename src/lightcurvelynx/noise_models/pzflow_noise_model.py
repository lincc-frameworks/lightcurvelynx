"""A wrapper that uses pzflow for sampling noise.

For the full pzflow package see:
https://github.com/jfcrenshaw/pzflow
"""

import numpy as np
import pandas as pd
from citation_compass import CiteClass

from lightcurvelynx.noise_models.base_noise_models import FluxNoiseModel


class PZFlowNoiseModel(FluxNoiseModel, CiteClass):
    """A noise model that uses pzflow to sample noise parameters (standard
    deviation of the noise) for bandflux measurements.

    The names of the pzflow input parameters should either match the column names
    or be mapped to column names in the ObsTable using the `input_col_map` when
    creating the PZFlowNoiseModel. The exception to this is `bandflux` input,
    which would be passed directly as an argument to the `apply_noise` method.

    References
    ----------
    * Paper: Crenshaw et. al. 2024 - https://ui.adsabs.harvard.edu/abs/2024AJ....168...80C
    * Zenodo: Crenshaw et. al. 2024 - https://doi.org/10.5281/zenodo.10710271

    Attributes
    ----------
    flow : pzflow.flow.Flow or pzflow.flowEnsemble.FlowEnsemble
        The object from which to sample.

    Parameters
    ----------
    flow_obj : pzflow.flow.Flow or pzflow.flowEnsemble.FlowEnsemble
        The object from which to sample.
    input_col_map : dict, optional
        A dictionary where the keys are the input parameter names for the pzflow
        model and the values are the names of the columns in the ObsTable that
        should be used for those parameters.
    """

    def __init__(
        self,
        flow_obj,
        *,
        input_col_map=None,
    ):
        # Validate the pzflow object has the expected output column
        # and save its name.
        self.flow = flow_obj
        if len(flow_obj.data_columns) != 1:
            raise ValueError(
                "PZFlowNoiseModel currently only supports flows with a"
                "single data column (the standard deviation of the noise)."
            )
        self._output_column = flow_obj.data_columns[0]

        # Save the mapping of input parameter names to ObsTable column names.
        if input_col_map is None:
            self._input_col_map = {}
        else:
            self._input_col_map = input_col_map

    @classmethod
    def from_file(cls, filename, *, input_col_map=None):
        """Create a PZFlowNoiseModel from a saved flow file.

        Parameters
        ----------
        filename : str or Path
            The location of the saved flow.
        input_col_map : dict, optional
            A dictionary where the keys are the input parameter names for the pzflow
            model and the values are the names of the columns in the ObsTable that
            should be used for those parameters.
        """
        try:
            from pzflow import Flow
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "pzflow package is not installed by default. You can install it with "
                "`pip install pzflow` or `conda install conda-forge::pzflow`."
            ) from err

        flow_to_use = Flow(file=filename)
        return PZFlowNoiseModel(flow_to_use, input_col_map=input_col_map)

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
        if self.flow.conditional_columns is not None and len(self.flow.conditional_columns) > 0:
            input_params = {}
            for col in self.flow.conditional_columns:
                if col == "bandflux":
                    input_params[col] = bandflux
                else:
                    key = self._input_col_map.get(col, col)
                    input_params[col] = obs_table.get_value_per_row(key, indices=indices)
            input_df = pd.DataFrame(input_params)
        else:
            input_df = None

        # Sample from the flow to get the noise parameters.
        rng = np.random.default_rng(rng)
        pzflow_seed = rng.integers(0, 1e9)
        samples = self.flow.sample(nsamples=1, conditions=input_df, seed=pzflow_seed)
        flurerr = np.clip(samples[self._output_column].values, a_min=0, a_max=None)

        # Apply noise to the input bandflux using the sampled noise parameters.
        noisy_bandflux = rng.normal(loc=bandflux, scale=flurerr)
        return noisy_bandflux, flurerr
