"""Samplers used for testing that produces precomputed results. These
can be used in testing to produce known results or to use data previously
sampled from another method (such as pzflow).
"""

import warnings

import numpy as np
import pandas as pd
from astropy.table import Table

from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc


class BinarySampler(NumpyRandomFunc):
    """A FunctionNode that randomly returns True or False according
    to a given probability. This function is particularly useful in
    probabilistically applying effects or making decisions in the
    simulation.

    Attributes
    ----------
    probability : float
        The probability of returning True.
    """

    def __init__(self, probability, seed=None, **kwargs):
        if probability < 0 or probability > 1:
            raise ValueError(f"Probability must be between 0 and 1. Got {probability}.")
        self.probability = probability

        super().__init__("uniform", seed=seed, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        rng = rng_info if rng_info is not None else self._rng

        if graph_state.num_samples == 1:
            results = rng.random() < self.probability
        else:
            results = rng.random(graph_state.num_samples) < self.probability
        self._save_results(results, graph_state)

        return results


class GivenValueList(FunctionNode):
    """A FunctionNode that returns given results for a single parameter
    in the order in which they are provided.

    Note
    ----
    This is a stateful node that keeps track of the next index to return
    that cannot be used in parallel sampling.

    Attributes
    ----------
    values : float, list, or numpy.ndarray
        The values to return.
    next_ind : int
        The index of the next value.
    """

    def __init__(self, values, **kwargs):
        self.values = np.asarray(values)
        if len(values) == 0:
            raise ValueError("No values provided for GivenValueList")
        self.next_ind = 0

        super().__init__(self._non_func, **kwargs)

    def __getstate__(self):
        raise RuntimeError("GivenValueList cannot be pickled. This node does not support parallel sampling.")

    def reset(self):
        """Reset the next index to use."""
        self.next_ind = 0

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            Unused in this function, but included to provide consistency with other
            compute functions.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        sample_ind = self.next_ind
        if graph_state.sample_offset is not None:
            sample_ind += graph_state.sample_offset

        if graph_state.num_samples == 1:
            if sample_ind >= len(self.values):
                raise IndexError(
                    f"GivenValueList ran out of entries to sample. Index {sample_ind} out "
                    f"of bounds for a list with {len(self.values)} entries."
                )

            results = self.values[sample_ind]
            self.next_ind += 1
        else:
            end_ind = sample_ind + graph_state.num_samples
            if end_ind > len(self.values):
                raise IndexError(
                    f"GivenValueList ran out of entries to sample. Index {sample_ind} out "
                    f"of bounds for a list with {len(self.values)} entries."
                )

            results = self.values[sample_ind:end_ind]
            self.next_ind += graph_state.num_samples

        # Save and return the results.
        self._save_results(results, graph_state)
        return results


class GivenValueSampler(NumpyRandomFunc):
    """A FunctionNode that returns randomly selected items from a given list
    with replacement.

    Attributes
    ----------
    values : int, list, or numpy.ndarray
        The values to select from. If an integer is provided, it is treated as a range
        from 0 to that value - 1.
    _num_values : int
        The number of values that can be sampled.
    _weights : numpy.ndarray, optional
        The weights for each value, if provided. If None, all values are equally likely.
    """

    def __init__(self, values, weights=None, seed=None, **kwargs):
        if isinstance(values, int):
            values = np.arange(values)
        self.values = np.asarray(values)

        self._num_values = len(values)
        if self._num_values == 0:
            raise ValueError("No values provided for NumpySamplerNode")

        # Compute the normalized weights for each value.
        if weights is not None:
            self._weights = np.asarray(weights)
            if len(self._weights) != self._num_values:
                raise ValueError(
                    f"Number of weights ({len(self._weights)}) must match the number "
                    f"of values provided ({self._num_values})."
                )
            self._weights /= np.sum(self._weights)
        else:
            self._weights = None

        super().__init__("uniform", seed=seed, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        rng = rng_info if rng_info is not None else self._rng

        if graph_state.num_samples == 1:
            inds = rng.choice(self._num_values, p=self._weights)
        else:
            inds = rng.choice(self._num_values, size=graph_state.num_samples, p=self._weights)
        results = self.values[inds]
        self._save_results(results, graph_state)

        return results


class GivenValueSelector(FunctionNode):
    """A FunctionNode that selects a single value from a list of parameters.

    Parameters
    ----------
    values : float, list, or numpy.ndarray
        The values that can be selected.
    index : parameter
        The parameter that selects which value to return. This should return an
        integer index corresponding to the position in `values`.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, values, index, **kwargs):
        # The index parameter will automatically be added as input by the FunctionNode constructor.
        super().__init__(self._select, index=index, **kwargs)
        self.values = np.asarray(values)
        if len(values) == 0:
            raise ValueError("No values provided for GivenValueList")

    def _select(self, index):
        """Select the value at the given index."""
        if np.any(index < 0) or np.any(index >= len(self.values)):
            raise IndexError(f"Index {index} out of bounds for values of length {len(self.values)}")
        return self.values[index]


class TableSampler(FunctionNode):
    """A FunctionNode that returns values from a table-like data,
    including a Pandas DataFrame or AstroPy Table. The results returned
    can be in-order (for testing) or randomly selected with replacement.

    Note
    ----
    This is NOT a stateful node. When in_order=True the node will always
    return the first N rows of the table, where N is the number of samples
    requested.

    Parameters
    ----------
    data : pandas.DataFrame, astropy.table.Table, or dict
        The object containing the data to sample.
    in_order : bool
        Return the given data in order of the rows (True). If False, performs
        random sampling with replacement. Default: False

    Attributes
    ----------
    columns : list of str
        The names of the columns in the table.
    data : astropy.table.Table
        The object containing the data to sample.
    in_order : bool
        Return the given data in order of the rows (True). If False, performs
        random sampling with replacement. Default: False
    num_values : int
        The total number of items from which to draw the data.
    """

    def __init__(self, data, in_order=False, **kwargs):
        self.in_order = in_order
        self._last_start_index = -1

        if isinstance(data, dict):
            self.data = Table(data)
        elif isinstance(data, Table):
            self.data = data.copy()
        elif isinstance(data, pd.DataFrame):
            self.data = Table.from_pandas(data)
        else:
            raise TypeError("Unsupported data type for TableSampler.")

        # Check there are some rows.
        self._num_values = len(self.data)
        if self._num_values == 0:
            raise ValueError("No data provided to TableSampler.")

        # Save a list of the column names.
        self.columns = [col for col in self.data.colnames]

        # Initialize the FunctionNode with each column as an output.
        super().__init__(self._non_func, outputs=self.data.colnames, **kwargs)

        # If we are using random sampling, add a random index generator.
        if not self.in_order:
            self.add_parameter(
                "selected_table_index",
                NumpyRandomFunc("integers", low=0, high=self._num_values),
                "The index of the selected row in the table.",
            )

    def __len__(self):
        """Return the number of items in the table."""
        return self._num_values

    def reset(self):
        """Reset the next index to use. Only used for in-order sampling."""
        self.next_ind = 0

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        # Compute the indices to sample.
        if self.in_order:
            start_ind = 0
            if graph_state.sample_offset is not None:
                start_ind += graph_state.sample_offset

            if start_ind == self._last_start_index:
                warnings.warn(
                    "TableSampler in_order sampling called multiple times with the same sample_offset. "
                    "This may indicate unintended behavior, because the same parameter values are used "
                    "multiple times instead of iterating over the table. Consider to set different "
                    "sample_offset values for different objects or chunks."
                )
            self._last_start_index = start_ind

            # Check that we have enough points left to sample.
            end_index = start_ind + graph_state.num_samples
            if end_index > len(self.data):
                raise IndexError(
                    f"TableSampler ran out of entries to sample. Index {end_index} out "
                    f"of bounds for a table with {len(self.data)} entries."
                )

            sample_inds = np.arange(start_ind, end_index)
        else:
            sample_inds = self.get_param(graph_state, "selected_table_index")

        # Parse out each column into a separate parameter with the column name as its name.
        results = []
        for attr_name in self.outputs:
            attr_values = np.asarray(self.data[attr_name][sample_inds])
            results.append(attr_values)

        # Save and return the results.
        self._save_results(results, graph_state)
        return results
