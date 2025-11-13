"""A wrapper around pzflow sampling.

For the full pzflow package see:
https://github.com/jfcrenshaw/pzflow
"""

import numpy as np
import pandas as pd
from citation_compass import CiteClass

from lightcurvelynx.base_models import FunctionNode


class PZFlowNode(FunctionNode, CiteClass):
    """A node that wraps sampling from pzflow.

    References
    ----------
    * Paper: Crenshaw et. al. 2024 - https://ui.adsabs.harvard.edu/abs/2024AJ....168...80C
    * Zenodo: Crenshaw et. al. 2024 - https://doi.org/10.5281/zenodo.10710271

    Attributes
    ----------
    flow : pzflow.flow.Flow or pzflow.flowEnsemble.FlowEnsemble
        The object from which to sample.
    columns : list of str
        The column names for the output columns.
    conditional_cols : list of str
        The names of the conditional columns used when sampling the flow.
    """

    def __init__(self, flow_obj, node_label=None, **kwargs):
        self.flow = flow_obj

        # Add each of the flow's data columns as an output parameter.
        self.columns = [x for x in flow_obj.data_columns]

        # Add each of the flow's conditional columns as an input parameter and
        # check that the column is include in the kwargs (and thus has an input).
        if flow_obj.conditional_columns is not None and len(flow_obj.conditional_columns) > 0:
            self.conditional_cols = []
            for col in flow_obj.conditional_columns:
                if col not in kwargs:
                    raise ValueError(f"Missing input parameter '{col}' needed by flow model.")
                self.conditional_cols.append(col)
        else:
            self.conditional_cols = None

        super().__init__(self._non_func, node_label=node_label, outputs=self.columns, **kwargs)

    @classmethod
    def from_file(cls, filename, node_label=None, **kwargs):
        """Create a PZFlowNode from a saved flow file.

        Parameters
        ----------
        filename : str or Path
            The location of the saved flow.
        node_label : str
            An optional human readable identifier (name) for the current node.
        **kwargs : dict, optional
            Additional function arguments, including the input parameters for the flow.
        """
        try:
            from pzflow import Flow
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "pzflow package is not installed be default. You can install it with "
                "`pip install pzflow` or `conda install conda-forge::pzflow`."
            ) from err

        flow_to_use = Flow(file=filename)
        return PZFlowNode(flow_to_use, node_label=node_label, **kwargs)

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
        # If a random number generator is used, use that to compute the seed.
        seed = None if rng_info is None else int.from_bytes(rng_info.bytes(4), byteorder="big")

        # If the flow has conditional columns, extract them from the GraphState and use
        # them to sample the flow (with num_samples=1).
        if self.conditional_cols is not None and len(self.conditional_cols) > 0:
            all_params = self.get_local_params(graph_state)
            input_params = {}
            for col in self.conditional_cols:
                input_params[col] = all_params[col]
            input_df = pd.DataFrame(input_params)

            samples = self.flow.sample(1, conditions=input_df, seed=seed)
        else:
            samples = self.flow.sample(graph_state.num_samples, seed=seed)

        # Parse out each output column in the flow samples as its own result vector.
        results = []
        for attr_name in self.flow.data_columns:
            attr_values = samples[attr_name].values
            if graph_state.num_samples == 1:
                results.append(attr_values[0])
            else:
                results.append(np.array(attr_values))

        # Save and return the results. If we only have a single output column,
        # return that directly.
        if len(self.flow.data_columns) == 1:
            results = results[0]
        self._save_results(results, graph_state)
        return results
