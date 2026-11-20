"""Wrapper class for interfacing with the caskade package, which provides tools for
defining and sampling a directed acyclic graph (DAG) of parameters.

This code uses the `caskade` package (https://github.com/ConnorStoneAstro/caskade),
which is not installed by default. Users who want to use this functionality must
install it separately, e.g. `pip install caskade`.
"""

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.base_models import FunctionNode


class CaskadeParameterNode(FunctionNode, CiteClass):
    """The base class for sampling from the caskade package:
    https://github.com/ConnorStoneAstro/caskade

    Note
    ----
    The caskade package is not installed by default. To use the caskade functionality, users will
    need to install the caskade package separately with `pip install caskade`.

    Parameters
    ----------
    caskade_node : ck.Module
        The Caskade node whose parameters are being accessed.
    **kwargs : dict, optional
        Any additional keyword arguments.

    References
    ----------
    * Stone et al., (2025). caskade: building Pythonic scientific simulators.
      Journal of Open Source Software, 10(113), 8786,
      https://doi.org/10.21105/joss.08786
    * https://github.com/ConnorStoneAstro/caskade
    """

    def __init__(self, caskade_node, **kwargs):
        # Keep a reference to the original Caskade node
        self._node = caskade_node

        # Set the outputs to be the names of all the parameters in the model.
        outputs = [param.name for param in caskade_node.all_params]
        super().__init__(self._non_func, outputs=outputs, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Sample the parameters from their value in the Caskade node.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A numpy random number generator to forward to `modeldag.ModelDAG.draw`.
            If not provided, `modeldag` will use its default randomness source.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        # Get all of the current parameter objects.
        param_data = {param.name: param for param in self._node.all_params}

        # Extract the current values of the parameter from the caskade node.
        results = []
        for key in self.outputs:
            if key not in param_data:
                raise ValueError(f"Parameter {key} not found in the Caskade node.")

            # Get the values of the current parameter.
            param = param_data[key]
            values = param.npvalue
            batch_size = 1 if not param.batched else param.batch_size
            print(f"Parameter {key}: values = {values}, batch_size = {batch_size}")

            if graph_state.num_samples == 1:
                # If we need a single sample, check the batch size is 1 and extract the
                # single value (from a 0-dimensional array) if necessary.
                if batch_size != 1:
                    raise ValueError(f"Parameter {key} has batch size {batch_size} but num_samples is 1")
                if len(values.shape) == 0:
                    values = values.item()
            elif batch_size == 1:
                # If we have a batch size of 1 and need more samples, repeat the values
                # while ensuring the first dimension is num_samples.
                values = np.repeat(np.expand_dims(values, axis=0), graph_state.num_samples, axis=0)
            elif batch_size != graph_state.num_samples:
                # If we have a batch size that does not match the number of samples, raise an error.
                raise ValueError(
                    f"Parameter {key} has batch size {batch_size} but num_samples "
                    f"is {graph_state.num_samples}"
                )

            # Values should be the correct shape now. Add it to our results list.
            results.append(values)

        # Save and return the values.
        self._save_results(results, graph_state)
        return results
