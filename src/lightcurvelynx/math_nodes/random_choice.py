"""The RandomChoiceNode allows the user to randomly select one of the values
from a given list of parameters.
"""

import numpy as np

from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.math_nodes.given_sampler import GivenValueSampler


class RandomChoiceNode(FunctionNode):
    """A FunctionNode that returns randomly selected parameters from a given list
    with replacement. This is a version of the GivenValueSampler that is designed to
    work with parameterized (chained) inputs.

    Parameters
    ----------
    values : list-like of parameters
        The list of input parameters from which to randomly select.
    weights : list-like of float, optional
        The weights corresponding to each value. If not provided, all values
        are equally likely.
        Default: None
    seed : int, optional
        The seed for the random number generator. If not provided, the node will
        use a random seed.
        Default: None
    """

    def __init__(self, values, *, weights=None, seed=None, **kwargs):
        super().__init__(self._non_func, **kwargs)

        self._num_values = len(values)
        if self._num_values == 0:
            raise ValueError("No values provided for RandomChoiceNode")

        # Register each value as a parameter to allow chaining.
        self._param_names = []
        for i, value in enumerate(values):
            param_name = f"input_{i}"
            self.add_parameter(param_name, value, f"Value {i} that can be randomly selected.")
            self._param_names.append(param_name)

        # Create a parameter for the chosen index (to save the state on which input was selected).
        self.add_parameter(
            "selected_index",
            GivenValueSampler(self._num_values, weights=weights, seed=seed),
        )

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
        params = self.get_local_params(graph_state)

        # Use the pre-selected indices from the "selected_index" parameter
        # to determine which values to return.
        inds = params["selected_index"]

        if graph_state.num_samples == 1:
            results = params[self._param_names[inds]]
        else:
            # We use a list comprehension to select the appropriate value for each sample
            # based on the selected index. We do this instead of iterating over the names
            # and using a mask because it cleaner for type inference for the results array.
            results = np.array([params[self._param_names[j]][i] for i, j in enumerate(inds)])

        self._save_results(results, graph_state)

        return results
