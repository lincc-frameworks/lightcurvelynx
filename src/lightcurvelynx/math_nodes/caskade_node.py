"""Wrapper class for an initial interface that allows caskade to pass a module
full of parameters to LightCurveLynx.

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
            print(f"Processing parameter: {key}")
            values = param.npvalue
            if not param.batched:
                print(f"Parameter {key} is not batched.")
                # If there is no batching, treat this as a constant value for all samples
                if len(values.shape) == 0:
                    # Get rid of 0-dimensional array by converting it to a scalar.
                    values = values.item()

                # If we need more than one sample, repeat the values to match the number
                # of samples.
                if graph_state.num_samples > 1:
                    values = np.repeat(
                        np.expand_dims(values, axis=0),
                        graph_state.num_samples,
                        axis=0,
                    )
            else:
                # If is a batched parameter, we want to extract the appropriate slice.
                start_idx = graph_state.sample_offset
                end_idx = start_idx + graph_state.num_samples
                if end_idx > param.batch_shape[0]:
                    raise ValueError(
                        f"Parameter {key} has batch size {param.batch_shape[0]} but requested slice "
                        f"from {start_idx} to {end_idx} exceeds the batch size."
                    )
                values = param.npvalue[start_idx:end_idx]

            # Values should be the correct shape now. Add it to our results list.
            results.append(values)

        # Save and return the values.
        self._save_results(results, graph_state)
        return results


class CaskadeFunctionNode(FunctionNode, CiteClass):
    """The base class for sampling from the function of a Caskade module.
    https://github.com/ConnorStoneAstro/caskade

    Note
    ----
    The caskade package is not installed by default. To use the caskade functionality, users will
    need to install the caskade package separately with `pip install caskade`.

    Parameters
    ----------
    caskade_node : ck.Module
        The Caskade node whose parameters are being accessed.
    function_name : str
        The name of the function within the Caskade module to be wrapped.
    **kwargs : dict, optional
        Any additional keyword arguments.

    References
    ----------
    * Stone et al., (2025). caskade: building Pythonic scientific simulators.
      Journal of Open Source Software, 10(113), 8786,
      https://doi.org/10.21105/joss.08786
    * https://github.com/ConnorStoneAstro/caskade
    """

    def __init__(self, caskade_module, function_name, **kwargs):
        # Keep a reference to the original Caskade node
        self._node = caskade_module
        self._function_name = function_name
        self._func = getattr(caskade_module, function_name, None)
        if self._func is None:
            raise ValueError(f"Failed to retrieve the function '{function_name}' from the Caskade module.")
        super().__init__(self._func, outputs=[function_name], **kwargs)
