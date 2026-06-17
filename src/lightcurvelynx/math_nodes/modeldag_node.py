"""Wrapper class for sampling from the optional `modeldag` package, which
provides tools for defining and sampling a directed acyclic graph (DAG) of
parameters.

This code uses the `modeldag` package (https://github.com/MickaelRigault/modeldag), which
is not installed by default. Users who want to use this functionality must
install it separately, e.g. `pip install modeldag`.
"""

from citation_compass import CiteClass

from lightcurvelynx.base_models import FunctionNode


class ModelDAGNode(FunctionNode, CiteClass):
    """The base class for sampling from the modeldag package:
    https://github.com/MickaelRigault/modeldag

    Note
    ----
    The modeldag package is not installed by default. To use the modeldag functionality, users will
    need to install the modeldag package separately with `pip install modeldag`.

    Attributes
    ----------
    model : modeldag.ModelDAG or dict

    Parameters
    ----------
    model : modeldag.ModelDAG or dict
        The modeldag object to sample from or a dictionary representing the modeldag. If a dictionary
        is provided, it will be converted to a modeldag.ModelDAG object.

    References
    ----------
    https://github.com/MickaelRigault/modeldag
    """

    def __init__(self, model, **kwargs):
        # Set the model. If the model is provided as a dictionary, convert it to a modeldag.ModelDAG object.
        if isinstance(model, dict):
            try:
                from modeldag import ModelDAG
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "modeldag package is not installed by default. To use the modeldag functionality, "
                    "please install it. For example, you can install it with `pip install modeldag`."
                ) from err
            model = ModelDAG(model)
        self.model = model

        # Set the outputs to be the names of the parameters in the model.
        outputs = [param for param in self.model.entries]
        super().__init__(self._non_func, outputs=outputs, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Sample from the wrapped prior.

        The input arguments are taken from the current graph_state and the outputs
        are written to graph_state.

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

        Raises
        ------
        ValueError
            if func attribute is None.
        """
        # Sample from the modeldag model.
        data_table = self.model.draw(size=graph_state.num_samples, rng=rng_info)

        # Extract all the results from the data table and save them to the graph state.
        results = []
        for key in self.outputs:
            values = data_table[key]
            if graph_state.num_samples == 1:
                values = values[0]
            results.append(values)
        self._save_results(results, graph_state)

        return results
