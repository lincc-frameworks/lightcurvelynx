"""Wrapper classes for calling numpy random number generators."""

from os import urandom

import numpy as np

from lightcurvelynx.base_models import FunctionNode


class NumpyRandomFunc(FunctionNode):
    """The base class for numpy random number generators.

    Attributes
    ----------
    func_name : str
        The name of the random function to use.
    _rng : numpy.random._generator.Generator
        This object's random number generator.
    sample_size : tuple
        The shape of the array to generate for each sample. The actual returned value
        will be (num_samples, *size). If an empty tuple will generate a single value per sample.

    Parameters
    ----------
    func_name : str
        The name of the random function to use.
    size : int or tuple, optional
        The shape of the array to generate for each sample. Actual
        returned value will be (num_samples, *size).
        Default: None (single values for each sample)
    seed : int, optional
        The seed to use.

    Notes
    -----
    Since we need to create a new random number generator for this object
    and use that generator's functions, we cannot pass in the function directly.
    Instead we need to pass in the function's name.

    The NumpyRandomFunc node does not support the `choice` function.

    Examples
    --------
    # Create a uniform random number generator between 100.0 and 150.0
    func_node = NumpyRandomFunc("uniform", low=100.0, high=150.0)

    # Create a normal random number generator with mean=5.0 and std=1.0
    func_node = NumpyRandomFunc("normal", loc=5.0, scale=1.0)
    """

    def __init__(self, func_name, size=1, seed=None, **kwargs):
        self.func_name = func_name

        # The node does not support the 'choice' function since it cannot take a list of different
        # lists to use for each sampling run (sample 1 chooses a value from list 1, sample 2 from
        # list 2, etc.).
        if func_name == "choice":
            raise ValueError("The 'choice' function is not supported. Use GivenValueSampler instead.")

        # Convert the given size into a tuple of dimensions or None for a single value per sample.
        if size is None or size == 1:
            self.sample_size = ()
        else:
            # Convert a scalar into a tuple and validate.
            if np.isscalar(size):
                size = (size,)
            if len(size) == 0 or np.any(np.array(size) <= 0):
                raise ValueError(
                    f"Invalid size. Size of output must have at least one dimension and "
                    f"be >= 0 in each dimension. Received {size}."
                )
            self.sample_size = size

        # Get a default random number generator for this object, using the
        # given seed if one is provided.
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        self._rng = np.random.default_rng(seed=seed)

        # Check that the function exists in numpy's random number generator library.
        if not hasattr(self._rng, func_name):
            raise ValueError(f"Random function {func_name} does not exist.")
        func = getattr(self._rng, func_name)
        super().__init__(func, **kwargs)

    def set_seed(self, new_seed):
        """Update the random number generator's seed to a given value.

        Parameters
        ----------
        new_seed : int
            The given seed
        """
        self._rng = np.random.default_rng(seed=new_seed)
        self.func = getattr(self._rng, self.func_name)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Execute the wrapped function.

        The input arguments are taken from the current graph_state and the outputs
        are written to graph_state.

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

        Raises
        ------
        ValueError is func attribute is None.
        """
        args = self._build_inputs(graph_state, **kwargs)

        # If a random number generator is given use that. Otherwise use the default one.
        func = self.func if rng_info is None else getattr(rng_info, self.func_name)

        # Set the size according to the number of samples.
        size_param = (graph_state.num_samples, *self.sample_size)
        if size_param == (1,):
            # If we are generating a single sample use None so it isn't in an array.
            size_param = None

        # Generate the values. Then save and return the results.
        results = func(**args, size=size_param)
        self._save_results(results, graph_state)
        return results
