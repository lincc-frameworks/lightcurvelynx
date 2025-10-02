"""Wrapper classes for some of scipy's sampling functions."""

from os import urandom

import numpy as np
import scipy.stats
from scipy.stats.sampling import NumericalInversePolynomial

from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.graph_state import transpose_dict_of_list


class NumericalInversePolynomialFunc(FunctionNode):
    """A class for sampling from scipy's NumericalInversePolynomial
    given a distribution function, an object with a pdf function,
    or a class from which to create such an object.

    Note
    ----
    If a class is provided, then the sampling function will create a new
    object (with the sampled parameters) for each sampling. This is very expensive.

    Attributes
    ----------
    _dist : object or class
        An object or class with either a pdf() or logpdf() method that defines
        the distribution from which to sample.
    _inv_poly: scipy.stats.sampling.NumericalInversePolynomial
        The scipy object to use for sampling. Set to None if _dist is a class.
    _vect_sample : numpy.vectorize
        The vectorized function to create a distribution from a class and sample it.
        Set to None if _dist is an object.
    _rng : numpy.random._generator.Generator
        This object's random number generator.

    Parameters
    ----------
    dist : object or class
        An object or class with either a pdf() or logpdf() method that defines
        the distribution from which to sample.
    seed : int, optional
        The seed to use.
    """

    def __init__(self, dist=None, seed=None, **kwargs):
        # Check that the distribution object/class has a pdf or logpdf function
        # or that we have provided a function directly.
        if not hasattr(dist, "pdf") and not hasattr(dist, "logpdf"):
            raise ValueError("Distribution must have either pdf() or logpdf().")
        self._dist = dist

        # Classes show up as type="type". In this case we will need to create
        # a concrete object from the class and any given parameters.
        if isinstance(dist, type):
            self._inv_poly = None
            self._vect_sample = np.vectorize(self._create_and_sample)
        else:
            self._inv_poly = NumericalInversePolynomial(self._dist)
            self._vect_sample = None

        # Get a default random number generator for this object, using the
        # given seed if one is provided.
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        self._rng = np.random.default_rng(seed=seed)

        # Set the function and add all the kwargs as parameters.
        super().__init__(self._rvs, **kwargs)

    def _rvs(self):
        """A place holder function to use for object naming."""
        pass

    def set_seed(self, new_seed):
        """Update the random number generator's seed to a given value.

        Parameters
        ----------
        new_seed : int
            The given seed
        """
        self._rng = np.random.default_rng(seed=new_seed)

    def _create_and_sample(self, args, rng):
        """Create the distribution function and sample it. This is only
        needed if our distribution is in the form of a class that must
        be instantiated with different parameters each sampling run.

        Parameters
        ----------
        args : dict
            A dictionary mapping argument name to individual values.
        rng : numpy.random._generator.Generator
            The random number generator to use.

        Returns
        -------
        sample : float
            The result of sampling the function.
        """
        dist = self._dist(**args)
        sample = NumericalInversePolynomial(dist).rvs(1, rng)[0]
        return sample

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
        """
        rng = rng_info if rng_info is not None else self._rng

        if self._inv_poly is not None:
            # Batch sample all the results.
            num_samples = None if graph_state.num_samples == 1 else graph_state.num_samples
            results = self._inv_poly.rvs(num_samples, rng)
        else:
            # This is a class so we will need to create a new distribution object
            # for each sample (with a single instance of the input parameters).
            args = self._build_inputs(graph_state, **kwargs)

            if graph_state.num_samples == 1:
                dist = self._dist(**args)
                results = NumericalInversePolynomial(dist).rvs(1, rng)[0]
            else:
                # Transpose the dict of arrays to a list of dicts.
                arg_list = transpose_dict_of_list(args, graph_state.num_samples)
                results = self._vect_sample(arg_list, rng)

        # Save and return the results.
        self._save_results(results, graph_state)
        return results


class PDFFunctionWrapper:
    """A class that just wraps a given PDF function.

    Attributes
    ----------
    _pdf : function
        The PDF function.
    """

    def __init__(self, func):
        self._pdf = func
        self.pdf = self._pdf


class SamplePDF(NumericalInversePolynomialFunc):
    """A node for sampling from a given PDF function.

    Parameters
    ----------
    dist : function, class, or object
        The pdf function from which to sample or a class/object with that function.
    """

    def __init__(self, dist, **kwargs):
        if hasattr(dist, "pdf"):
            self.dist_obj = dist
        elif callable(dist):
            self.dist_obj = PDFFunctionWrapper(dist)
        else:
            raise ValueError("No pdf function detected.")
        super().__init__(self.dist_obj, **kwargs)


class LogPDFFunctionWrapper:
    """A class that just wraps a given Log PDF function.

    Attributes
    ----------
    _log_pdf : function
        The log PDF function.
    """

    def __init__(self, func):
        self._log_pdf = func
        self.logpdf = self._log_pdf


class SampleLogPDF(NumericalInversePolynomialFunc):
    """A node for sampling from a given Log PDF function.

    Parameters
    ----------
    dist : function, class, or object
        The pdf function from which to sample or a class/object with that function.
    """

    def __init__(self, dist, **kwargs):
        if hasattr(dist, "logpdf"):
            self.dist_obj = dist
        elif callable(dist):
            self.dist_obj = LogPDFFunctionWrapper(dist)
        else:
            raise ValueError("No logpdf function detected.")
        super().__init__(self.dist_obj, **kwargs)


class ScipyRandomDist(FunctionNode):
    """The base class sampling from scipy.stats distributions.

    Attributes
    ----------
    dist_class : scipy.stats distribution object
        The distribution from which to sample.
    _rng : numpy.random._generator.Generator
        This object's random number generator.
    sample_size : tuple
        The shape of the array to generate for each sample. The actual returned value
        will be (num_samples, *size). If an empty tuple will generate a single value per sample.
    params_names : list of str
        The names of the parameters used to create the distribution object.

    Parameters
    ----------
    dist_name : str
        The name of the distribution from which to sample.
    seed : int, optional
        The seed to use.
    node_label : str, optional
        An optional label for the node. If not provided, a default label
        will be created based on the function name and parameters.
    **kwargs : dict, optional
        The parameters to use to create the distribution object.
    """

    def __init__(self, dist_name, seed=None, node_label=None, **kwargs):
        super().__init__(self._non_func, node_label=node_label)

        # Create the template of the distribution class and save the parameters.
        if not hasattr(scipy.stats, dist_name):
            raise ValueError(f"The distribution {dist_name} is not found in scipy.stats.")
        self.dist_class = getattr(scipy.stats, dist_name)

        self.params_names = []
        for key, val in kwargs.items():
            self.add_parameter(key, val)
            self.params_names.append(key)

        # Get a default random number generator for this object, using the
        # given seed if one is provided.
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        self._rng = np.random.default_rng(seed=seed)

    def set_seed(self, new_seed):
        """Update the random number generator's seed to a given value.

        Parameters
        ----------
        new_seed : int
            The given seed
        """
        self._rng = np.random.default_rng(seed=new_seed)

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
        # If a random number generator is given use that. Otherwise use the default one.
        rng = rng_info if rng_info is not None else self._rng

        # Set the size according to the number of samples.
        size_param = graph_state.num_samples if graph_state.num_samples > 1 else None

        # Create the distribution object using the sampled parameters.
        param_values = {name: self.get_param(graph_state, name) for name in self.params_names}
        print(param_values)
        dist = self.dist_class(**param_values)

        # Generate the values. Then save and return the results.
        results = dist.rvs(size=size_param, random_state=rng)
        self._save_results(results, graph_state)
        return results
