"""Wrapper classes for sampling from Bilby's prior module."""

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.base_models import FunctionNode


class BilbyPriorNode(FunctionNode, CiteClass):
    """The base class for sampling from Bilby's prior module.

    Attributes
    ----------
    prior : bilby.prior.PriorDict
        The Bilby prior object to sample from.

    Parameters
    ----------
    prior : dict or bilby.prior.PriorDict
        A dictionary mapping the names of the parameters to their prior distributions.

    Citations
    ---------
    Cite as::

          @article{bilby_paper,
              author = "Ashton, Gregory and others",
              title = "{BILBY: A user-friendly Bayesian inference library for gravitational-wave astronomy}",
              eprint = "1811.02042",
              archivePrefix = "arXiv",
              primaryClass = "astro-ph.IM",
              doi = "10.3847/1538-4365/ab06fc",
              journal = "Astrophys. J. Suppl.",
              volume = "241",
              number = "2",
              pages = "27",
              year = "2019"
          }
    """

    def __init__(self, prior, **kwargs):
        # Set the prior. If the prior is provided as a dictionary, convert it to a Bilby
        # PriorDict object.
        if isinstance(prior, dict):
            try:
                from bilby.core.prior import PriorDict
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "Bilby package is not installed by default. To use the bilby priors, "
                    "please install it. For example, you can install it with `pip install bilby`."
                ) from err
            prior = PriorDict(prior)
        if len(prior) == 0:
            raise ValueError("The provided prior is empty.")
        self.prior = prior

        # Set the outputs to be the names of the parameters in the prior.
        outputs = [param for param in prior]
        super().__init__(self._non_func, outputs=outputs, **kwargs)

    def _set_seed(self, new_seed):
        """Update the random number generator's seed to a given value.

        Parameters
        ----------
        new_seed : int
            The given seed
        """
        # Import bilby to set the seed.
        try:
            from bilby.core.utils import random as bibly_random
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "Bilby package is not installed by default. To use the bilby priors, "
                "please install it. For example, you can install it with `pip install bilby`."
            ) from err
        bibly_random.seed(new_seed)

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
            This random number generator is not used by this node. Instead, Bilby's internal
            random number generator is used. You can set the seed for this using the `seed`
            argument when initializing the node or by calling the `set_seed()` method.
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
        # Set the seed for Bilby's internal random number generator using the provided RNG.
        if rng_info is None:
            rng_info = np.random.default_rng()
        self._set_seed(rng_info.integers(0, 2**32 - 1))

        # Sample from the Bilby prior model and extract the parameters from the dictionary.
        param_dict = self.prior.sample(graph_state.num_samples)
        results = []
        for key in self.outputs:
            values = param_dict[key]
            if graph_state.num_samples == 1:
                values = values[0]
            graph_state.set(self.node_string, key, values)
            results.append(values)
        return results
