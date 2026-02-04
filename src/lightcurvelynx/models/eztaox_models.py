"""Wrappers for the models defined in EZTaoX.

https://github.com/LSST-AGN-Variability/EzTaoX
"""
import importlib

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.physical_model import BandfluxModel


class EzTaoXWrapperModel(BandfluxModel, CiteClass):
    """A wrapper for an eztaox model.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    Additional parameterized values are used for specific eztaox models.

    References
    ----------
    * Weixiang Yu et al., 2025 “Scalable and Robust Multiband Modeling of AGN Light Curves in Rubin-LSST"
      DOI: 10.48550/arXiv.2511.21479

    Attributes
    ----------
    kernel : eztaox kernel object
        An eztaox kernel object to use for the Gaussian process modeling of the light curve.
    num_kernel_params : int
        The number of kernel parameters for the eztaox kernel.
    num_filters : int
        The number of filters in the model.
    zero_mean : bool
        Whether to use a zero mean model.
    has_lag : bool
        Whether the model includes lag parameters.
    filter_idx : dict
        A mapping from filter name to an integer index. If not provided,
        the default mapping for ugrizy filters is used.

    Parameters
    ----------
    kernel : eztaox kernel object
        An eztaox kernel object to use for the Gaussian process modeling of the light curve.
    log_kernel_param : list of setters, required
        Setters for each of the log kernel parameters. These must be in the order expected by
        the kernel functions.
    amp_scale_func : Callable, optional
        A callable amplitude scaling function, defaults to None.
    log_amp_scale : list of setters, optional
        Setters for the log amplitude scale for each filter (length N) if amp_scale_func is
        not provided.
        Default is None.
    zero_mean : bool
        Whether to use a zero mean model. If False then the program will try to use (in order):
        mean_func, mean (values), or a default mean function.
        Default is True.
    mean_func : Callable, optional
        If provided and zero_mean is False this is used to compute the mean function
        for bands.
        Default is None.
    mean_mag : list of setters, optional
        Setters for the mean magnitude for each filter except the first (length N-1) if
        zero_mean is False and mean_func is not provided.
        Default is None.
    has_lag : bool
        Whether the model includes lag parameters. If True then the program will try to
        use (in order): lag_func, lag (values), or a default lag function.
        Default is False.
    lag_func : Callable, optional
        If provided and has_lag is True this is used to compute the lag.
        Default is None.
    lag : list of setters, optional
        Setters for the lag for each filter except the first (length N) if has_lag is True
        and lag_func is not provided.
        Default is None.
    filter_idx : dict, optional
        A mapping from filter name to an integer index. If not provided,
        the default mapping for ugrizy filters is used.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # Convenience mapping from filter name to index in the parameter list.
    _default_filter_idx = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    def __init__(
        self,
        kernel,
        *,
        log_kernel_param=None,
        amp_scale_func=None,
        log_amp_scale=None,
        zero_mean=True,
        mean_func=None,
        mean_mag=None,
        has_lag=False,
        lag_func=None,
        lag=None,
        filter_idx=None,
        seed_param=None,
        **kwargs,
    ):
        self._cached_data = {}
        super().__init__(**kwargs)

        # Confirm that the needed packages are installed.
        if importlib.util.find_spec("eztaox") is None:
            raise ImportError(
                "The EzTaoX package is required to use the EzTaoXWrapperModel. "
                "Please install it from https://github.com/LSST-AGN-Variability/EzTaoX"
            )
        if importlib.util.find_spec("jax") is None:
            raise ImportError(
                "JAX is required to use the EzTaoXWrapperModel class, please "
                "install with `pip install jax` or `conda install conda-forge::jax`"
            )

        # Store the kernel and filter index mapping.
        self.kernel = kernel
        self.filter_idx = filter_idx if filter_idx is not None else self._default_filter_idx
        self.num_filters = len(self.filter_idx)

        # Add the log kernel parameters to this model node. Since the order is
        # the important aspect, we just name them by index.
        if log_kernel_param is None:
            raise ValueError("The log_kernel_param parameter setters must be provided.")
        self.num_kernel_params = len(log_kernel_param)
        for i, setter in enumerate(log_kernel_param):
            self.add_parameter(f"eztaox_log_kernel_param_{i}", setter)

        # Add the amplitude scale information as either a function or parameters to this node.
        self._amp_scale_func = amp_scale_func
        if log_amp_scale is None and amp_scale_func is None:
            raise ValueError("One of either amp_scale_func or log_amp_scale must be provided.")
        self._has_log_amp_scale = log_amp_scale is not None
        if self._has_log_amp_scale:
            for i, setter in enumerate(log_amp_scale):
                self.add_parameter(f"eztaox_log_amp_scale_{i}", setter)
            if len(log_amp_scale) != self.num_filters:
                raise ValueError(
                    f"The number of log amplitude scale parameter setters {len(log_amp_scale)} "
                    f"must be equal to the number of filters {self.num_filters}."
                )

        # Store the mean magnitude parameters and callable if provided.
        self.zero_mean = zero_mean
        self._mean_func = mean_func  # The callable if it is provided.
        self._has_mean_vals = mean_mag is not None
        if self._has_mean_vals:
            if len(mean_mag) != self.num_filters - 1:
                raise ValueError(
                    f"The number of mean parameter setters {len(mean_mag)} must be equal to the "
                    f"number of filters minus one {self.num_filters - 1}."
                )
            for i, setter in enumerate(mean_mag):
                self.add_parameter(f"eztaox_band_mean_{i}", setter)

        # Store the lag parameters if provided.
        self.has_lag = has_lag
        self._lag_func = lag_func  # The callable if it is provided.
        self._has_lag_values = lag is not None
        if self._has_lag_values:
            if len(lag) != self.num_filters:
                raise ValueError(
                    f"The number of lag parameter setters {len(lag)} must be equal to the "
                    f"number of filters {self.num_filters}."
                )
            for i, setter in enumerate(lag):
                self.add_parameter(f"eztaox_band_lag_{i}", setter)

        # The seed used per run can be defined by the seed_param. If is not provided,
        # we randomly generate a seed for each run.
        if seed_param is None:
            seed_param = NumpyRandomFunc("integers", low=0, high=2**32 - 1)
        self.add_parameter("eztaox_seed_param", seed_param)

    def clear_cache(self):
        """Clear any cached data for this model node."""
        self._cached_data = {}

    def _compute_all_bandfluxes(self, times, filters, state):
        """Evaluate the model at the passband level for a single, given graph state and
        and all of the filters at once. We do this and cache the results to avoid
        recomputing the same model for each band.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : list of str
            The names of the filters (one at each time).
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
            This is not used in this model, but is required for the function signature.

        Returns
        -------
        dict
            A dictionary of cached data with the filters, times, and computed bandfluxes.
        """
        if "bandfluxes" in self._cached_data:
            # Basic error checking that we do not have a dirty cache.
            if not np.array_equal(self._cached_data["filters"], filters):
                raise ValueError("Dirty cache detected in _compute_all_bandfluxes.")
            if not np.array_equal(self._cached_data["times"], times):
                raise ValueError("Dirty cache detected in _compute_all_bandfluxes.")
            return self._cached_data

        # Cache the input data (times and filters).
        self._cached_data = {}  # Clear the cache (just in case).
        self._cached_data["times"] = times
        self._cached_data["filters"] = filters

        # Import the dependencies that we need for this computation (these have all
        # been checked in the constructor).
        import jax
        import jax.numpy as jnp
        from eztaox.simulator import MultiVarSim

        # Extract the local parameters for this object from the full state object and build
        # the parameter dict needed by the simulator. This parameter dict must include:
        # - log_kernel_params a JAX numpy array of shape (num_kernel_params,)
        # - log_amp_scales a JAX numpy array of shape (num_filters,)
        # Optional it may also include:
        # - band_means a JAX numpy array of shape (num_filters - 1,)
        # - band_lags a JAX numpy array of shape (num_filters,)
        local_params = self.get_local_params(state)
        init_params = {}
        init_params["log_kernel_params"] = jnp.array(
            [local_params[f"eztaox_log_kernel_param_{i}"] for i in range(self.num_kernel_params)]
        )
        if self._has_log_amp_scale:
            init_params["log_amp_scales"] = jnp.array(
                [local_params[f"eztaox_log_amp_scale_{i}"] for i in range(self.num_filters)]
            )
        if not self._has_mean_vals:
            init_params["band_means"] = jnp.array(
                [local_params[f"eztaox_band_mean_{i}"] for i in range(self.num_filters - 1)]
            )
        if self._has_lag_values:
            init_params["band_lags"] = jnp.array(
                [local_params[f"eztaox_band_lag_{i}"] for i in range(self.num_filters)]
            )

        # Create the simulator object using the given parameters and run it.
        sim = MultiVarSim(
            self.kernel,
            0.01,
            np.max(times),  # The last time to simulate.
            self.num_filters,
            init_params=init_params,
            mean_func=self._mean_func,
            amp_scale_func=self._amp_scale_func,
            lag_func=self._lag_func,
            zero_mean=self.zero_mean,
            has_lag=self.has_lag,
        )

        # Compute the list of bands as integer indices for the simulator and save them to the cache.
        band_indices = jnp.array([self.filter_idx[f] for f in self.filter_idx])

        # Compute the list of magnitudes for all given times, transform them to fluxes,
        # save them in the cache, and return them.
        mags = sim.fixed_input_fast(
            jnp.asarray(times),
            band_indices,
            jax.random.PRNGKey(local_params["eztaox_seed_param"]),  # Use the per-run seed.
        )
        bandfluxes = mag2flux(np.asarray(mags))
        self._cached_data["bandfluxes"] = bandfluxes
        return self._cached_data

    def compute_bandflux(self, times, filter, state):
        """Evaluate the model at the passband level for a single, given graph state and filter.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filter : str
            The name of the filter.
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
            This is not used in this model, but is required for the function signature.

        Returns
        -------
        bandflux : numpy.ndarray
            A length T array of band fluxes for this model in this filter.
        """
        bandflux_data = self._compute_all_bandfluxes(times, filter, state)
        filter_mask = bandflux_data["filters"] == filter
        return bandflux_data["bandfluxes"][filter_mask]

    def evaluate_bandfluxes(self, passband_or_group, times, filters, state, rng_info=None) -> np.ndarray:
        """Get the band fluxes for a given Passband or PassbandGroup.

        Parameters
        ----------
        passband_or_group : Passband or PassbandGroup
            The passband (or passband group) to use.
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : numpy.ndarray or None
            A length T array of filter names. It may be None if
            passband_or_group is a Passband.
        state : GraphState
            An object mapping graph parameters to their values.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A matrix of the band fluxes. If only one sample is provided in the GraphState,
            then returns a length T array. Otherwise returns a size S x T array where S is the
            number of samples in the graph state.
        """
        # Check that we do not have a dirty cache.
        self.clear_cache()

        # Do the normal computation (relying on the cached bandfluxes).
        bandfluxes = super().evaluate_bandfluxes(passband_or_group, times, filters, state, rng_info)

        # Clear the cache.
        self.clear_cache()

        return bandfluxes
