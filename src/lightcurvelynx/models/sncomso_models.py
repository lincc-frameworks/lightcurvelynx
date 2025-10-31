"""Wrappers for the models defined in sncosmo.

https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
https://sncosmo.readthedocs.io/en/stable/models.html
"""

import numpy as np
from astropy import units as u
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.models.physical_model import SEDModel


class SncosmoWrapperModel(SEDModel, CiteClass):
    """A wrapper for sncosmo models.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    Additional parameterized values are used for specific sncosmo models.

    References
    ----------
    * sncosmo - https://zenodo.org/records/14714968
    * Individual models might require citation. See references in the sncosmo documentation.

    Attributes
    ----------
    source : sncosmo.Source
        The underlying source model.
    source_name : str
        The name used to set the source.
    source_param_names : list
        A list of the source model's parameters that we need to set.

    Parameters
    ----------
    source_name : str
        The name used to set the source.
    node_label : str, optional
        An identifier (or name) for the current node.
    wave_extrapolation : WaveExtrapolationModel, optional
        The extrapolation model to use for wavelengths that fall outside
        the model's defined bounds.  If None then the model will use all zeros.
    seed : int, optional
        The seed for a random number generator.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # A class variable for the units so we are not computing them each time.
    _FLAM_UNIT = u.erg / u.second / u.cm**2 / u.AA

    def __init__(
        self,
        source_name,
        node_label=None,
        wave_extrapolation=None,
        seed=None,
        **kwargs,
    ):
        try:
            from sncosmo.models import get_source
        except ImportError as err:
            raise ImportError(
                "sncosmo package is not installed be default. To use the SncosmoWrapperModel, "
                "please install sncosmo. For example, you can install it with "
                "`pip install sncosmo` or `conda install conda-forge::sncosmo`."
            ) from err

        # We explicitly ask for and pass along the PhysicalModel parameters such
        # as node_label and wave_extrapolation so they do not go into kwargs
        # and get added to the sncosmo model below.
        super().__init__(
            node_label=node_label,
            wave_extrapolation=wave_extrapolation,
            seed=seed,
            **kwargs,
        )
        self.source_name = source_name
        self.source = get_source(source_name)

        # Use the kwargs to initialize the sncosmo model's parameters.
        self.source_param_names = []
        for key, value in kwargs.items():
            if key not in self.setters:
                self.add_parameter(key, value, description="Parameter for sncosmo model.")
            if key in self.source.param_names:
                self.source_param_names.append(key)

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.source.param_names

    @property
    def parameter_values(self):
        """Return a list of the model's parameter values."""
        return self.source.parameters

    def minwave(self, graph_state=None):
        """Get the minimum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. Not used
            for this model.

        Returns
        -------
        minwave : float or None
            The minimum wavelength of the model (in angstroms) or None
            if the model does not have a defined minimum wavelength.
        """
        return self.source.minwave()

    def maxwave(self, graph_state=None):
        """Get the maximum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. Not used
            for this model.

        Returns
        -------
        maxwave : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        return self.source.maxwave()

    def _update_sncosmo_model_parameters(self, graph_state):
        """Update the parameters for the wrapped sncosmo model."""
        local_params = graph_state.get_node_state(self.node_string, 0)
        sn_params = {}
        for name in self.source_param_names:
            sn_params[name] = local_params[name]
        self.source.set(**sn_params)

    def get(self, name):
        """Get the value of a specific parameter.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        The parameter value.
        """
        return self.source.get(name)

    def set(self, **kwargs):
        """Set the parameters of the model.

        These must all be constants to be compatible with sncosmo.

        Parameters
        ----------
        **kwargs : dict
            The parameters to set and their values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.set_parameter(key, value)
            else:
                self.add_parameter(key, value, description="Parameter for sncosmo model.")
            if key not in self.source_param_names:
                self.source_param_names.append(key)
        self.source.set(**kwargs)

    def _sample_helper(self, graph_state, seen_nodes, rng_info=None):
        """Internal recursive function to sample the model's underlying parameters
        if they are provided by a function or ParameterizedNode.

        Calls ParameterNode's _sample_helper() then updates the parameters
        for the sncosmo model.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        seen_nodes : dict
            A dictionary mapping nodes seen during this sampling run to their ID.
            Used to avoid sampling nodes multiple times and to validity check the graph.
        num_samples : int
            A count of the number of samples to compute.
            Default: 1
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.

        Raises
        ------
        Raise a ValueError the sampling encounters a problem with the order of dependencies.
        """
        super()._sample_helper(graph_state, seen_nodes, rng_info=rng_info)
        self._update_sncosmo_model_parameters(graph_state)

    def mask_by_time(self, times, graph_state=None):
        """Compute a mask for whether a given time is of interest for a given object.
        For example, a user can use this function to generate a mask to include
        only the observations of interest for a window around the supernova.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        graph_state : GraphState, optional
            An object mapping graph parameters to their values.

        Returns
        -------
        time_mask : numpy.ndarray
            A length T array of Booleans indicating whether the time is of interest.
        """
        if graph_state is None:
            raise ValueError("graph_state needed to compute mask_by_time")

        z = self.get_param(graph_state, "redshift", 0.0)
        if z is None:
            z = 0.0

        t0 = self.get_param(graph_state, "t0", 0.0)
        if t0 is None:
            t0 = 0.0

        # Compute the mask.
        good_times = (times > t0 + self.source.minphase() * (1.0 + z)) & (
            times < t0 + self.source.maxphase() * (1.0 + z)
        )
        return good_times

    def compute_sed(self, times, wavelengths, graph_state=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)
        self._update_sncosmo_model_parameters(graph_state)

        # sncosmo gives an error if the wavelengths are out of bounds, so we need to use
        # extrapolation if the wavelengths are out of bounds.
        if np.any(wavelengths < self.source.minwave()) or np.any(wavelengths > self.source.maxwave()):
            return self.compute_sed_with_extrapolation(times, wavelengths, graph_state, **kwargs)

        # Query the model and convert the output to nJy.
        model_flam = self.source.flux(times - params["t0"], wavelengths)
        model_fnu = flam_to_fnu(
            model_flam,
            wavelengths,
            wave_unit=u.AA,
            flam_unit=self._FLAM_UNIT,
            fnu_unit=u.nJy,
        )
        return model_fnu
