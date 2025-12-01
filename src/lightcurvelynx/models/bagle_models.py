"""Wrappers for the models defined in bagle.

https://github.com/MovingUniverseLab/BAGLE_Microlensing
"""

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.models.physical_model import BandfluxModel


class BagleWrapperModel(BandfluxModel, CiteClass):
    """A wrapper for bagle models.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    Additional parameterized values are used for specific bagle models.

    References
    ----------
    * Lu et al., “The BAGLE Python Package for Bayesian Analysis of Gravitational Lensing Events”,
      AAS Journals, submitted
    * Bhadra et al., “Modeling Binary Lenses and Sources with the BAGLE Python Package”, AAS Journals,
      submitted
    * Chen et al., “Adjusting Gaussian Process Priors for BAGLE's Gravitational Microlensing Model Fits”,
      in prep.

    Parameters
    ----------
    model_info : str or class
        The name of the bagle model class to use in the simulation or the class itself.
    parameter_dict : dict
        A dictionary of parameter names and values to use for the model. The keys should
        match the parameter names expected by the bagle model.
    filter_idx : dict, optional
        A mapping from filter names to indices expected by the bagle model. If not provided,
        a default mapping for ugrizy filters will be used.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # Convenience mapping from filter name to index in the parameter list.
    _default_filter_idx = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    def __init__(self, model_info, parameter_dict, filter_idx=None, **kwargs):
        # We start by extracting the parameter information needed for a general physical model.
        # We check the parameter dictionary first, falling back to kwargs if needed.
        ra = parameter_dict.get("raL", None)
        if "ra" in kwargs:
            if ra is None:
                ra = kwargs.pop("ra")
            else:
                raise ValueError(
                    "The 'ra' parameter is specified in both parameter_dict (as 'raL') "
                    " and kwargs (as 'ra'). Please only use the parameter_dict."
                )

        dec = parameter_dict.get("decL", None)
        if "dec" in kwargs:
            if dec is None:
                dec = kwargs.pop("dec")
            else:
                raise ValueError(
                    "The 'dec' parameter is specified in both parameter_dict (as 'decL') "
                    " and kwargs (as 'dec'). Please only use the parameter_dict."
                )

        # The t0 parameter can be specified different ways from the bagle model,
        # including being computed from other parameters. We need a base value to use
        # for the time bounds, so we use t0 if it is in the parameter_dict or otherwise
        # anchor on any parameter that starts with "t0" (under the assumption that it will
        # indicate the same general range of times).
        t0 = parameter_dict.get("t0", None)
        if t0 is None:
            for key, value in parameter_dict.items():
                if key.startswith("t0"):
                    t0 = value
                    break
        if "t0" in kwargs:
            raise ValueError(
                "The 't0' parameter must be specified in only the 'parameter_dict' for the BagleWrapperModel."
            )

        super().__init__(ra=ra, dec=dec, t0=t0, **kwargs)

        # Add all of the parameters in the dictionary as settable parameters (if they are not
        # already set by the parent class) and save their names (in order) for later use.
        self._parameter_names = []
        for param_name, param_value in parameter_dict.items():
            self._parameter_names.append(param_name)
            if param_name in self.list_params():
                self.set_parameter(param_name, param_value)
            else:
                self.add_parameter(param_name, param_value)

        # Save the model class, but DO NOT create the model object yet. We allow the
        # user to pass in a class to simplify testing when bagle is not installed.
        if isinstance(model_info, str):
            try:
                from bagle import model
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "The bagle package is required to use the BagleWrapperModel. Please install it. "
                    "See https://bagle.readthedocs.io/en/latest/installation.html for instructions."
                ) from err
            self._model_class = getattr(model, model_info)
        else:
            self._model_class = model_info

        # Save the filter index mapping, using the default if none is provided.
        if filter_idx is None:
            self._filter_idx = self._default_filter_idx
        else:
            self._filter_idx = filter_idx

    @property
    def parameter_names(self):
        """The names of the parameters for this model."""
        return self._parameter_names

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
        # Extract the local parameters for this object from the full state object.
        local_params = self.get_local_params(state)

        # Create the bagle model object and set the parameters from the current state. We do this
        # here because the parameters saved in `state` will be different in each run.
        current_params = {param_name: local_params[param_name] for param_name in self.parameter_names}
        model_obj = self._model_class(**current_params)

        # If the model computes a t0 that is different from the one in the graph state, save the new one.
        if hasattr(model_obj, "t0") and "t0" in local_params:
            base_t0 = local_params["t0"]
            computed_t0 = model_obj.t0
            if np.abs(computed_t0 - base_t0) > 1e-8:
                node_name = str(self)
                state.set(node_name, "t0", computed_t0)

        # Use the newly created model object with the current parameters to compute the photometry.
        mags = model_obj.get_photometry(times, self._filter_idx[filter])
        bandflux = mag2flux(mags)
        return bandflux
