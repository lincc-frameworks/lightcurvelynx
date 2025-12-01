"""Wrappers for the models defined in bagle.

https://github.com/MovingUniverseLab/BAGLE_Microlensing
"""

from citation_compass import CiteClass

from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.math_nodes.given_sampler import TableSampler
from lightcurvelynx.models.physical_model import BandfluxModel


def _load_bagle_model_class(model_name):
    """Load a bagle model class by name.

    Parameters
    ----------
    model_name : str
        The name of the bagle model class to load.

    Returns
    -------
    model_class : class
        The bagle model class.
    """
    try:
        from bagle import model
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "The bagle package is required to use the BagleWrapperModel. Please install it. "
            "See https://bagle.readthedocs.io/en/latest/installation.html for instructions."
        ) from err
    model_class = getattr(model, model_name)
    return model_class


class BagleWrapperModel(BandfluxModel, CiteClass):
    """A wrapper for single bagle models (one model type).

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    Additional parameterized values are used for specific bagle models.

    Note
    ----
    The t0 parameter saved in the results may be approximate depending on the bagle model
    used. Some models compute t0 from other parameters (e.g., time of closest approach).
    This updated t0 is not saved in the results.

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
        a default mapping for ugrizy filters to [0, 1, 2, 3, 4, 5] will be used.
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
            self._model_class = _load_bagle_model_class(model_info)
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

        # Use the newly created model object with the current parameters to compute the photometry.
        mags = model_obj.get_photometry(times, self._filter_idx[filter])
        bandflux = mag2flux(mags)
        return bandflux


class BagleMultiWrapperModel(BandfluxModel, CiteClass):
    """A wrapper for multiple bagle models (multiple model types).

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

    Atttributes
    -----------
    num_models : int
        The number of models being wrapped.
    parameter_dicts : dict
        A list of parameter dictionaries, one per model, each containing the parameter names
        and values for use in the corresponding model.

    Parameters
    ----------
    models : list of str or class
        The bagle model classes (or their names as strings) to use in the simulation.
    parameter_dicts : dict
        A list of parameter dictionaries, one per model, each containing.
    filter_idx : dict, optional
        A mapping from filter names to indices expected by the bagle model. If not provided,
        a default mapping for ugrizy filters to [0, 1, 2, 3, 4, 5] will be used.
    in_order : bool
        Return the given data in order of the rows (True). If False, performs
        random sampling with replacement. Default: False
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # Convenience mapping from filter name to index in the parameter list.
    _default_filter_idx = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    def __init__(self, models, parameter_dicts, filter_idx=None, in_order=False, **kwargs):
        self.parameter_dicts = parameter_dicts.copy()
        if len(models) != len(parameter_dicts):
            raise ValueError("The number of models must match the number of parameter dictionaries.")
        self.num_models = len(models)

        # We start by building a list of all the parameters across all the models.
        all_params = set()
        for param_dict in parameter_dicts:
            all_params.update(param_dict.keys())
        all_params = list(all_params)
        all_params.sort()

        # Next, we build a table of the model information (names and parameters). We also handle the
        # different t0 labels here, extracting the first one we find for use in the base physical model.
        sampler_data = {
            "_model_info": models,
            "_model_idx": list(range(len(models))),
            "_model_t0": [None] * len(models),
        }
        for param in all_params:
            sampler_data[param] = []
            for idx in range(self.num_models):
                value = parameter_dicts[idx].get(param, None)
                sampler_data[param].append(value)

                # Check for t0 or t0-like parameters to use for the base physical model.
                if param.startswith("t0") and sampler_data["_model_t0"][idx] is None:
                    sampler_data["_model_t0"][idx] = value
        self.sampler = TableSampler(sampler_data, in_order=in_order)

        # For the base physical model, we need to extract ra, dec, and t0. We check the parameter
        # dictionaries first, falling back to kwargs if needed.
        if "raL" in sampler_data:
            ra = self.sampler.raL
            if "ra" in kwargs:
                raise ValueError(
                    "The 'ra' parameter is specified in both parameter_dicts (as 'raL') "
                    " and kwargs (as 'ra'). Please only use the parameter_dicts."
                )
        elif "ra" in kwargs:
            ra = kwargs.pop("ra")
        else:
            ra = None

        if "decL" in sampler_data:
            dec = self.sampler.decL
            if "dec" in kwargs:
                raise ValueError(
                    "The 'dec' parameter is specified in both parameter_dicts (as 'decL') "
                    " and kwargs (as 'dec'). Please only use the parameter_dicts."
                )
        elif "dec" in kwargs:
            dec = kwargs.pop("dec")
        else:
            dec = None

        if "t0" in kwargs:
            raise ValueError(
                "The 't0' parameter must be specified in only the 'parameter_dict' "
                " for the BagleMultiWrapperModel."
            )

        super().__init__(ra=ra, dec=dec, t0=self.sampler._model_t0, **kwargs)

        # Add all of the parameters in the dictionary as settable parameters (if they are not
        # already set by the parent class).
        for param_name in sampler_data:
            if param_name in self.list_params():
                self.set_parameter(param_name, self.sampler.get_parameter_indicator(param_name))
            else:
                self.add_parameter(param_name, self.sampler.get_parameter_indicator(param_name))

        # Save the filter index mapping, using the default if none is provided.
        if filter_idx is None:
            self._filter_idx = self._default_filter_idx
        else:
            self._filter_idx = filter_idx

    def __len__(self):
        """The number of models wrapped by this object."""
        return self.num_models

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

        # Get the class of the model to use.
        model_info = local_params["_model_info"]
        if isinstance(model_info, str):
            model_class = _load_bagle_model_class(model_info)
        else:
            model_class = model_info

        # Get the model index and the list of keys.
        model_idx = local_params["_model_idx"]
        model_keys = self.parameter_dicts[model_idx].keys()

        # Create the bagle model object and set the parameters from the current state. We do this
        # here because the parameters saved in `state` will be different in each run.
        current_params = {}
        for param_name in model_keys:
            value = local_params.get(param_name, None)
            if value is not None:
                current_params[param_name] = value
        model_obj = model_class(**current_params)

        # Use the newly created model object with the current parameters to compute the photometry.
        mags = model_obj.get_photometry(times, self._filter_idx[filter])
        bandflux = mag2flux(mags)
        return bandflux
