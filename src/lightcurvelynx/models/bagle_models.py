"""Wrappers for the models defined in bagle.

https://github.com/MovingUniverseLab/BAGLE_Microlensing
"""

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
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # Convenience mapping from filter name to index in the parameter list.
    _filter_idx = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    def __init__(self, model_info, parameter_dict, **kwargs):
        # We start by extracting the parameter information needed for a general physical model.
        ra = parameter_dict.get("raL", None)
        dec = parameter_dict.get("decL", None)
        t0 = parameter_dict.get("t0", None)
        super().__init__(ra=ra, dec=dec, t0=t0, **kwargs)

        # Add all of the parameters in the dictionary as settable parameters (if they are not
        # already set by the parent class) and save their names (in order) for later use.
        self._parameter_names = []
        for param_name, param_value in parameter_dict.items():
            self._parameter_names.append(param_name)
            if param_name not in self.setters:
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
