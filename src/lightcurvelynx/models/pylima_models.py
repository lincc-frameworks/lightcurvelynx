"""Wrappers for the models defined in pyLIMA.

https://github.com/ebachelet/pyLIMA
"""

import warnings

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.mag_flux import Mag2FluxNode
from lightcurvelynx.consts import MJD_OFFSET
from lightcurvelynx.models.physical_model import BandfluxModel


def _load_pylima_model_class(model_name):
    """Load a pyLIMA model class by name.

    Parameters
    ----------
    model_name : str
        The name of the pyLIMA model type to load. E.g., 'PSPL'

    Returns
    -------
    model_class : class
        The pylima model class.
    """
    try:
        from pyLIMA import models
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "The pyLIMA package is required to use the PyLIMAWrapperModel. Please install it. "
            "See https://pylima.readthedocs.io/en/latest/ for instructions."
        ) from err

    # The package and the class uses different naming conventions with the
    # package having '_model' suffix and the class having 'model' suffix.
    model_package = getattr(models, f"{model_name}_model")
    model_class = getattr(model_package, f"{model_name}model")
    return model_class


class PyLIMAWrapperModel(BandfluxModel, CiteClass):
    """A wrapper for single pyLIMA models (one model type).

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    Additional parameterized values are used for specific PyLIMA models.

    Parameters
    ----------
    model_info : str or class
        The name of the pyLIMA model class to use in the simulation or the class itself.
    source_mags : dict
        A mapping from filter names to source magnitudes for the microlensed source.
    pylima_model_params : dict, optional
        A dictionary of additional pyLIMA parameters for the model, such as
        'u0', 'tE', 'rho', etc. If a parameter is already added to the model,
        its value will be updated instead.
    parallax_model : str, optional
        The pyLIMA parallax model type: 'None', 'Annual', 'Terrestrial', or 'Full'.
        The times for the parallax are automatically set during the evaluation.
    **kwargs : dict, optional
        Any additional keyword arguments.

    Attributes
    ----------
    filters : list
        The list of filters supported by this model.
    parallax_model : str, optional
        The pyLIMA parallax model type: 'None', 'Annual', 'Terrestrial', or 'Full'.
    blend_flux_parameter : str,
        The pyLIMA blend flux parameter type: 'fblend', 'gblend', 'ftotal', or
        or 'noblend'
    """

    def __init__(
        self,
        model_info,
        source_mags,
        *,
        pylima_params=None,
        parallax_model="None",
        blend_flux_parameter="noblend",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Save the pyLIMA parameters and add the corresponding model parameters.
        self.parallax_model = parallax_model
        self.blend_flux_parameter = blend_flux_parameter

        # Add each source flux as a parameter by converting the input magnitudes.
        self.filters = list(source_mags.keys())
        for filter_name, mag in source_mags.items():
            self.add_parameter(
                f"fsource_{filter_name}",
                Mag2FluxNode(mag),
            )

        # Add any of the pyLIMA parameters from the pylima_params dictionary.
        if pylima_params is not None:
            for name, value in pylima_params.items():
                if name not in self.setters:
                    self.add_parameter(name, value)
                else:
                    warnings.warn(f"Parameter {name} already exists in the model. Overriding its value.")
                    self.set_parameter(name, value)

        # Save the model class and create a model object. We allow the user
        # to pass in a class to simplify testing when PyLIMA is not installed.
        if isinstance(model_info, str):
            self._model_class = _load_pylima_model_class(model_info)
        else:
            self._model_class = model_info

        # Create a dummy pyLIMA model instance to get the parameter names and
        # check that they are all added.
        event = self.make_pylima_event(ra=0.0, dec=0.0, filter="r", times=np.array([60676.0]))
        model = self._model_class(
            event,
            parallax=[self.parallax_model, 60676.0 + MJD_OFFSET],
            blend_flux_parameter=self.blend_flux_parameter,
            double_source=self.double_source,
            orbital_motion=self.orbital_motion,
            origin=self.origin,
            fancy_parameters=self.fancy_parameters,
        )

        expected_params_map = model.pyLIMA_standards_dictionnary
        for name in expected_params_map:
            if name not in self.setters:
                if name in kwargs:
                    self.add_parameter(name, kwargs[name])
                else:
                    raise ValueError(
                        f"The pyLIMA model '{self._model_class.__name__}' uses parameter {name} "
                        f"but this was not provided as an argument or added as a model parameter."
                    )

    def make_pylima_emakevent(self, ra, dec, filter=None, times=None):
        """Create a pyLIMA event object and attach a telescope if filter and times are given.

        Parameters
        ----------
        ra : float
            The right ascension of the event in degrees.
        dec : float
            The declination of the event in degrees.
        filter : str, optional
            The name of the filter for the telescope to attach.
        times : numpy.ndarray, optional
            A length T array of observer frame timestamps in MJD for the telescope to attach.
        """
        try:
            from pyLIMA import event
            from pyLIMA.simulations import simulator
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The pyLIMA package is required to use the PyLIMAWrapperModel. Please install it. "
                "See https://pylima.readthedocs.io/en/latest/ for instructions."
            ) from err

        pylima_event = event.Event(ra=ra, dec=dec)
        pylima_event.name = "LightCurveLynx_Event"

        if filter is not None and times is not None:
            # Create a telescope object for the given filter.
            tel = simulator.simulate_a_telescope(
                name=filter,
                location="Earth",
                timestamps=times,
                astrometry=False,
            )
            pylima_event.telescopes.append(tel)

        return pylima_event

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
        params = self.get_local_params(state)

        try:
            from pyLIMA.simulations import simulator
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The pyLIMA package is required to use the PyLIMAWrapperModel. Please install it. "
                "See https://pylima.readthedocs.io/en/latest/ for instructions."
            ) from err

        # Create the PyLIMA event object with the attached telescope.
        current_event = self.make_pylima_event(
            params["ra"],
            params["dec"],
            filter=filter,
            times=times,
        )

        # Compute the T0 in JD (which PyLIMA uses) from the MJD value in our parameters.
        if self.parallax_model == "None":
            t0_jd = 0.0
        else:
            t0_jd = params["t0"] + MJD_OFFSET

        # Create the model instance.
        model = self._model_class(
            current_event,
            parallax=[self.parallax_model, t0_jd],
            blend_flux_parameter=self.blend_flux_parameter,
            double_source=self.double_source,
            orbital_motion=self.orbital_motion,
            origin=self.origin,
            fancy_parameters=self.fancy_parameters,
        )

        # Get the expected parameter mapping and create the ordered parameter list.
        expected_params_map = model.pyLIMA_standards_dictionnary
        ordered_values = [0.0] * len(expected_params_map)
        for name, index in expected_params_map.items():
            if name in params:
                ordered_values[index] = params[name]
        pyLIMA_params = model.compute_pyLIMA_parameters(ordered_values)

        # Simulate the lightcurve without noise.
        simulator.simulate_lightcurve(model, pyLIMA_params, add_noise=False)
        fluxes = current_event.telescopes[0].lightcurve
        return fluxes
