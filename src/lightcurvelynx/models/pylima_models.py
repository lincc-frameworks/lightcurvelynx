"""Wrappers for the models defined in pyLIMA.

https://github.com/ebachelet/pyLIMA
"""

import warnings

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.astro_utils.mag_flux import Mag2FluxNode
from lightcurvelynx.consts import MJD_TO_JD_OFFSET
from lightcurvelynx.models.physical_model import BandfluxModel
from lightcurvelynx.utils.io_utils import SquashOutput


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
    blend_mags : dict, optional
        A mapping from filter names to blending magnitudes for the microlensed source.
    pylima_model_params : dict, optional
        A dictionary of additional pyLIMA parameters for the model, such as
        'u0', 'tE', 'rho', etc. If a parameter is already added to the model,
        its value will be updated instead.
    parallax_model : str, optional
        The pyLIMA parallax model type: 'None', 'Annual', 'Terrestrial', or 'Full'.
        The times for the parallax are automatically set during the evaluation.
    blend_flux_parameter : str, optional
        The pyLIMA blend flux parameter type. Currently only 'fblend' is supported.
        See also https://github.com/lincc-frameworks/lightcurvelynx/issues/691
    time_frame_offset : float, optional
        PyLIMA models use JD for time while users may specify any time system. This offset
        is added to the input times to convert them to JD. By default, this is set to
        MJD_TO_JD_OFFSET to convert from MJD to JD.
    observer_location : str, optional
        The location of the observer. Default is 'Earth'.
    **kwargs : dict, optional
        Any additional keyword arguments.

    Attributes
    ----------
    filters : list
        The list of filters supported by this model.
    parallax_model : str, optional
        The pyLIMA parallax model type: 'None', 'Annual', 'Terrestrial', or 'Full'.
    blend_flux_parameter : str, optional
        The pyLIMA blend flux parameter type. Currently only 'fblend' is supported.
        See also https://github.com/lincc-frameworks/lightcurvelynx/issues/691
    time_frame_offset: float, optional
        PyLIMA models use JD for time while users may specify any time system. This offset
        is added to the input times to convert them to JD.
    observer_location : str, optional
        The location of the observer.
    """

    def __init__(
        self,
        model_info,
        source_mags,
        *,
        blend_mags=None,
        pylima_params=None,
        parallax_model="None",
        blend_flux_parameter="fblend",
        time_frame_offset=MJD_TO_JD_OFFSET,
        observer_location="Earth",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time_frame_offset = time_frame_offset
        self.observer_location = observer_location

        # Save the pyLIMA parameters and add the corresponding model parameters.
        self.parallax_model = parallax_model
        self.blend_flux_parameter = blend_flux_parameter
        if blend_flux_parameter != "fblend":
            raise ValueError(
                f"Invalid blend_flux_parameter '{blend_flux_parameter}'. "
                f"Currently only 'fblend' is supported. See: "
                "https://github.com/lincc-frameworks/lightcurvelynx/issues/691"
            )

        # Add each source flux as a parameter by converting the input magnitudes.
        self.filters = list(source_mags.keys())
        for filter_name, mag in source_mags.items():
            self.add_parameter(
                f"fsource_{filter_name}",
                Mag2FluxNode(mag),
            )

        # Do the same for the (optional) blending magnitudes.
        if blend_mags is None:
            blend_mags = {}
        for filter_name in self.filters:
            param_name = f"{blend_flux_parameter}_{filter_name}"
            param_val = Mag2FluxNode(blend_mags[filter_name]) if filter_name in blend_mags else 0.0
            self.add_parameter(param_name, param_val)

        # Add any of the pyLIMA parameters from the pylima_params dictionary.
        if pylima_params is not None:
            for name, value in pylima_params.items():
                if name not in self.setters:
                    self.add_parameter(name, value)
                else:  # pragma: no cover
                    warnings.warn(f"Parameter {name} already exists in the model. Overriding its value.")
                    self.set_parameter(name, value)

        # Save the model class and create a model object. We allow the user
        # to pass in a class to simplify testing when PyLIMA is not installed.
        if isinstance(model_info, str):
            self._model_class = _load_pylima_model_class(model_info)
        else:
            self._model_class = model_info

        # Create a dummy pyLIMA model instance to get the parameter names and
        # check that they are all added. We use an arbitrary test time for parallax.
        test_time = 60676.0 + self.time_frame_offset
        with SquashOutput():
            event = self.make_pylima_event(
                ra=0.0,
                dec=0.0,
                filter="r",
                times=np.array([test_time]),
            )
            model = self._model_class(
                event,
                parallax=[self.parallax_model, test_time],
                blend_flux_parameter=self.blend_flux_parameter,
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

    def make_pylima_event(self, ra, dec, filter=None, times=None):
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
            A length T array of observer frame timestamps in JD for the telescope to attach.
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
                location=self.observer_location,
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
        # Check that the filter is supported.
        if filter not in self.filters:
            raise ValueError(f"Filter '{filter}' is not supported by this model (mags / blend).")

        params = self.get_local_params(state)

        try:
            from pyLIMA.simulations import simulator
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The pyLIMA package is required to use the PyLIMAWrapperModel. Please install it. "
                "See https://pylima.readthedocs.io/en/latest/ for instructions."
            ) from err

        # Compute the T0 in JD (which PyLIMA uses) from the MJD value in our parameters.
        t0_jd = params["t0"] + self.time_frame_offset
        times_jd = times + self.time_frame_offset

        # Squash pyLIMA's print output.
        with SquashOutput():
            # Create the PyLIMA event object with the attached telescope and a model instance.
            current_event = self.make_pylima_event(
                params["ra"],
                params["dec"],
                filter=filter,
                times=times_jd,
            )

            # Create the model instance.
            model = self._model_class(
                current_event,
                parallax=[self.parallax_model, t0_jd],
                blend_flux_parameter=self.blend_flux_parameter,
            )

            # Get the expected parameter mapping and create the ordered parameter list.
            expected_params_map = model.pyLIMA_standards_dictionnary
            ordered_values = [0.0] * len(expected_params_map)
            for name, index in expected_params_map.items():
                if name == "t0":
                    # We need to special case t0 because LightCurveLynx uses MJD and PyLIMA uses JD.
                    ordered_values[index] = t0_jd
                elif name in params:
                    ordered_values[index] = params[name]
            pyLIMA_params = model.compute_pyLIMA_parameters(ordered_values)

            # Simulate the lightcurve without noise.
            simulator.simulate_lightcurve(model, pyLIMA_params, add_noise=False)
            fluxes = current_event.telescopes[0].lightcurve["flux"]

        return fluxes
