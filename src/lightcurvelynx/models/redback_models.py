"""Wrappers for the models defined in redback.

https://github.com/nikhil-sarin/redback
https://redback.readthedocs.io/en/latest/
"""

import astropy.units as uu
import numpy as np
from citation_compass import CiteClass, cite_inline

from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.math_nodes.bilby_priors import BilbyPriorNode
from lightcurvelynx.models.physical_model import SEDModel


class RedbackWrapperModel(SEDModel, CiteClass):
    """A wrapper for redback models.

    Parameterized values include:

    * dec - The object's declination in degrees. [from BasePhysicalModel]
    * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
    * ra - The object's right ascension in degrees. [from BasePhysicalModel]
    * redshift - The object's redshift. [from BasePhysicalModel]
    * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]

    Additional parameterized values are used for specific redback models.

    References
    ----------
    * redback - https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.1203S/abstract
    * Individual models might require citation. See references in the redback documentation.

    Attributes
    ----------
    source : function
        The underlying source function that maps time + wavelength to flux.
    source_name : str
        The name used to set the source.
    source_param_names : list
        A list of the source model's parameters that we need to set.

    Parameters
    ----------
    source : str or function
        The name of the redback model function used to generate the SEDs or
        the actual function itself.
    priors : dict, bilby.prior.PriorDict, or BilbyPriorNode, optional
        The redback model's Bilby priors.
    parameters : dict, optional
        A dictionary of parameter setters to pass to the source function.
    wave_bounds : tuple of (float, float), optional
        A tuple of (min_wave, max_wave) in angstroms to set the wavelength bounds for the
        model. If not provided, the code will try to infer the bounds from the model result.
        However this is not always possible and may lead to evaluating the model at invalid points.
        Default: (None, None) which corresponds to (-inf, inf).
    phase_bounds : tuple of (float, float), optional
        A tuple of (min_phase, max_phase) in days to set the phase bounds for the model.
        If not provided, the code will try to infer the bounds from the model result. However this
        is not always possible and may lead to evaluating the model at invalid points.
        Default: (None, None) which corresponds to (-inf, inf).
    **kwargs : dict, optional
        Any additional keyword arguments.

    Note
    ----
    You can automatically extract the priors for a model (in the correct format)
    using redback's `get_priors()` function and passing the name of the model
    as the `model` argument: `priors = get_priors(model="one_component_kilonova_model")`
    """

    # A class variable for the units so we are not computing them each time.
    _FLAM_UNIT = uu.erg / uu.second / uu.cm**2 / uu.AA

    def __init__(
        self,
        source,
        *,
        priors=None,
        parameters=None,
        wave_bounds=(None, None),
        phase_bounds=(None, None),
        **kwargs,
    ):
        # Check that the parameters passed in the dictionary and keyword arguments
        # do not overlap, so we only have one source of truth. This is needed for
        # parameters like `redshift` that overlap core parameters.
        if parameters is None:
            parameters = {}
        for key in parameters:
            if key in kwargs:
                raise ValueError(
                    f"Parameter '{key}' specified in both the parameters dictionary "
                    "and as a parameter itself. Please include it only in the dictionary."
                )

        super().__init__(**kwargs)

        # Add all of the items from the bilby prior node as settable parameters.
        parameters = parameters.copy()
        if priors is not None:
            if not isinstance(priors, BilbyPriorNode):
                priors = BilbyPriorNode(prior=priors)
            for param_name in priors.outputs:
                if param_name not in parameters:
                    parameters[param_name] = getattr(priors, param_name)

        # Use the parameter dictionary to create settable parameters for the model.
        # Some of these might have already been added by the superclass's constructor,
        # so we just change it.
        self.source_param_names = []
        for key, value in parameters.items():
            if key in self.setters:
                self.set_parameter(key, value)
            else:
                self.add_parameter(key, value, description="Parameter for redback model.")
            self.source_param_names.append(key)

        # Create the source itself.
        if isinstance(source, str):
            try:
                import redback
            except ImportError as err:  # pragma: no cover
                raise ImportError(
                    "redback package is not installed by default. To use the RedbackWrapperModel, "
                    "please install redback. For example, you can install it with "
                    "`pip install redback`."
                ) from err

            self.source_name = source
            if source not in redback.model_library.all_models_dict:  # pragma: no cover
                raise ValueError(f"Redback model '{source}' not found in redback model library.")
            self.source = redback.model_library.all_models_dict[source]
        else:
            self.source_name = source.__name__
            self.source = source

        # Check if the model has a citation parameter we should include.
        if hasattr(self.source, "citation"):
            cite_inline("redback model", self.source.citation)

        # Redback models already handle redshift, so we do not want to double apply it.
        self.apply_redshift = False

        # We save a cached version of the last computed SED. This starts as None
        # since we have not computed any SEDs yet.
        self._cached_data = {}

        # Save the bounds in wavelength and time.
        if len(wave_bounds) != 2:
            raise ValueError("wave_bounds should be a tuple of (min_wave, max_wave).")
        self._min_wave = wave_bounds[0]
        self._max_wave = wave_bounds[1]

        if len(phase_bounds) != 2:
            raise ValueError("phase_bounds should be a tuple of (min_phase, max_phase).")
        self._min_phase = phase_bounds[0]
        self._max_phase = phase_bounds[1]

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.source_param_names

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
        if self._min_wave is not None:
            return self._min_wave
        return self._cached_data.get("minwave", None)

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
        if self._max_wave is not None:
            return self._max_wave
        return self._cached_data.get("maxwave", None)

    def minphase(self, **kwargs):
        """Get the minimum supported phase of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        minphase : float or None
            The minimum phase of the model (in days) or None
            if the model does not have a defined minimum phase.
        """
        if self._min_phase is not None:
            return self._min_phase
        return self._cached_data.get("minphase", None)

    def maxphase(self, **kwargs):
        """Get the maximum supported phase of the model in days.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        maximum : float or None
            The maximum phase of the model (in days) or None
            if the model does not have a defined maximum phase.
        """
        if self._max_phase is not None:
            return self._max_phase
        return self._cached_data.get("maxphase", None)

    def _do_bounds_precomputation(self, times, wavelengths, graph_state=None, **kwargs):
        """Precompute the SED model for the given times and wavelengths.

        This is needed for redback models because bounds [minwave, maxwave] and
        [minphase, maxphase] depend on the last computed SED.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps (MJD).
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
           Any additional keyword arguments.
        """
        # Check that the cached data is empty.
        if len(self._cached_data) != 0:  # pragma: no cover
            raise RuntimeError(
                "Cached data should be empty before precomputation. This indicates a bug "
                " in the code where the cached data is not being cleared after use."
            )

        # Build the function arguments from the parameter values.
        params = self.get_local_params(graph_state)
        fn_args = {}
        for name in self.source_param_names:
            fn_args[name] = params[name]

        # Compute the shifted times.
        t0 = params.get("t0", 0.0)
        if t0 is None:
            t0 = 0.0
        shifted_times = times - t0

        # Only evaluate the model within its phase bounds. Explosion-based redback
        # models are only defined for time >= 0 (time since explosion), so we always
        # clip to the effective minimum phase (max of 0 and any user-supplied lower
        # bound). If we end up with no post-explosion times, use phase=0.1 so we can
        # still infer the wavelength bounds from a valid model evaluation.
        min_phase = max(0.0, self._min_phase) if self._min_phase is not None else 0.0
        shifted_times = shifted_times[shifted_times >= min_phase]
        if self._max_phase is not None:
            shifted_times = shifted_times[shifted_times <= self._max_phase]
        if len(shifted_times) == 0:
            shifted_times = np.array([0.1])

        # Call the source function to get the RedbackTimeSeriesSource object.
        # We create this object with each call, because it depends on the parameters (fn_args).
        try:
            rb_result = self.source(
                shifted_times,
                output_format="sncosmo_source",
                **fn_args,
            )
        except Exception as err:  # pragma: no cover
            raise RuntimeError(
                "Error calling the redback model function. This is often due to invalid parameter values "
                "or time/wavelength values outside the model's bounds. If needed, you can set the bounds "
                "manually using the `wave_bounds` and `phase_bounds` parameters of the RedbackWrapperModel."
                "Original error message: " + str(err)
            ) from err

        # Save the computed RedbackTimeSeriesSource and the bounds.
        self._cached_data["minwave"] = rb_result.minwave()
        self._cached_data["maxwave"] = rb_result.maxwave()
        self._cached_data["minphase"] = rb_result.minphase()
        self._cached_data["maxphase"] = rb_result.maxphase()
        self._cached_data["last_sed"] = rb_result

    def compute_sed_with_extrapolation(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object, extrapolating
        to times and wavelengths where the model is not defined.

        We override this method because the extrapolation bounds will depend on the materialized
        RedbackTimeSeriesSource object, so we need to do the precomputation to get that object.
        We will cache the object so we don't perform the computation twice.

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
        # Do any precomputation that is needed to set the bounds. This generates
        # cached data that will be used in the compute_sed method to avoid redundant
        # computation.
        self._do_bounds_precomputation(times, wavelengths, graph_state, **kwargs)

        # Call the superclass method to do the extrapolation. This will call compute_sed,
        # which will use the cached data to avoid redundant computation.
        sed = super().compute_sed_with_extrapolation(times, wavelengths, graph_state, **kwargs)

        # Clear the cached data values.
        self._cached_data.clear()

        return sed

    def compute_sed(self, times, wavelengths, graph_state=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps (MJD).
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
        if self._cached_data.get("last_sed") is None:
            # If we have not computed the result object for the current parameters, do so now.
            self._do_bounds_precomputation(times, wavelengths, graph_state, **kwargs)
            created_cache = True
        else:
            created_cache = False

        # Load the RedbackTimeSeriesSource we need for this computation.
        rb_result = self._cached_data["last_sed"]

        # Compute the shifted times. Note these might be different from the times used in
        # precomputation because of the applied phase bounds.
        # times used for precomputation.
        params = self.get_local_params(graph_state)
        t0 = params.get("t0", 0.0)
        if t0 is None:
            t0 = 0.0
        shifted_times = times - t0

        # Split times into post-explosion (>= 0) and pre-explosion (< 0).
        # Explosion-based redback models (kilonovae, supernovae, etc.) are only
        # defined for time >= 0 (time since explosion). Pre-explosion observations
        # are assigned zero flux so that they appear as non-detections in the
        # simulated survey, which is the physically correct behaviour.
        #
        # get_flux_density returns shape (n_times, n_waves). We evaluate only the
        # post-explosion times, then reconstruct the full (n_times, n_waves) array
        # with zeros for pre-explosion rows.
        post_explosion = shifted_times >= 0.0
        post_times = shifted_times[post_explosion]
        n_time = len(shifted_times)

        if post_times.size > 0:
            model_flam_post = rb_result.get_flux_density(post_times, wavelengths)
            model_fnu_post = flam_to_fnu(
                model_flam_post,
                wavelengths,
                wave_unit=uu.AA,
                flam_unit=self._FLAM_UNIT,
                fnu_unit=uu.nJy,
            )
            # Reconstruct full (n_time, n_waves) array with zeros for pre-explosion rows.
            if model_fnu_post.ndim == 2:
                model_fnu = np.zeros((n_time, model_fnu_post.shape[1]))
                model_fnu[post_explosion] = model_fnu_post
            else:
                model_fnu = np.zeros(n_time)
                model_fnu[post_explosion] = model_fnu_post
        else:
            # All times are pre-explosion — probe shape with a dummy call at t=0.1.
            dummy_flam = rb_result.get_flux_density(np.array([0.1]), wavelengths)
            if dummy_flam.ndim == 2:
                model_fnu = np.zeros((n_time, dummy_flam.shape[1]))
            else:
                model_fnu = np.zeros(n_time)

        # Clear the cached data values if we created them locally.
        if created_cache:
            self._cached_data.clear()

        return model_fnu
