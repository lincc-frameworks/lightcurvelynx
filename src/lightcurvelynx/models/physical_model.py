"""The base classes for all models.

The code supports two types of models: 1) SEDModels define recipes for computing SEDs
at given times and wavelengths, accounting for redshift and other effects.
2) BandfluxModels only compute band fluxes for specific passbands instead of the SEDs. This is used for models
that are empirically fit from observed band fluxes.

We strongly recommend using the full SED models (SEDModels) whenever possible since they
more accurately simulate aspects such as the impact of redshift on rest frame effects.
"""

from abc import ABC
from os import urandom

import numpy as np

from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.astro_utils.redshift import RedshiftDistFunc, obs_to_rest_times_waves, rest_to_obs_flux
from lightcurvelynx.base_models import ParameterizedNode


class BasePhysicalModel(ParameterizedNode, ABC):
    """The abstract base class used to represent a physical model of a source of flux. This includes
    basic attributes, such as right ascension, declination, redshift, and distance.

    Physical models can have fixed attributes (where you need to create a new model or use
    a setter function to change them) and settable model parameters that can be passed functions
    or constants and are stored in the graph's (external) graph_state dictionary.

    Physical models also support adding and applying a variety of effects, such as redshift.

    Parameterized values include:
      * dec - The object's declination in degrees.
      * distance - The object's luminosity distance in pc.
      * ra - The object's right ascension in degrees.
      * redshift - The object's redshift.
      * t0 - The t0 of the zero phase (if applicable), date.

    Parameters
    ----------
    ra : float
        The object's right ascension (in degrees)
    dec : float
        The object's declination (in degrees)
    redshift : float
        The object's redshift.
    t0 : float
        The phase offset in MJD. For non-time-varying phenomena, this has no effect.
    distance : float
        The object's luminosity distance (in pc). If no value is provided and
        a cosmology parameter is given, the model will try to derive from
        the redshift and the cosmology.
    node_label : str, optional
        The label for the node in the model graph.
    seed : int, optional
        The seed for a random number generator.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        *,
        ra=None,
        dec=None,
        redshift=None,
        t0=None,
        distance=None,
        node_label=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(node_label=node_label, **kwargs)

        # Set the parameters for the model.
        self.add_parameter(
            "ra", ra, description="The object's right ascension (in degrees)", allow_gradient=False
        )
        self.add_parameter(
            "dec", dec, description="The object's declination (in degrees)", allow_gradient=False
        )
        self.add_parameter("redshift", redshift, description="The object's redshift.", allow_gradient=False)
        self.add_parameter("t0", t0, description="The phase offset in MJD.")

        # If the luminosity distance is provided, use that. Otherwise try the
        # redshift value using the cosmology (if given). Finally, default to None.
        if distance is not None:
            self.add_parameter(
                "distance",
                distance,
                description="The object's luminosity distance (in pc)",
                allow_gradient=False,
            )
        elif redshift is not None and kwargs.get("cosmology") is not None:
            self._redshift_func = RedshiftDistFunc(redshift=self.redshift, **kwargs)
            self.add_parameter(
                "distance",
                self._redshift_func,
                description="The object's luminosity distance (in pc)",
                allow_gradient=False,
            )
        else:
            self.add_parameter(
                "distance", None, description="The object's luminosity distance (in pc)", allow_gradient=False
            )

        # Get a default random number generator for this object, using the
        # given seed if one is provided.
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        self._rng = np.random.default_rng(seed=seed)

    def minwave(self, graph_state=None):
        """Get the minimum supported wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. If provided,
            the function will use the graph state to compute the minimum wavelength.

        Returns
        -------
        minwave : float or None
            The minimum wavelength of the model (in angstroms) or None
            if the model does not have a defined minimum wavelength.
        """
        return None

    def maxwave(self, graph_state=None):
        """Get the maximum supported wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. If provided,
            the function will use the graph state to compute the maximum wavelength.

        Returns
        -------
        maximum : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        return None

    def add_effect(self, effect):
        """Add an effect to the model. This effect will be applied to all
        fluxes densities simulated by the model.

        Any effect parameters that are not already in the model
        will be added to this node's parameters.

        Parameters
        ----------
        effect : EffectModel
            The effect to add.
        """
        raise NotImplementedError

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
        return np.full(len(times), True)

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
        # Check if we need to sample the graph.
        if state is None:
            state = self.sample_parameters(num_samples=1, rng_info=rng_info)

        if isinstance(passband_or_group, Passband):
            # If we are just given a passband, turn it into a passband group and save
            # the list of the filter name (repeated).
            passband_group = PassbandGroup([passband_or_group])
            if filters is None:
                filters = np.full(len(times), passband_or_group.filter_name)
        else:
            # This could be a PassbandGroup or (in limited cases) None.
            passband_group = passband_or_group

        if filters is None:
            raise ValueError("If passband_or_group is a PassbandGroup, filters must be provided.")
        filters = np.asarray(filters)
        if len(filters) != len(times):
            raise ValueError("Filters array must have the same length as times array.")

        # If we only have a single sample, we can return the band fluxes directly.
        if state.num_samples == 1:
            return self._evaluate_bandfluxes_single(passband_group, times, filters, state, rng_info=rng_info)

        # Fill in the band fluxes one at a time and return them all.
        bandfluxes = np.empty((state.num_samples, len(times)))
        for sample_num, current_state in enumerate(state):
            current_fluxes = self._evaluate_bandfluxes_single(
                passband_group,
                times,
                filters,
                current_state,
                rng_info=rng_info,
            )
            bandfluxes[sample_num, :] = current_fluxes[np.newaxis, :]
        return bandfluxes


class SEDModel(BasePhysicalModel):
    """A model of a source of flux that is defined at the SED level.

    Attributes
    ----------
    rest_frame_effects : list of EffectModel
        A list of effects to apply in the rest frame.
    obs_frame_effects : list of EffectModel
        A list of effects to apply in the observer frame.
    wave_extrapolation : WaveExtrapolationModel, optional
        The extrapolation model to use for wavelengths that fall outside
        the model's defined bounds.  If None then the model will use all zeros.
    apply_redshift : bool
        Whether to apply redshift to the model.

    Parameters
    ----------
    wave_extrapolation : WaveExtrapolationModel, optional
        The extrapolation model to use for wavelengths that fall outside
        the model's defined bounds.  If None then the model will use all zeros.
    """

    def __init__(
        self,
        wave_extrapolation=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Initialize the effect settings to their default values.
        self.apply_redshift = kwargs.get("redshift") is not None
        self.rest_frame_effects = []
        self.obs_frame_effects = []

        # Set the extrapolation for values outside the model's defined bounds.
        self.wave_extrapolation = wave_extrapolation

    def set_apply_redshift(self, apply_redshift):
        """Toggles the apply_redshift setting. If set to True, the model will
        apply redshift during the flux density computation including applying wavelength
        and time transformations.

        Parameters
        ----------
        apply_redshift : bool
            The new value for apply_redshift.
        """
        self.apply_redshift = apply_redshift

    def add_effect(self, effect, skip_params=False):
        """Add an effect to the model. This effect will be applied to all
        fluxes densities simulated by the model.

        Any effect parameters that are not already in the model
        will be added to this node's parameters.

        Parameters
        ----------
        effect : EffectModel
            The effect to add.
        skip_params : bool
            Skip adding the parameters to the model. This should only be done
            in very limited cases where the parameters are added via another mechanism.
            Most users should NOT change this setting.
            Default: False
        """
        # Add any effect parameters that are not already in the model.
        if not skip_params:
            for param_name, setter in effect.parameters.items():
                if param_name not in self.setters:
                    self.add_parameter(
                        param_name,
                        setter,
                        description=f"Added parameter by effect {effect}",
                        allow_gradient=False,
                    )

        # Add the effect to the appropriate list.
        if effect.rest_frame:
            self.rest_frame_effects.append(effect)
        else:
            self.obs_frame_effects.append(effect)

    def list_effects(self):
        """Return a list of all effects in the order in which they are applied."""
        return self.rest_frame_effects + self.obs_frame_effects

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free rest frame flux densities.
        The rest-frame flux is defined as F_nu = L_nu / 4*pi*D_L**2,
        where D_L is the luminosity distance.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of rest frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of rest frame SED values (in nJy).
        """
        raise NotImplementedError()

    def compute_sed_with_extrapolation(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object, extrapolating
        to wavelengths where the model is not defined.

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
        min_query_wave = np.min(wavelengths)
        min_valid_wave = self.minwave(graph_state=graph_state)
        if min_valid_wave is None:
            min_valid_wave = min_query_wave

        max_query_wave = np.max(wavelengths)
        max_valid_wave = self.maxwave(graph_state=graph_state)
        if max_valid_wave is None:
            max_valid_wave = max_query_wave

        # If no extrapolation is needed, just call compute SED.
        if min_query_wave >= min_valid_wave and max_query_wave <= max_valid_wave:
            return self.compute_sed(times, wavelengths, graph_state, **kwargs)

        # Truncate the wavelengths on which we evaluate the model.
        before_mask = wavelengths < min_valid_wave
        after_mask = wavelengths > max_valid_wave
        in_range = ~before_mask & ~after_mask

        # Pad the wavelengths with the min and max values and compute the flux at those points.
        query_waves = np.concatenate(([min_valid_wave], wavelengths[in_range], [max_valid_wave]))
        computed_flux = self.compute_sed(times, query_waves, graph_state)

        # Initially zero pad the full array until we fill in the values with extrapolation.
        # We drop the first and last flux values since they were added to get the flux at the
        # min and max wavelengths.
        flux_density = np.zeros((len(times), len(wavelengths)))
        flux_density[:, in_range] = computed_flux[:, 1:-1]

        # Do extrapolation for wavelengths that fell outside the model's bounds.
        if self.wave_extrapolation is not None:
            # Compute the flux values before the model's first valid wavelength.
            flux_density[:, before_mask] = self.wave_extrapolation(
                min_valid_wave,
                computed_flux[:, 0],
                wavelengths[before_mask],
            )

            # Compute the flux values after the model's last valid wavelength.
            flux_density[:, after_mask] = self.wave_extrapolation(
                max_valid_wave,
                computed_flux[:, -1],
                wavelengths[after_mask],
            )

        return flux_density

    def _evaluate_single(self, times, wavelengths, state, rng_info=None, **kwargs):
        """Evaluate the model and apply the effects for a single, given graph state.
        This function applies redshift, computes the flux density for the object,
        applies rest frames effects, performs the redshift correction (if needed),
        and applies the observer frame effects.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray
            A length N array of wavelengths (in angstroms).
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            All the other keyword arguments.
        """
        if state is None or state.num_samples != 1:
            raise ValueError("A GraphState with num_samples=1 required.")
        params = self.get_local_params(state)

        # Pre-effects are adjustments done to times and/or wavelengths, before flux density
        # computation. We skip if redshift is 0.0 since there is nothing to do.
        if self.apply_redshift and params["redshift"] != 0.0:
            if params.get("redshift", None) is None:
                raise ValueError("The 'redshift' parameter is required for redshifted models.")
            if params.get("t0", None) is None:
                raise ValueError("The 't0' parameter is required for redshifted models.")
            rest_times, rest_wavelengths = obs_to_rest_times_waves(
                times, wavelengths, params["redshift"], params["t0"]
            )
        else:
            rest_times = times
            rest_wavelengths = wavelengths

        # Compute the flux density for the object and apply any rest frame effects.
        flux_density = self.compute_sed_with_extrapolation(rest_times, rest_wavelengths, state, **kwargs)
        for effect in self.rest_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=rest_times,
                wavelengths=rest_wavelengths,
                rng_info=rng_info,
                **params,  # Provide all the node's parameters to the effect.
            )

        # Post-effects are adjustments done to the flux density after computation.
        if self.apply_redshift and params["redshift"] != 0.0:
            # We have alread checked that redshift is not None.
            flux_density = rest_to_obs_flux(flux_density, params["redshift"])

        # Apply observer frame effects.
        for effect in self.obs_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=times,
                wavelengths=wavelengths,
                rng_info=rng_info,
                **params,  # Provide all the node's parameters to the effect.
            )
        return flux_density

    def evaluate_sed(self, times, wavelengths, graph_state=None, given_args=None, rng_info=None, **kwargs):
        """Draw observations for this object and apply the noise.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState, optional
            An object mapping graph parameters to their values.
        given_args : dict, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length S x T x N matrix of SED values (in nJy), where S is the number of samples,
            T is the number of time steps, and N is the number of wavelengths.
            If S=1 then the function returns a T x N matrix.
        """
        # Make sure times and wavelengths are numpy arrays.
        times = np.asarray(times)
        wavelengths = np.asarray(wavelengths)

        # Check if we need to sample the graph.
        if graph_state is None:
            graph_state = self.sample_parameters(
                given_args=given_args, num_samples=1, rng_info=rng_info, **kwargs
            )

        # If we only have a single sample, do not bother to iterate through the states.
        if graph_state.num_samples == 1:
            return self._evaluate_single(
                times,
                wavelengths,
                graph_state,
                rng_info=rng_info,
                **kwargs,
            )

        # Iterate through each graph state computing the flux for each sample.
        results = np.empty((graph_state.num_samples, len(times), len(wavelengths)))
        for sample_num, state in enumerate(graph_state):
            # Compute the flux (handling redshift and applying all effects)
            # then save the result to the array of all results.
            results[sample_num, :, :] = self._evaluate_single(
                times,
                wavelengths,
                state,
                rng_info=rng_info,
                **kwargs,
            )
        return results

    def _evaluate_bandfluxes_single(self, passband_group, times, filters, state, rng_info=None) -> np.ndarray:
        """Get the band fluxes for a given PassbandGroup and a single, given graph state.

        Parameters
        ----------
        passband_group : PassbandGroup
            The passband group to use.
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : numpy.ndarray
            A length T array of filter names.
        state : GraphState
            An object mapping graph parameters to their values.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes for this sample.
        """
        bandfluxes = np.empty(len(times))
        for filter_name in np.unique(filters):
            # Compute the band fluxes for the times at which this filter is used.
            passband = passband_group[filter_name]
            filter_mask = filters == filter_name

            # Compute the spectral fluxes at the same wavelengths used to define the passband.
            # The evaluate function applies all effects (rest and observation frame) for the source
            # as well as handling all the redshift conversions.
            spectral_fluxes = self.evaluate_sed(times[filter_mask], passband.waves, state, rng_info=rng_info)
            bandfluxes[filter_mask] = passband.fluxes_to_bandflux(spectral_fluxes)
        return bandfluxes


class BandfluxModel(BasePhysicalModel, ABC):
    """A model of a source of flux that is only defined by band pass values
    in the observer frame (instead of a full SED).

    Instead of calling `compute_sed()` the model calls `compute_bandflux()` during
    its computation.

    Note
    ----
    We strongly recommend using the full SED models (SEDModel) whenever possible
    since they more accurately simulate aspects such as the impact of redshift on rest
    frame effects.

    Attributes
    ----------
    band_pass_effects : list of EffectModel
        A list of effects to apply in to the band pass fluxes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.band_pass_effects = []

    def set_apply_redshift(self, apply_redshift):
        """Toggles the apply_redshift setting.

        Parameters
        ----------
        apply_redshift : bool
            The new value for apply_redshift.
        """
        raise NotImplementedError("BandfluxModel does not support apply_redshift.")

    def add_effect(self, effect, skip_params=False):
        """Add an effect to the model.

        Parameters
        ----------
        effect : EffectModel
            The effect to add.
        skip_params : bool
            Skip adding the parameters to the model. This should only be done
            in very limited cases where the parameters are added via another mechanism.
            Most users should NOT change this setting.
            Default: False
        """
        # Add any effect parameters that are not already in the model.
        if not skip_params:
            for param_name, setter in effect.parameters.items():
                if param_name not in self.setters:
                    self.add_parameter(
                        param_name,
                        setter,
                        description=f"Added parameter by effect {effect}",
                        allow_gradient=False,
                    )

        # Add the effect to the band pass effects list.
        self.band_pass_effects.append(effect)

    def list_effects(self):
        """Return a list of all effects in the order in which they are applied."""
        return self.band_pass_effects

    def compute_bandflux(self, times, filters, state, rng_info=None):
        """Evaluate the model at the passband level for a single, given graph state.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : numpy.ndarray
            A length T array of filter names.
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        """
        raise NotImplementedError

    def _evaluate_bandfluxes_single(self, passband_group, times, filters, state, rng_info=None) -> np.ndarray:
        """Get the band fluxes for a given PassbandGroup and a single, given graph state.

        Note
        ----
        This function does not compute SEDs and integrate them through the passbands, but
        rather uses band fluxes directly.

        Parameters
        ----------
        passband_group : PassbandGroup
            The passband group to use.
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : numpy.ndarray
            A length T array of filter names.
        state : GraphState
            An object mapping graph parameters to their values.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.

        Returns
        -------
        bandflux : numpy.ndarray
            A length T array of band fluxes for this sample.
        """
        params = self.get_local_params(state)

        # Compute the flux (applying all effects) and save the result. Note that
        # BandfluxModel does not apply redshift, so all effects are applied in observer frame.
        bandfluxes = self.compute_bandflux(times, filters, state, rng_info=rng_info)
        for effect in self.band_pass_effects:
            bandfluxes = effect.apply_bandflux(
                bandfluxes,
                times=times,
                filters=filters,
                rng_info=rng_info,
                **params,  # Provide all the node's parameters to the effect.
            )
        return bandfluxes
