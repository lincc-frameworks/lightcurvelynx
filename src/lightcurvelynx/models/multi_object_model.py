"""Multiple object models wrap multiple BasePhysicalModels, allowing the
user to define such operations as additive models, where each object
contributes to the total flux density, or random object models, where
only one object is selected at random for each flux calculation.
"""

import numpy as np

from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.math_nodes.given_sampler import GivenValueSampler
from lightcurvelynx.models.physical_model import BandfluxModel, BasePhysicalModel, SEDModel


class MultiObjectModel(SEDModel):
    """A MultiObjectModel wraps multiple BasePhysicalModels (including BandfluxModels).

    All rest frame effects are applied to each object, allowing different redshifts
    for each object (for unresolved objects).  The observer frame effects are applied
    to the weighted sum of the objects.

    While this model supports both BandfluxModels and SED, it inherits from SEDModel
    to pick up some of the helper functions.

    Note: Each object may have its own sampled (RA, dec) position, which are not
    required to align.

    Attributes
    ----------
    objects : list
        A list of BasePhysicalModel objects to use in the flux calculation.
    num_objects : int
        The number of objects in the model.
    _is_bandflux : list
        A list of Booleans indicating whether each model is a BandfluxModel.
    _any_bandflux : bool
        True if any of the models are bandflux models.

    Parameters
    ----------
    objects : list
        A list of BasePhysicalModel objects to use in the flux calculation.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        objects,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Check that all objects are BasePhysicalModel objects and mark whether they are BandfluxModels.
        self._is_bandflux = [False] * len(objects)
        self._any_bandflux = False
        for idx, object in enumerate(objects):
            if isinstance(object, BandfluxModel):
                self._is_bandflux[idx] = True
                self._any_bandflux = True
            elif not isinstance(object, BasePhysicalModel):
                raise ValueError("All objects must be BasePhysicalModel objects.")

        self.objects = objects
        self.num_objects = len(objects)

    def set_graph_positions(self, seen_nodes=None):
        """Force an update of the graph structure (numbering of each node).

        Parameters
        ----------
        seen_nodes : set, optional
            A set of nodes that have already been processed to prevent infinite loops.
            Caller should not set.
        """
        if seen_nodes is None:
            seen_nodes = set()

        # Set the graph positions for this node and each object.
        super().set_graph_positions(seen_nodes=seen_nodes)
        for object in self.objects:
            object.set_graph_positions(seen_nodes=seen_nodes)

    def set_apply_redshift(self, apply_redshift):
        """Toggles the apply_redshift setting.

        Parameters
        ----------
        apply_redshift : bool
            The new value for apply_redshift.
        """
        for idx, object in enumerate(self.objects):
            if self._is_bandflux[idx]:
                object.set_apply_redshift(apply_redshift)

    def add_effect(self, effect):
        """Add an effect to each of the submodels.

        Parameters
        ----------
        effect : Effect
            The effect to add to the model.
        """
        if effect.rest_frame:
            # The rest frame effects are applied to each object.
            for object in self.objects:
                object.add_effect(effect)
        else:
            # Add observer frame effects to this model.
            # Add any effect parameters that are not already in the model.
            for param_name, setter in effect.parameters.items():
                if param_name not in self.setters:
                    self.add_parameter(
                        param_name,
                        setter,
                        description=f"A parameter added by Effect {effect}",
                        allow_gradient=False,
                    )
            self.obs_frame_effects.append(effect)

    def sample_parameters(self, given_args=None, num_samples=1, rng_info=None):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedModel.

        Parameters
        ----------
        given_args : dict, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        num_samples : int
            A count of the number of samples to compute.
            Default: 1
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Returns
        -------
        graph_state : GraphState
            An object mapping graph parameters to their values.
        """
        # If the graph has not been sampled ever, update the node positions for every node.
        if self.node_pos is None:
            self.set_graph_positions()

        # We use the same seen_nodes for all sampling calls so each node
        # is sampled at most one time regardless of link structure.
        graph_state = GraphState(num_samples)
        if given_args is not None:
            graph_state.update(given_args, all_fixed=True)

        seen_nodes = {}
        for object in self.objects:
            object._sample_helper(graph_state, seen_nodes, rng_info=rng_info)
        self._sample_helper(graph_state, seen_nodes, rng_info=rng_info)

        return graph_state

    def _evaluate_single(self, times, wavelengths, state, **kwargs):
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
        **kwargs : dict, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        raise NotImplementedError()  # pragma: no cover

    def _evaluate_bandfluxes_single(self, passband_group, times, filters, state) -> np.ndarray:
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

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes for this sample.
        """
        raise NotImplementedError()  # pragma: no cover


class AdditiveMultiObjectModel(MultiObjectModel):
    """An AdditiveMultiObjectModel computes the flux from multiple overlapping objects,
    including (host galaxy and source pairs) or unresolved sources.

    All rest frame effects are applied to each model, allowing different redshifts
    for each model (for unresolved objects).  The observer frame effects are applied
    to the weighted sum of the models.

    Note: Each model may have its own sampled (RA, dec) position, which are not
    required to align.

    Attributes
    ----------
    objects : list
        A list of BasePhysicalModel objects to use in the flux calculation.
    weights : numpy.ndarray, optional
        A length N array of weights to apply to each object. If None, all objects
        will be weighted equally.
    num_objects : int
        The number of objects in the model.

    Parameters
    ----------
    objects : list
        A list of BasePhysicalModel objects to use in the flux calculation.
    weights : numpy.ndarray, optional
        A length N array of weights to apply to each object. If None, all objects
        will be weighted equally.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        objects,
        weights=None,
        **kwargs,
    ):
        super().__init__(objects, **kwargs)

        if weights is None:
            self.weights = np.full(len(objects), 1.0)
        elif len(weights) != len(objects):
            raise ValueError("Length of weights must match length of objects.")
        else:
            self.weights = weights

    def minwave(self, graph_state=None):
        """Get the minimum wavelength of the model. For additive models, this is
        a list of minimums for each object.

        Note
        ----
        Wavelength extrapolation is handled by each object. So the actual wavelength's
        can be evaluated outside the range of each object.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. If provided,
            the function will use the graph state to compute the minimum wavelength.

        Returns
        -------
        minwave : list of float or None
            The minimum wavelength of the each object (in angstroms) or None
        """
        return [object.minwave(graph_state=graph_state) for object in self.objects]

    def maxwave(self, graph_state=None):
        """Get the maximum wavelength of the model. For additive models, this is
        a list of maximums for each object.

        Note
        ----
        Wavelength extrapolation is handled by each object. So the actual wavelength's
        can be evaluated outside the range of each object.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. If provided,
            the function will use the graph state to compute the maximum wavelength.

        Returns
        -------
        maxwave : list of float or None
            The maximum wavelength of the each object (in angstroms) or None
        """
        return [object.maxwave(graph_state=graph_state) for object in self.objects]

    def _evaluate_single(self, times, wavelengths, state, **kwargs):
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
        **kwargs : dict, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        # Check that all models can compute SEDs.
        if self._any_bandflux:
            raise TypeError(
                "AdditiveMultiObjectModel contains at least one BandfluxModel, "
                "which does not support the evaluation of SEDs."
            )

        # Compute the weighted sum of contributions from each object. Since we use each
        # object's _evaluate_single function, the rest frame effects are applied
        # correctly for each object and wavelength extrapolation is handled by each object
        # (allowing them to have different wavelength ranges).
        flux_density = np.zeros((len(times), len(wavelengths)))
        for object, weight in zip(self.objects, self.weights, strict=False):
            flux_density += weight * object._evaluate_single(
                times,
                wavelengths,
                state,
                **kwargs,
            )

        # Apply the observer frame effects on the weighted sum of objects.
        params = self.get_local_params(state)
        for effect in self.obs_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=times,
                wavelengths=wavelengths,
                **params,
            )

        return flux_density

    def _evaluate_bandfluxes_single(self, passband_group, times, filters, state) -> np.ndarray:
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

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes for this sample.
        """
        # Compute the band fluxes as the weighted sum of all bandfluxes. All models will have rest frame
        # effects applied prior to summing. In the case of full SED models, the effects will be applied
        # to the entire SED before integrating with the filters to compute the band fluxes.
        bandfluxes = np.zeros(len(times))
        for idx, object in enumerate(self.objects):
            object_fluxes = object._evaluate_bandfluxes_single(
                passband_group,
                times,
                filters,
                state,
            )
            bandfluxes += self.weights[idx] * object_fluxes

        # Apply any common rest frame effects to the total bandflux.  We need to use the effects'
        # apply_bandflux() function since we no longer have SEDs.
        params = self.get_local_params(state)
        for effect in self.obs_frame_effects:
            bandfluxes = effect.apply_bandflux(
                bandfluxes,
                times=times,
                filters=filters,
                **params,
            )

        return bandfluxes


class RandomMultiObjectModel(MultiObjectModel):
    """A RandomMultiObjectModel selects one of its objects at random and
    computes the flux from that object.

    Attributes
    ----------
    object_map : dict
        A dictionary mapping each object name (or index) to a BasePhysicalModel object.
    num_objects : int
        The number of objects in the model.

    Parameters
    ----------
    objects : list
        A list of BasePhysicalModel objects to use in the flux calculation.
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        a object at random. If None, all objects will be weighted equally.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        objects,
        weights=None,
        object_names=None,
        **kwargs,
    ):
        super().__init__(objects, **kwargs)

        # Create a parameter to indicate which object was selected.
        object_names = object_names or [src.node_string for src in objects]
        self.object_map = {name: src for name, src in zip(object_names, objects, strict=False)}
        self._sampler_node = GivenValueSampler(object_names, weights=weights)
        self.add_parameter(
            "selected_object",
            value=self._sampler_node,
            allow_gradient=False,
            description="Index of the object selected for sampling.",
        )

    def minwave(self, graph_state=None):
        """Get the minimum wavelength of the model.

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
        idx = self.get_param(graph_state, "selected_object")
        return self.objects[idx].minwave(graph_state=graph_state)

    def maxwave(self, graph_state=None):
        """Get the maximum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. If provided,
            the function will use the graph state to compute the maximum wavelength.

        Returns
        -------
        maxwave : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        idx = self.get_param(graph_state, "selected_object")
        return self.objects[idx].maxwave(graph_state=graph_state)

    def _evaluate_single(self, times, wavelengths, state, **kwargs):
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
        **kwargs : dict, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        # Check that all models can compute SEDs.
        if self._any_bandflux:
            raise TypeError(
                "RandomMultiObjectModel contains at least one BandfluxModel, "
                "which does not support the evaluation of SEDs."
            )

        # Use the model selected by the sampler node to compute the flux density.
        model_name = self.get_param(state, "selected_object")
        flux_density = self.object_map[model_name]._evaluate_single(
            times,
            wavelengths,
            state,
            **kwargs,
        )

        # Apply the observer frame effects on the selected object.
        params = self.get_local_params(state)
        for effect in self.obs_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=times,
                wavelengths=wavelengths,
                **params,
            )

        return flux_density

    def _evaluate_bandfluxes_single(self, passband_group, times, filters, state) -> np.ndarray:
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

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes for this sample.
        """
        # Use the model selected by the sampler node to compute the flux density.
        model_name = self.get_param(state, "selected_object")
        bandfluxes = self.object_map[model_name]._evaluate_bandfluxes_single(
            passband_group,
            times,
            filters,
            state,
        )

        # Apply the observer frame effects on the selected object.
        params = self.get_local_params(state)
        for effect in self.obs_frame_effects:
            bandfluxes = effect.apply_bandflux(
                bandfluxes,
                times=times,
                filters=filters,
                **params,
            )
        return bandfluxes
