"""This file includes tests for combinations of effects and settings that will not be used
in production, but help stress test the underlying machinery and make sure that the models
are working as expected in a variety of scenarios.
"""

from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.state_expansion_node import StateExpansionNode
from lightcurvelynx.models.basic_models import ConstantSEDModel


class _DuplicatingObservationEffect(EffectModel):
    """An effect that duplcates the observations of an object a specified number of
    times with the same parameters.

    Attributes
    ----------
    repeats : parameter
        The number of times to duplicate the observations.
    """

    def __init__(self, repeats, **kwargs):
        super().__init__(**kwargs)
        self.add_effect_parameter(
            "number_of_duplicates",
            StateExpansionNode(repeats=repeats).repeats,
        )

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        flux_scale=None,
        **kwargs,
    ):
        """Apply the effect to observations (flux_density values).

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD). Not used for this effect.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms). Not used for this effect.
        flux_scale : float, optional
            The multiplicative factor by which to scale the flux. Raises an error if None is provided.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        # No change tothe flux density values.
        return flux_density

    def apply_bandflux(
        self,
        bandfluxes,
        *,
        times=None,
        filters=None,
        flux_scale=None,
        **kwargs,
    ):
        """Apply the effect to band fluxes.

        Parameters
        ----------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD).
        filters : numpy.ndarray, optional
            A length N array of filters. If not provided, the effect is applied to all
            band fluxes.
        flux_scale : float, optional
            The multiplicative factor by which to scale the flux. Raises an error if None is provided.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes after the effect is applied (in nJy).
        """
        # No change to the band flux values.
        return bandfluxes


def test_duplicate_object_effects():
    """Test that we can use the StateExpansionNode in an effect to duplicate the observations."""
    basic_model = ConstantSEDModel(
        brightness=GivenValueList([10.0, 20.0, 30.0]),
        ra=GivenValueList([0.0, 45.0, 90.0]),
        dec=GivenValueList([0.0, -10.0, 10.0]),
        node_label="basic_model",
    )
    state = basic_model.sample_parameters(num_samples=3)
    assert state.num_samples == 3
    assert state["basic_model"]["ra"].tolist() == [0.0, 45.0, 90.0]
    assert state["basic_model"]["dec"].tolist() == [0.0, -10.0, 10.0]
    assert state["basic_model"]["brightness"].tolist() == [10.0, 20.0, 30.0]

    # We can add a duplicating effect to the model that will create
    # duplicates of the observations.
    basic_model2 = ConstantSEDModel(
        brightness=GivenValueList([10.0, 20.0, 30.0]),
        ra=GivenValueList([0.0, 45.0, 90.0]),
        dec=GivenValueList([0.0, -10.0, 10.0]),
        node_label="basic_model",
    )
    effect = _DuplicatingObservationEffect(repeats=GivenValueList([2, 0, 3]))
    basic_model2.add_effect(effect)

    state2 = basic_model2.sample_parameters(num_samples=3)
    assert state2.num_samples == 5
    assert state2["basic_model"]["ra"].tolist() == [0.0, 0.0, 90.0, 90.0, 90.0]
    assert state2["basic_model"]["dec"].tolist() == [0.0, 0.0, 10.0, 10.0, 10.0]
    assert state2["basic_model"]["brightness"].tolist() == [10.0, 10.0, 30.0, 30.0, 30.0]
