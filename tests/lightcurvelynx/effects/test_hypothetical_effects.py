"""This file includes tests for combinations of effects and settings that will not be used
in production, but help stress test the underlying machinery and make sure that the models
are working as expected in a variety of scenarios.
"""

import numpy as np
from lightcurvelynx.effects.basic_effects import ScaleFluxEffect
from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.state_expansion_node import StateExpansionNode
from lightcurvelynx.models.basic_models import ConstantSEDModel, StepModel


class _DuplicatingObservationEffect(EffectModel):
    """An effect that duplicates the observations of an object a specified number of
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
            The multiplicative factor by which to scale the flux. Not used for this effect.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        # No change to the flux density values.
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


def test_fake_lensing_effect():
    """Test that we can use a combination of the StateExpansionNode and the
    add_parameter_offset method to create a fake lensing effect.
    """
    basic_model = StepModel(
        brightness=GivenValueList([10.0, 20.0, 30.0]),
        ra=GivenValueList([0.0, 45.0, 90.0]),
        dec=GivenValueList([0.0, -10.0, 10.0]),
        t0=GivenValueList([100.0, 110.0, 120.0]),
        t1=GivenValueList([200.0, 210.0, 220.0]),
        node_label="basic_model",
    )

    # Create a lensing effect that will create duplicates of the observations
    # with different parameters. We use a dictionary of lists to specify the number
    # and values of the parameters for each duplicate.
    lensing_data = StateExpansionNode(
        param_names=["magnification", "time_delay"],
        param_values=GivenValueList(
            [
                {"magnification": [2.0, 1.0], "time_delay": [10.0, 20.0]},
                {"magnification": [0.5, 0.8, 0.1], "time_delay": [5.0, 8.0, 100.0]},
                {"magnification": [1.0], "time_delay": [20.0]},
            ],
        ),
    )
    basic_model.add_parameter_offset("t0", lensing_data.time_delay)
    basic_model.add_parameter_offset("t1", lensing_data.time_delay)
    basic_model.add_effect(ScaleFluxEffect(flux_scale=lensing_data.magnification))

    state = basic_model.sample_parameters(num_samples=3)
    assert state.num_samples == 6
    assert state["basic_model"]["ra"].tolist() == [0.0, 0.0, 45.0, 45.0, 45.0, 90.0]
    assert state["basic_model"]["dec"].tolist() == [0.0, 0.0, -10.0, -10.0, -10.0, 10.0]
    assert state["basic_model"]["brightness"].tolist() == [10.0, 10.0, 20.0, 20.0, 20.0, 30.0]
    assert state["basic_model"]["t0"].tolist() == [110.0, 120.0, 115.0, 118.0, 210.0, 140.0]
    assert state["basic_model"]["t1"].tolist() == [210.0, 220.0, 215.0, 218.0, 310.0, 240.0]

    times = np.array([200.0])
    wavelengths = np.array([5000.0])
    flux_density = basic_model.evaluate_sed(times, wavelengths, state)
    assert flux_density.shape == (6, 1, 1)
    assert flux_density[:, 0, 0].tolist() == [20.0, 10.0, 10.0, 16.0, 0.0, 30.0]
