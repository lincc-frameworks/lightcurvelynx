import numpy as np
import pytest
from lightcurvelynx.effects.basic_effects import ScaleFluxEffect
from lightcurvelynx.models.basic_models import ConstantSEDModel


def test_scale_flux_effect() -> None:
    """Test that we can create and sample a ScaleFluxEffect object."""
    effect = ScaleFluxEffect(flux_scale=0.1)
    assert str(effect) == "ScaleFluxEffect"
    assert repr(effect) == "ScaleFluxEffect(flux_scale)"

    # We can apply the noise.
    values = np.full((5, 3), 100.0)
    values = effect.apply(values, flux_scale=0.1)
    assert np.all(values == 10.0)

    # We fail if we do not pass in the expected parameters.
    with pytest.raises(ValueError):
        _ = effect.apply(values)


def test_scale_flux_effect_bandflux() -> None:
    """Test that we can sample a ScaleFluxEffect object at the bandflux level."""
    values = np.full(10, 100.0)

    # We can apply the noise.
    effect = ScaleFluxEffect(flux_scale=0.1)
    values = effect.apply_bandflux(values, flux_scale=0.1)
    assert np.all(values == 10.0)

    # We fail if we do not pass in the expected parameters.
    with pytest.raises(ValueError):
        _ = effect.apply_bandflux(values)

    # We fail if we use a negative flux scale.
    with pytest.raises(ValueError):
        _ = effect.apply_bandflux(values, flux_scale=-1.0)


def test_constant_sed_model_scale_flux_effect() -> None:
    """Test that we can sample and create a ConstantSEDModel object with constant dimming."""
    model = ConstantSEDModel(brightness=10.0, node_label="my_constant_sed_model")
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 0

    # We can add the white noise effect.  By default it is a rest frame effect.
    effect = ScaleFluxEffect(flux_scale=0.1)
    model.add_effect(effect)
    assert len(model.rest_frame_effects) == 1
    assert len(model.obs_frame_effects) == 0

    # Check that the flux_scale parameter is stored in the model node.
    state = model.sample_parameters()
    assert state["my_constant_sed_model"]["flux_scale"] == 0.1

    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (6, 3)
    assert np.all(values == 1.0)

    # We can add the white noise effect as a observer frame effect instead.
    model2 = ConstantSEDModel(brightness=10.0, node_label="my_constant_sed_model")
    effect2 = ScaleFluxEffect(flux_scale=0.5, rest_frame=False)
    model2.add_effect(effect2)
    assert len(model2.rest_frame_effects) == 0
    assert len(model2.obs_frame_effects) == 1

    state2 = model2.sample_parameters()
    values2 = model2.evaluate_sed(times, wavelengths, state2)
    assert values2.shape == (6, 3)
    assert np.all(values2 == 5.0)


def test_constant_sed_scale_flux_effect_alt_params() -> None:
    """Test that we can turn off adding parameters, but this will fail."""
    model = ConstantSEDModel(brightness=10.0, node_label="my_constant_sed_model")
    effect = ScaleFluxEffect(flux_scale=0.2)
    model.add_effect(effect, skip_params=True)

    # Check that we sample the value from model node.
    state = model.sample_parameters()
    assert "flux_scale" not in state["my_constant_sed_model"]

    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])
    with pytest.raises(ValueError):
        _ = model.evaluate_sed(times, wavelengths, state)
