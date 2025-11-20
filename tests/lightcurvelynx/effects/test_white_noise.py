import numpy as np
import pytest
from lightcurvelynx.effects.white_noise import WhiteNoise
from lightcurvelynx.models.basic_models import ConstantSEDModel


def test_white_noise() -> None:
    """Test that we can sample and create a WhiteNoise object."""
    white_noise = WhiteNoise(white_noise_sigma=0.1)
    assert str(white_noise) == "WhiteNoise"
    assert repr(white_noise) == "WhiteNoise(white_noise_sigma,white_noise_seed)"

    # We can apply the noise.
    values = np.full((5, 3), 100.0)
    values = white_noise.apply(values, white_noise_sigma=0.1)
    assert not np.all(values == 100.0)
    assert np.all(np.abs(values - 100.0) <= 1.0)

    # We can override the default value using the parameters.
    values = white_noise.apply(values, white_noise_sigma=20.0)
    assert not np.all(values == 100.0)
    assert not np.all(np.abs(values - 100.0) <= 1.0)

    # We fail if we do not pass in the expected parameters.
    with pytest.raises(ValueError):
        _ = white_noise.apply(values)


def test_white_noise_bandflux() -> None:
    """Test that we can sample a WhiteNoise object at the bandflux level."""
    values = np.full(10, 100.0)

    # We can apply the noise.
    white_noise = WhiteNoise(white_noise_sigma=0.1)
    values = white_noise.apply_bandflux(values, white_noise_sigma=0.1)
    assert not np.all(values == 100.0)
    assert np.all(np.abs(values - 100.0) <= 1.0)

    # We can override the default value using the parameters.
    values = white_noise.apply_bandflux(values, white_noise_sigma=20.0)
    assert not np.all(values == 100.0)
    assert not np.all(np.abs(values - 100.0) <= 1.0)

    # We fail if we do not pass in the expected parameters.
    with pytest.raises(ValueError):
        _ = white_noise.apply_bandflux(values)


def test_constant_sed_model_white_noise() -> None:
    """Test that we can sample and create a ConstantSEDModel object with white noise."""
    model = ConstantSEDModel(
        brightness=10.0,
        node_label="my_constant_sed_model",
        seed=100,
    )
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 0

    # We can add the white noise effect.
    white_noise = WhiteNoise(white_noise_sigma=0.1)
    model.add_effect(white_noise)
    assert len(model.rest_frame_effects) == 1
    assert len(model.obs_frame_effects) == 0

    state = model.sample_parameters()
    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (6, 3)

    # We get noisy values around 10.0.
    assert len(np.unique(values)) > 10
    assert np.all(np.abs(values - 10.0) < 3.0)

    # Test that if we pass in an rng, we control the randomness.
    values1 = model.evaluate_sed(times, wavelengths, state, rng_info=np.random.default_rng(100))
    values2 = model.evaluate_sed(times, wavelengths, state, rng_info=np.random.default_rng(100))
    assert not np.any(values1 == 10.0)
    assert np.all(values1 == values2)


def test_constant_sed_model_white_noise_obs_frame() -> None:
    """Test that we can make the WhiteNoise an observer frame effect.
    While this does not make physical sense, it allows us to test that code path.
    """
    model = ConstantSEDModel(
        brightness=10.0,
        node_label="my_constant_sed_model",
        seed=100,
    )
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 0

    white_noise = WhiteNoise(rest_frame=False, white_noise_sigma=0.1)
    model.add_effect(white_noise)
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 1

    state = model.sample_parameters()
    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (6, 3)

    # We get noisy values around 10.0.
    assert len(np.unique(values)) > 10
    assert np.all(np.abs(values - 10.0) < 3.0)
