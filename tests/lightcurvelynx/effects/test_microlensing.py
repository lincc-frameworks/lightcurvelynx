import numpy as np
from lightcurvelynx.effects.microlensing import Microlensing
from lightcurvelynx.models.basic_models import ConstantSEDModel


def test_microlensing() -> None:
    """Test that we can apply a basic microlensing effect."""
    num_times = 50
    num_waves = 20
    times = np.arange(num_times, dtype=float)

    microlensing_t0 = 5.0
    microlensing = Microlensing(microlensing_t0=microlensing_t0, u_0=0.1, t_E=10.0)

    # Apply at the wavelength level. Note that when calling apply() manually, we
    # need to pass along the parameter values.
    wave_fluxes = np.full((num_times, num_waves), 100.0)
    values = microlensing.apply(
        wave_fluxes,
        times=times,
        apply_microlensing=True,
        microlensing_t0=microlensing_t0,
        u_0=0.1,
        t_E=10.0,
    )

    # Check that the fluxes are all magnified (>100) and that they increase
    # before t0 and decrease after it.
    assert values.shape == (num_times, num_waves)
    assert np.all(values >= 100.0)
    for wave_idx in range(num_waves):
        assert np.all(np.diff(values[0:5, wave_idx]) >= 0)
        assert np.all(np.diff(values[5:num_times, wave_idx]) <= 0)


def test_microlensing_bandflux() -> None:
    """Test that we can apply a basic microlensing effect at the bandflux level."""
    num_times = 50
    times = np.arange(num_times, dtype=float)

    microlensing_t0 = 5.0
    microlensing = Microlensing(microlensing_t0=microlensing_t0, u_0=0.1, t_E=10.0)

    # Apply at the wavelength level. Note that when calling apply() manually, we
    # need to pass along the parameter values.
    bandfluxes = np.full(num_times, 100.0)
    values = microlensing.apply_bandflux(
        bandfluxes,
        times=times,
        apply_microlensing=True,
        microlensing_t0=microlensing_t0,
        u_0=0.1,
        t_E=10.0,
    )

    # Check that the fluxes are all magnified (>100) and that they increase
    # before t0 and decrease after it.
    assert values.shape == (num_times,)
    assert np.all(values >= 100.0)
    assert np.all(np.diff(values[0:5]) >= 0)
    assert np.all(np.diff(values[5:num_times]) <= 0)


def test_constant_sed_microlensing() -> None:
    """Test that we can apply microlensing to a ConstantSEDModel."""
    num_times = 50
    microlensing_t0 = 5.0
    times = np.arange(num_times, dtype=float)

    model = ConstantSEDModel(brightness=10.0, node_label="my_constant_sed_model")
    microlensing = Microlensing(microlensing_t0=microlensing_t0, u_0=0.1, t_E=10.0)
    model.add_effect(microlensing)

    state = model.sample_parameters()
    wavelengths = np.array([1000.0, 2000.0, 3000.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (num_times, 3)

    # Check that the fluxes are all magnified (>100) and that they increase
    # before t0 and decrease after it.
    assert values.shape == (num_times, 3)
    assert np.all(values >= 10.0)
    for wave_idx in range(3):
        assert np.all(np.diff(values[0:5, wave_idx]) >= 0)
        assert np.all(np.diff(values[5:num_times, wave_idx]) <= 0)


def test_microlensing_constant_sed_probability() -> None:
    """Test that we can apply a basic microlensing effect with a probability."""
    num_times = 50
    microlensing_t0 = 5.0
    times = np.arange(num_times, dtype=float)

    model = ConstantSEDModel(brightness=10.0, node_label="model")
    microlensing = Microlensing(
        microlensing_t0=microlensing_t0,
        u_0=0.1,
        t_E=10.0,
        probability=0.5,
    )
    model.add_effect(microlensing)

    rgn_info = np.random.default_rng(101)
    state = model.sample_parameters(num_samples=100, rng_info=rgn_info)
    wavelengths = np.array([3000.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (100, num_times, 1)

    # Check that the effect is applied to about 50% of the samples.
    applied = state["model"]["apply_microlensing"]
    assert np.sum(applied) > 30
    assert np.sum(applied) < 70
    assert np.all(values[applied, 5, 0] >= 10.0)
    assert np.all(values[~applied, 5, 0] == 10.0)
