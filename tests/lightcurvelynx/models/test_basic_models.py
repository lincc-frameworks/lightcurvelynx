import random

import numpy as np
import pytest
from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.models.basic_models import (
    ConstantSEDModel,
    LinearWavelengthModel,
    SinWaveModel,
    StepModel,
)
from lightcurvelynx.utils.extrapolate import ConstantPadding, LinearDecay


def _sampler_fun(magnitude, offset=0.0, **kwargs):
    """Return a random value between 0 and magnitude.

    Parameters
    ----------
    magnitude : float
        The range of brightness magnitude
    offset : float, optional
        The offset.
    **kwargs : dict, optional
        Absorbs additional parameters
    """
    return magnitude * random.random() + offset


def test_constant_sed_model() -> None:
    """Test that we can sample and create a ConstantSEDModel object."""
    model = ConstantSEDModel(brightness=10.0, node_label="my_constant_sed_model")
    state = model.sample_parameters()
    assert model.get_param(state, "brightness") == 10.0
    assert model.get_param(state, "ra") is None
    assert model.get_param(state, "dec") is None
    assert model.get_param(state, "distance") is None
    assert str(model) == "my_constant_sed_model"

    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])

    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (6, 3)
    assert np.all(values == 10.0)

    # We can set a value we have already added.
    model.set_parameter("brightness", 5.0)
    values = model.evaluate_sed(times, wavelengths)
    assert values.shape == (6, 3)
    assert np.all(values == 5.0)


def test_constant_sed_model_pytree():
    """Test that the PyTree only contains brightness."""
    model = ConstantSEDModel(brightness=10.0, node_label="my_constant_sed_model")
    state = model.sample_parameters()

    pytree = model.build_pytree(state)
    assert pytree["my_constant_sed_model"]["brightness"] == 10.0
    assert pytree["my_constant_sed_model"]["t0"] is None
    assert len(pytree["my_constant_sed_model"]) == 2
    assert len(pytree) == 1


def test_constant_sed_model_host() -> None:
    """Test that we can sample and create a ConstantSEDModel object with properties
    derived from the host object."""
    host = ConstantSEDModel(brightness=15.0, ra=1.0, dec=2.0, distance=3.0)
    model = ConstantSEDModel(brightness=10.0, ra=host.ra, dec=host.dec, distance=host.distance)
    state = model.sample_parameters()

    assert model.get_param(state, "brightness") == 10.0
    assert model.get_param(state, "ra") == 1.0
    assert model.get_param(state, "dec") == 2.0
    assert model.get_param(state, "distance") == 3.0
    assert str(model) == "ConstantSEDModel_0"

    # Test that we have given a different name to the host.
    assert str(host) == "ConstantSEDModel_1"


def test_constant_sed_model_resample() -> None:
    """Check that we can call resample on the model parameters."""
    model = ConstantSEDModel(brightness=_sampler_fun, magnitude=100.0)

    num_samples = 100
    values = np.zeros((num_samples, 1))
    for i in range(num_samples):
        state = model.sample_parameters()
        values[i] = model.get_param(state, "brightness")

    # Check that the values fall within the expected bounds.
    assert np.all(values >= 0.0)
    assert np.all(values <= 100.0)

    # Check that the values are not all the same.
    assert not np.all(values == values[0])


def test_step_model() -> None:
    """Test that we can sample and create a StepModel object."""
    host = ConstantSEDModel(brightness=150.0, ra=1.0, dec=2.0, distance=3.0)
    model = StepModel(brightness=15.0, t0=1.0, t1=2.0, ra=host.ra, dec=host.dec, distance=host.distance)
    state = model.sample_parameters()

    param_values = model.get_local_params(state)
    assert param_values["brightness"] == 15.0
    assert param_values["t0"] == 1.0
    assert param_values["t1"] == 2.0
    assert param_values["ra"] == 1.0
    assert param_values["dec"] == 2.0
    assert param_values["distance"] == 3.0

    times = np.array([0.0, 1.0, 2.0, 3.0])
    wavelengths = np.array([100.0, 200.0])
    expected = np.array([[0.0, 0.0], [15.0, 15.0], [15.0, 15.0], [0.0, 0.0]])

    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (4, 2)
    assert np.array_equal(values, expected)


def test_step_model_resample() -> None:
    """Check that we can call resample on the model parameters."""
    random.seed(1111)

    model = StepModel(
        brightness=FunctionNode(_sampler_fun, magnitude=100.0),
        t0=0.0,
        t1=FunctionNode(_sampler_fun, magnitude=5.0, offset=1.0),
    )

    num_samples = 1000
    brightness_vals = np.zeros((num_samples, 1))
    t_end_vals = np.zeros((num_samples, 1))
    t_start_vals = np.zeros((num_samples, 1))
    for i in range(num_samples):
        state = model.sample_parameters()
        brightness_vals[i] = model.get_param(state, "brightness")
        t_end_vals[i] = model.get_param(state, "t1")
        t_start_vals[i] = model.get_param(state, "t0")

    # Check that the values fall within the expected bounds.
    assert np.all(brightness_vals >= 0.0)
    assert np.all(brightness_vals <= 100.0)
    assert np.all(t_start_vals == 0.0)
    assert np.all(t_end_vals >= 1.0)
    assert np.all(t_end_vals <= 6.0)

    # Check that the expected values are close.
    assert abs(np.mean(brightness_vals) - 50.0) < 5.0
    assert abs(np.mean(t_end_vals) - 3.5) < 0.5

    # Check that the brightness or end values are not all the same.
    assert not np.all(brightness_vals == brightness_vals[0])
    assert not np.all(t_end_vals == t_end_vals[0])


def test_sin_wave_model() -> None:
    """Test that we can sample and create a SinWaveModel object."""
    model = SinWaveModel(amplitude=15.0, frequency=1.0, t0=0.0)
    state = model.sample_parameters()

    times = np.array([0.0, 1.0 / 12.0, 1.0 / 4.0, 5.0 / 12.0, 0.5])
    wavelengths = np.array([100.0, 200.0])
    expected = np.array([[0.0, 0.0], [7.5, 7.5], [15.0, 15.0], [7.5, 7.5], [0.0, 0.0]])

    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (5, 2)
    assert np.allclose(values, expected)

    model2 = SinWaveModel(brightness=10.0, amplitude=15.0, frequency=1.0, t0=0.0)
    state2 = model2.sample_parameters()
    expected2 = np.array([[10.0, 10.0], [17.5, 17.5], [25.0, 25.0], [17.5, 17.5], [10.0, 10.0]])

    values2 = model2.evaluate_sed(times, wavelengths, state2)
    assert values2.shape == (5, 2)
    assert np.allclose(values2, expected2)


def test_linear_wavelength_model() -> None:
    """Test that we can sample and create a LinearWavelengthModel object."""
    model = LinearWavelengthModel(linear_base=1.0, linear_scale=0.1)
    state = model.sample_parameters()

    times = np.arange(0.0, 10.0, 0.5)
    wavelengths = np.array([0.0, 1000.0, 2000.0])
    expected = np.tile(np.array([1.0, 101.0, 201.0]), (len(times), 1))

    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (20, 3)
    assert np.allclose(values, expected)


def test_linear_wavelength_model_redshift() -> None:
    """Test that we correctly apply a redshift to the wavelengths."""
    model = LinearWavelengthModel(linear_base=1.0, linear_scale=0.1, redshift=0.2, t0=0.0)
    state = model.sample_parameters()

    times = np.arange(0.0, 10.0, 0.5)
    wavelengths = np.array([0.0, 1000.0, 1111.1, 1500.0, 2000.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (20, 5)

    # The shifted (rest frame) wavelengths are given by the equation:
    #   rest_frame_wavelengths = observer_frame_wavelengths / (1 + redshift)
    # so we evaluate the linear function at those shifted wavelengths. Then scale
    # those fluxes by (1 + redshift) to get back to the observer frame.
    rest_frame_wavelengths = wavelengths / (1.2)
    expected_single_time = 1.2 * (1.0 + 0.1 * rest_frame_wavelengths)
    expected = np.tile(expected_single_time, (len(times), 1))
    assert np.allclose(values, expected)


def test_linear_wavelength_model_bounds() -> None:
    """Test that we correctly apply a redshift to the wavelengths."""
    model = LinearWavelengthModel(linear_base=1.0, linear_scale=0.1, min_wave=1000.0, max_wave=2000.0)
    state = model.sample_parameters()

    times = np.arange(0.0, 10.0, 0.5)
    wavelengths = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    with pytest.warns(UserWarning):
        values = model.evaluate_sed(times, wavelengths, state)

    # Without any extrapolation we query the model at the points anyway.
    expected = np.tile(np.array([51.0, 101.0, 151.0, 201.0, 251.0]), (len(times), 1))
    assert np.allclose(values, expected)

    # We fill in with a constant value.
    model2 = LinearWavelengthModel(
        linear_base=1.0,
        linear_scale=0.1,
        min_wave=1000.0,
        max_wave=2000.0,
        wave_extrapolation=ConstantPadding(value=100.0),
    )
    state2 = model2.sample_parameters()
    values2 = model2.evaluate_sed(times, wavelengths, state2)
    expected2 = np.tile(np.array([100.0, 101.0, 151.0, 201.0, 100.0]), (len(times), 1))
    assert np.allclose(values2, expected2)

    # We linearly decay the values.
    model3 = LinearWavelengthModel(
        linear_base=1.0,
        linear_scale=0.1,
        min_wave=1000.0,
        max_wave=2000.0,
        wave_extrapolation=LinearDecay(decay_width=100.0),
    )
    wavelengths3 = np.array([500.0, 950.0, 1000.0, 1500.0, 2000.0, 2050.0, 2300.0])
    state3 = model3.sample_parameters()
    values3 = model3.evaluate_sed(times, wavelengths3, state3)
    expected3 = np.tile(np.array([0.0, 50.5, 101.0, 151.0, 201.0, 100.5, 0.0]), (len(times), 1))
    assert np.allclose(values3, expected3)
