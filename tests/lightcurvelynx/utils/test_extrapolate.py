import numpy as np
import pytest
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.utils.extrapolate import (
    ConstantPadding,
    ExponentialDecay,
    LastValue,
    LinearDecay,
    LinearFit,
    LinearFitOnMag,
    ZeroPadding,
)


def test_flux_extrapolation_model():
    """Test the base class for the extrapolation methods."""
    # Create an instance of the base class
    extrapolator = ZeroPadding()

    # Test that extrapolating along wavelength returns a zero matrix
    last_wave = 1000.0
    last_flux = np.array([100.0, 200.0, 300.0])
    query_waves = np.array([1000.0, 1025.0, 1050.0, 1200.0])
    result = extrapolator.extrapolate_wavelength(last_wave, last_flux, query_waves)
    expected_result = np.zeros((3, 4))
    np.testing.assert_allclose(result, expected_result)

    # Test that we get a correctly shaped 2-D array back even if we have
    # only one query point.
    result = extrapolator.extrapolate_wavelength(last_wave, last_flux, np.array([1025.0]))
    assert result.shape == (3, 1)

    # Test that extrapolating along time returns a zero matrix
    last_time = 0.0
    last_flux = np.array([100.0, 200.0, 300.0])
    query_times = np.array([0.0, 1.0, 2.0, 5.0])
    result = extrapolator.extrapolate_time(last_time, last_flux, query_times)
    expected_result = np.zeros((4, 3))
    np.testing.assert_allclose(result, expected_result)

    # Test that we get a correctly shaped 2-D array back even if we have
    # only one query point.
    result = extrapolator.extrapolate_time(last_time, last_flux, np.array([1.0]))
    assert result.shape == (1, 3)


def test_constant_extrapolation():
    """Test that the constant extrapolation function works."""
    # Create an instance of the ConstantPadding class
    extrapolator = ConstantPadding(value=100.0)

    # Test extrapolation past the last valid point.
    last_wave = 1000.0
    last_flux = np.array([100.0, 200.0, 300.0])
    query_waves = np.array([1000.0, 1025.0, 1050.0, 1200.0])
    expected_flux = np.full((3, 4), 100.0)
    result = extrapolator.extrapolate_wavelength(last_wave, last_flux, query_waves)
    np.testing.assert_allclose(result, expected_flux)

    # Test that extrapolating along time returns a zero matrix
    last_time = 0.0
    last_flux = np.array([100.0, 200.0, 300.0])
    query_times = np.array([0.0, 1.0, 2.0, 5.0])
    result = extrapolator.extrapolate_time(last_time, last_flux, query_times)
    expected_result = np.full((4, 3), 100.0)
    np.testing.assert_allclose(result, expected_result)

    # Test that we fail if the value is not positive
    with pytest.raises(ValueError):
        _ = ConstantPadding(value=-1)


def test_last_value_extrapolation():
    """Test that the last value extrapolation function works."""
    # Create an instance of the LastValue class
    extrapolator = LastValue()

    # Test extrapolation past the last valid point.
    last_wave = 1000.0
    last_flux = np.array([100.0, 200.0, 300.0])
    query_waves = np.array([1000.0, 1025.0, 1050.0, 1200.0])
    expected_flux = np.array(
        [
            [100.0, 100.0, 100.0, 100.0],
            [200.0, 200.0, 200.0, 200.0],
            [300.0, 300.0, 300.0, 300.0],
        ]
    )
    result = extrapolator.extrapolate_wavelength(last_wave, last_flux, query_waves)
    np.testing.assert_allclose(result, expected_flux)

    # Test extrapolation along time.
    first_time = 0.0
    last_flux = np.array([100.0, 200.0, 300.0])
    query_times = np.array([-2.0, -1.0])
    expected_flux = np.array(
        [
            [100.0, 200.0, 300.0],
            [100.0, 200.0, 300.0],
        ]
    )
    result = extrapolator.extrapolate_time(first_time, last_flux, query_times)
    np.testing.assert_allclose(result, expected_flux)


def test_linear_decay_extrapolate():
    """Test that the linear decay function works."""
    decay_width = 100.0
    extrapolator = LinearDecay(decay_width=decay_width)

    # Test extrapolation past the last valid point.
    last_wave = 1000.0
    last_flux = np.array([100.0, 200.0, 300.0])
    query_waves = np.array([1000.0, 1025.0, 1050.0, 1075.0, 1100.0, 1150.0])
    expected_flux = np.array(
        [
            [100.0, 75.0, 50.0, 25.0, 0.0, 0.0],
            [200.0, 150.0, 100.0, 50.0, 0.0, 0.0],
            [300.0, 225.0, 150.0, 75.0, 0.0, 0.0],
        ]
    )
    result = extrapolator.extrapolate_wavelength(last_wave, last_flux, query_waves)
    np.testing.assert_allclose(result, expected_flux)

    # Test extrapolation before the first valid point.
    query_waves = np.array([1000.0, 975.0, 950.0, 925.0, 900.0, 850.0])
    result = extrapolator.extrapolate_wavelength(last_wave, last_flux, query_waves)
    np.testing.assert_allclose(result, expected_flux)

    # Test that we fail if the decay width is not positive
    with pytest.raises(ValueError):
        _ = LinearDecay(decay_width=-1.0)


def test_exponential_decay_extrapolate():
    """Test that the exponential decay function works."""
    extrapolator = ExponentialDecay(rate=0.1)

    # Test extrapolation past the last valid point.
    last_wave = 1000.0
    last_flux = np.array([100.0, 200.0, 300.0])
    query_waves = np.array([1000.0, 1025.0, 1050.0, 1075.0, 1100.0, 1150.0])

    t0_flux = 100.0 * np.exp([-0.0, -2.5, -5.0, -7.5, -10.0, -15.0])
    expected_flux = np.vstack((t0_flux, t0_flux * 2, t0_flux * 3))
    result = extrapolator.extrapolate_wavelength(last_wave, last_flux, query_waves)
    np.testing.assert_allclose(result, expected_flux)

    # Test extrapolation before the first valid point.
    query_waves = np.array([1000.0, 975.0, 950.0, 925.0, 900.0, 850.0])
    result = extrapolator.extrapolate_wavelength(last_wave, last_flux, query_waves)
    np.testing.assert_allclose(result, expected_flux)

    # Test that we fail if the decay rate is not positive
    with pytest.raises(ValueError):
        _ = ExponentialDecay(rate=-0.1)


def test_linear_fit_extrapolate():
    """Test that the linear fit extrapolation works."""

    extrapolator = LinearFit()

    last_times = np.array([30.0, 40.0, 50.0, 55.0])
    last_fluxes = np.repeat([[300.0], [250.0], [200.0], [175]], 5, axis=1)
    query_times = np.array([60.0, 70.0, 80.0])
    expected_flux = np.repeat([[150.0], [100.0], [50.0]], 5, axis=1)
    result = extrapolator.extrapolate_time(last_times, last_fluxes, query_times)
    np.testing.assert_allclose(result, expected_flux)

    last_waves = np.array([30.0, 40.0, 50.0, 55.0])
    last_fluxes = np.repeat([[300.0], [250.0], [200.0], [175]], 5, axis=1).T
    query_waves = np.array([60.0, 70.0, 80.0])
    expected_flux = np.repeat([[150.0], [100.0], [50.0]], 5, axis=1).T
    result = extrapolator.extrapolate_wavelength(last_waves, last_fluxes, query_waves)
    np.testing.assert_allclose(result, expected_flux)


def test_linear_fit_on_mag_extrapolate():
    """Test that the linear fit extrapolation works."""

    extrapolator = LinearFitOnMag()

    last_times = np.array([30.0, 40.0, 50.0, 55.0])
    last_fluxes = mag2flux(np.repeat([[21.0], [21.2], [21.4], [21.5]], 5, axis=1))
    query_times = np.array([60.0, 70.0, 80.0])
    expected_flux = mag2flux(np.repeat([[21.6], [21.8], [22.0]], 5, axis=1))
    result = extrapolator.extrapolate_time(last_times, last_fluxes, query_times)
    np.testing.assert_allclose(result, expected_flux)

    last_waves = np.array([30.0, 40.0, 50.0, 55.0])
    last_fluxes = mag2flux(np.repeat([[21.0], [21.2], [21.4], [21.5]], 5, axis=1)).T
    query_waves = np.array([60.0, 70.0, 80.0])
    expected_flux = mag2flux(np.repeat([[21.6], [21.8], [22.0]], 5, axis=1)).T
    result = extrapolator.extrapolate_wavelength(last_waves, last_fluxes, query_waves)
    np.testing.assert_allclose(result, expected_flux)
