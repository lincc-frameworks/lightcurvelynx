import numpy as np
import pytest
from astropy.cosmology import WMAP9, Planck18
from lightcurvelynx.astro_utils.redshift import (
    RedshiftDistFunc,
    obs_to_rest_times_waves,
    redshift_to_distance,
    rest_to_obs_flux,
)
from lightcurvelynx.models.basic_models import StepModel


def test_obs_to_rest_times_waves() -> None:
    """Test that we correctly convert observer frame times and wavelengths to rest frame."""
    observer_times = np.array([10.0, 20.0, 30.0])
    observer_waves = np.array([4000.0, 5000.0, 6000.0])
    t0 = 5.0

    # Test with redshift = 0. This should be a no-op.
    rest_times, rest_waves = obs_to_rest_times_waves(observer_times, observer_waves, 0.0, t0)
    assert np.all(rest_times == observer_times)
    assert np.all(rest_waves == observer_waves)

    # Test with redshift = 1. Times should decrease (relative to t0) and
    # wavelengths should halve.
    rest_times, rest_waves = obs_to_rest_times_waves(observer_times, observer_waves, 1.0, t0)
    expected_rest_times = np.array([7.5, 12.5, 17.5])
    expected_rest_waves = np.array([2000.0, 2500.0, 3000.0])
    assert np.allclose(rest_times, expected_rest_times)
    assert np.allclose(rest_waves, expected_rest_waves)

    # Check that we fail if redshift is negative.
    with pytest.raises(ValueError):
        obs_to_rest_times_waves(observer_times, observer_waves, -0.5, t0)


def test_rest_to_obs_flux() -> None:
    """Test that we correctly convert rest frame flux densities to observer frame."""
    rest_flux = np.array([10.0, 20.0, 30.0])
    redshift = 0.8
    obs_flux = rest_to_obs_flux(rest_flux, redshift)
    expected_obs_flux = rest_flux * (1 + redshift)
    assert np.allclose(obs_flux, expected_obs_flux)

    # Check that we fail if redshift is negative.
    with pytest.raises(ValueError):
        rest_to_obs_flux(rest_flux, -0.5)


def test_redshifted_flux_densities() -> None:
    """Test that we correctly calculate redshifted values."""
    times = np.linspace(0, 100, 1000)
    wavelengths = np.array([100.0, 200.0, 300.0])
    t0 = 10.0
    t1 = 30.0
    brightness = 50.0

    for redshift in [0.0, 0.5, 2.0, 3.0, 30.0]:
        model_redshift = StepModel(brightness=brightness, t0=t0, t1=t1, redshift=redshift)
        values_redshift = model_redshift.evaluate_sed(times, wavelengths)

        for i, time in enumerate(times):
            if t0 <= time and time <= (t1 - t0) * (1 + redshift) + t0:
                # Note that the multiplication by (1+z) is due to the fact we are working in f_nu
                # units, instead of f_lambda units and may be unintuitive for users who are used to
                # working in f_lambda units. This factor can be derived by equaling the integrated
                # flux in f_nu unit before and after redshift is applied.
                assert np.all(values_redshift[i] == brightness * (1 + redshift))
            else:
                assert np.all(values_redshift[i] == 0.0)


def test_redshift_to_distance():
    """Test that we can convert the redshift to a distance using a given cosmology."""
    wmap9_val = redshift_to_distance(1100, cosmology=WMAP9)
    planck18_val = redshift_to_distance(1100, cosmology=Planck18)

    assert abs(planck18_val - wmap9_val) > 1000.0
    assert 13.0 * 1e12 < wmap9_val < 16.0 * 1e12
    assert 13.0 * 1e12 < planck18_val < 16.0 * 1e12

    # We fail with invalid redshift or no cosmology.
    with pytest.raises(ValueError):
        redshift_to_distance(-0.1, cosmology=WMAP9)
    with pytest.raises(ValueError):
        redshift_to_distance(1.0, cosmology=None)


def test_redshift_dist_func_node():
    """Test the RedshiftDistFunc node."""
    node = RedshiftDistFunc(redshift=1100, cosmology=Planck18)
    state = node.sample_parameters()
    assert 13.0 * 1e12 < node.get_param(state, "function_node_result") < 16.0 * 1e12

    # Test that we can generate multiple samples.
    state = node.sample_parameters(num_samples=10)
    assert np.all(node.get_param(state, "function_node_result") > 13.0 * 1e12)
    assert np.all(node.get_param(state, "function_node_result") < 16.0 * 1e12)
