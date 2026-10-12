import os

import jax.numpy as jnp
import pytest

from lightcurvelynx.astro_utils.salt2_color_law import (
    _SALT2CL_B,
    _SALT2CL_V,
    _WAVESCALE,
    SALT2ColorLaw,
)


def test_salt2_color_law():
    """Test that we can create and query a simple SALT2ColorLaw object."""
    test_cl = SALT2ColorLaw(_SALT2CL_B, _SALT2CL_V, [0.5, 0.25, 0.1])

    # The coefficients are shifted (with 1 - sum(others) prepended) and zero padded.
    assert jnp.allclose(test_cl.coeffs, jnp.asarray([0.15, 0.5, 0.25, 0.1, 0.0, 0.0, 0.0]))
    assert test_cl.scaled_wave_min == pytest.approx(0.0)
    assert test_cl.scaled_wave_max == pytest.approx(1.0)
    assert test_cl.value_at_min == pytest.approx(0.0)
    assert test_cl.value_at_max == pytest.approx(1.0)
    assert test_cl.deriv_at_min == pytest.approx(0.15)
    assert test_cl.deriv_at_max == pytest.approx(2.3)

    # Test that we can compute the results for some manually specified points.
    half_step = (_SALT2CL_V - _SALT2CL_B) / 2.0
    test_wavelengths = [
        _SALT2CL_B - half_step,  # Shifted and scaled -> -0.5
        _SALT2CL_B,  # Shifted and scaled -> 0.0
        _SALT2CL_B + half_step,  # Shifted and scaled -> 0.5
        _SALT2CL_V,  # Shifted and scaled -> 1.0
        _SALT2CL_V + half_step,  # Shifted and scaled -> 1.5
    ]
    results = test_cl.apply(test_wavelengths)

    assert results[0] == pytest.approx(0.0 + 0.15 * 0.5)
    assert results[1] == pytest.approx(0.0)
    assert results[2] == pytest.approx(-0.2375)
    assert results[3] == pytest.approx(-1.0)
    assert results[4] == pytest.approx(-1.0 - 2.3 * 0.5)


def test_salt2_color_law_load(test_data_dir):
    """Test loading a Salt2ColorLaw object from a file."""
    filename = os.path.join(test_data_dir, "truncated-salt2-h17/salt2_color_correction.dat")
    test_cl = SALT2ColorLaw.from_file(filename)

    expected_coeffs = [1.83205876, -1.33154627, 0.61225710, -0.12117791, 0.00840832, 0.0, 0.0]
    assert jnp.allclose(test_cl.coeffs, jnp.asarray(expected_coeffs))
    assert test_cl.scaled_wave_min == pytest.approx((2800 - _SALT2CL_B) * _WAVESCALE)
    assert test_cl.scaled_wave_max == pytest.approx((9500 - _SALT2CL_B) * _WAVESCALE)
