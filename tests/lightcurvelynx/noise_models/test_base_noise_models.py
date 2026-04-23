import numpy as np
import pytest
from lightcurvelynx.noise_models.base_noise_models import ConstantFluxNoiseModel
from numpy.testing import assert_allclose


def test_constant_flux_noise_model_init_raises_for_negative_noise_level():
    """Test that initializing ConstantFluxNoiseModel with a negative noise level
    raises a ValueError.
    """
    with pytest.raises(ValueError, match="non-negative"):
        ConstantFluxNoiseModel(noise_level=-1.0)


def test_constant_flux_noise_model_apply_noise_with_seeded_rng_is_deterministic():
    """Test that applying noise with a seeded RNG produces deterministic results."""
    noise_level = 0.5
    bandflux = np.array([1.0, 2.0, 3.0])
    model = ConstantFluxNoiseModel(noise_level=noise_level)

    rng_for_model = np.random.default_rng(1234)
    flux, flux_err = model.apply_noise(bandflux, rng=rng_for_model)
    assert len(np.unique(flux)) == 3, "Expected some variation in flux due to noise."

    rng_for_expected = np.random.default_rng(1234)
    expected_flux = rng_for_expected.normal(loc=bandflux, scale=noise_level)

    assert_allclose(flux, expected_flux)
    assert_allclose(flux_err, np.full_like(bandflux, noise_level))


def test_constant_flux_noise_model_apply_noise_with_zero_noise_level():
    """Test that applying noise with a zero noise level returns the original bandflux."""
    bandflux = np.array([4.5, 0.0, -3.2])
    model = ConstantFluxNoiseModel(noise_level=0.0)

    flux, flux_err = model.apply_noise(bandflux, rng=np.random.default_rng(42))

    assert_allclose(flux, bandflux)
    assert_allclose(flux_err, np.zeros_like(bandflux))
