import numpy as np
import pytest
from lightcurvelynx.noise_models.base_noise_models import (
    ConstantFluxNoiseModel,
    PoissonFluxNoiseModel,
)
from lightcurvelynx.noise_models.noise_utils import poisson_bandflux_std
from numpy.testing import assert_allclose


class LookupOnlyObsTable:
    """A simple dummy class to simulate the ObsTable's get_value_per_row method."""

    def __init__(self, values):
        self.values = values

    def get_value_per_row(self, key, indices, default=None):
        """Simulate the ObsTable's get_value_per_row method."""
        if key in self.values:
            return np.asarray(self.values[key])[indices]
        if default is not None:
            return np.full(len(indices), default)
        raise KeyError(f"Missing required key: {key}")


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


def test_poisson_flux_noise_model():
    """Test that the PoissonFluxNoiseModel correctly computes flux errors
    and applies noise to the bandflux."""
    model = PoissonFluxNoiseModel()
    bandflux = np.array([100.0, 200.0, 200.0])
    dummy_data = {
        "exptime": np.array([30.0, 35.0, 40.0]),
        "nexposure": np.array([1, 2, 1]),
        "sky_bg_e": np.array([100.0, 110.0, 120.0]),
        "psf_footprint": np.array([2.0, 2.5, 3.0]),
        "zp": np.array([25.0, 26.0, 27.0]),
        "read_noise": np.array([4.0, 4.5, 5.0]),
        "dark_current": np.array([0.01, 0.02, 0.03]),
        "zp_err_mag": np.array([0.001, 0.002, 0.003]),
    }
    obs_table = LookupOnlyObsTable(dummy_data)

    # We fail without an ObsTable, without indices, or with mismatched
    # indices length.
    with pytest.raises(ValueError, match="ObsTable must be provided"):
        model.apply_noise(bandflux, indices=np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="Indices must be provided"):
        model.apply_noise(bandflux, obs_table=obs_table)
    with pytest.raises(ValueError):
        model.apply_noise(
            bandflux,
            obs_table=obs_table,
            indices=np.array([0, 1]),
        )

    # Compute the expected flux error using the same features that
    # the model will extract from the ObsTable.
    expected_flux_err = poisson_bandflux_std(
        bandflux,
        total_exposure_time=dummy_data["exptime"],
        exposure_count=dummy_data["nexposure"],
        psf_footprint=dummy_data["psf_footprint"],
        sky=dummy_data["sky_bg_e"],
        zp=dummy_data["zp"],
        readout_noise=dummy_data["read_noise"],
        dark_current=dummy_data["dark_current"],
        zp_err_mag=dummy_data["zp_err_mag"],
    )
    assert np.all(expected_flux_err > 0), "Expected positive flux errors."

    rng_for_model = np.random.default_rng(2024)
    flux, flux_err = model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),
        rng=rng_for_model,
    )
    assert not np.any(bandflux == flux)
    assert_allclose(flux_err, expected_flux_err)

    rng_for_expected = np.random.default_rng(2024)
    expected_flux = rng_for_expected.normal(loc=bandflux, scale=expected_flux_err)
    assert_allclose(flux, expected_flux)


def test_poisson_flux_noise_model_defaults():
    """Test that the PoissonFluxNoiseModel correctly computes flux errors
    when some of the values are not given."""
    model = PoissonFluxNoiseModel()
    bandflux = np.array([500.0, 400.0, 400.0])
    dummy_data = {
        "exptime": np.array([31.0, 36.0, 41.0]),
        "sky_bg_e": np.array([105.0, 105.0, 115.0]),
        "psf_footprint": np.array([1.5, 2.1, 2.2]),
        "zp": np.array([25.5, 26.5, 27.5]),
        "read_noise": np.array([4.0, 4.5, 5.0]),
        "dark_current": np.array([0.01, 0.02, 0.03]),
    }
    obs_table = LookupOnlyObsTable(dummy_data)

    # Compute the expected flux error using the same features that
    # the model will extract from the ObsTable.
    expected_flux_err = poisson_bandflux_std(
        bandflux,
        total_exposure_time=dummy_data["exptime"],
        exposure_count=1,  # default
        psf_footprint=dummy_data["psf_footprint"],
        sky=dummy_data["sky_bg_e"],
        zp=dummy_data["zp"],
        readout_noise=dummy_data["read_noise"],
        dark_current=dummy_data["dark_current"],
        zp_err_mag=0.0,  # default
    )
    assert np.all(expected_flux_err > 0), "Expected positive flux errors."

    rng_for_model = np.random.default_rng(2024)
    flux, flux_err = model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),
        rng=rng_for_model,
    )
    assert not np.any(bandflux == flux)
    assert_allclose(flux_err, expected_flux_err)

    rng_for_expected = np.random.default_rng(2024)
    expected_flux = rng_for_expected.normal(loc=bandflux, scale=expected_flux_err)
    assert_allclose(flux, expected_flux)
