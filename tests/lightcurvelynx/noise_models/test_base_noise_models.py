import numpy as np
import pytest
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.noise_models.base_noise_models import (
    ConstantFluxNoiseModel,
    FiveSigmaDepthNoiseModel,
    GivenNoiseModel,
    PoissonFluxNoiseModel,
)
from lightcurvelynx.noise_models.noise_utils import poisson_bandflux_std

# Local helper class.
from lookup_only_obstable import LookupOnlyObsTable
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
    assert np.array_equal(model.required_values, [])

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
    assert set(model.required_values) == {
        "exptime",
        "sky_bg_e",
        "psf_footprint",
        "zp",
        "read_noise",
        "dark_current",
    }

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

    # Check that the model is compatible with our ObsTable.
    assert model.check_compatibility(obs_table, fail_on_incompatible=True)

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

    # Check that the model is compatible with our ObsTable even though
    # it is missing the "nexposure" and "zp_err_mag" columns (which have defaults).
    assert model.check_compatibility(obs_table, fail_on_incompatible=True)

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


def test_poisson_flux_noise_model_missing():
    """Test that the PoissonFluxNoiseModel fails the compatibility check when
    required columns are missing.
    """
    model = PoissonFluxNoiseModel()
    dummy_data = {
        "nexposure": np.array([1, 2, 1]),
        "sky_bg_e": np.array([100.0, 110.0, 120.0]),
        "psf_footprint": np.array([2.0, 2.5, 3.0]),
        "zp": np.array([25.0, 26.0, 27.0]),
        "read_noise": np.array([4.0, 4.5, 5.0]),
        "dark_current": np.array([0.01, 0.02, 0.03]),
        "zp_err_mag": np.array([0.001, 0.002, 0.003]),
    }
    obs_table = LookupOnlyObsTable(dummy_data)

    with pytest.raises(ValueError):
        model.check_compatibility(obs_table, fail_on_incompatible=True)


def test_poisson_flux_noise_model_nans():
    """Test that the PoissonFluxNoiseModel fails the compatibility check when
    required columns contain NaNs.
    """
    model = PoissonFluxNoiseModel()
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
    assert model.check_compatibility(obs_table)

    # Introduce a NaN into a required column.
    dummy_data["exptime"][0] = np.nan
    obs_table = LookupOnlyObsTable(dummy_data)
    assert not model.check_compatibility(obs_table)
    with pytest.raises(ValueError):
        model.check_compatibility(obs_table, fail_on_incompatible=True)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_check_compatibility_rejects_non_finite_numeric_required_values(invalid_value):
    """Required numeric values must be finite for compatibility checks."""
    model = GivenNoiseModel()
    obs_table = LookupOnlyObsTable({"bandflux_error": np.array([1.0, invalid_value, 3.0])})

    assert not model.check_compatibility(obs_table)
    with pytest.raises(ValueError):
        model.check_compatibility(obs_table, fail_on_incompatible=True)


def test_given_noise_model():
    """Test that the GivenNoiseModel correctly reads per-row flux errors
    from the ObsTable and applies noise to the bandflux."""
    model = GivenNoiseModel()
    assert set(model.required_values) == {"bandflux_error"}

    bandflux = np.array([100.0, 200.0, 300.0])
    dummy_data = {
        "bandflux_error": np.array([5.0, 10.0, 15.0]),
    }
    obs_table = LookupOnlyObsTable(dummy_data)

    # We fail without an ObsTable, without indices, or with mismatched indices length.
    with pytest.raises(ValueError, match="ObsTable must be provided"):
        model.apply_noise(bandflux, indices=np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="Indices must be provided"):
        model.apply_noise(bandflux, obs_table=obs_table)
    with pytest.raises(ValueError):
        model.apply_noise(bandflux, obs_table=obs_table, indices=np.array([0, 1]))

    # Check compatibility.
    assert model.check_compatibility(obs_table, fail_on_incompatible=True)

    # Verify that noise is applied using per-row errors from the ObsTable.
    rng_for_model = np.random.default_rng(42)
    flux, flux_err = model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),
        rng=rng_for_model,
    )
    assert_allclose(flux_err, dummy_data["bandflux_error"])

    rng_for_expected = np.random.default_rng(42)
    expected_flux = rng_for_expected.normal(loc=bandflux, scale=dummy_data["bandflux_error"])
    assert_allclose(flux, expected_flux)


def test_given_noise_model_per_filter():
    """Test that the GivenNoiseModel correctly reads per-filter flux errors
    from the ObsTable and applies noise to the bandflux."""
    model = GivenNoiseModel()
    assert set(model.required_values) == {"bandflux_error"}

    # We make bandflux error constant per filter.
    filters = np.array(["g", "r", "i", "g", "r", "i", "r", "g", "r"])
    bandflux_error = {"g": 5.0, "r": 10.0, "i": 15.0}
    obs_table = LookupOnlyObsTable(
        table_values={"filter": filters},
        const_values={"bandflux_error": bandflux_error},
    )

    bandflux = np.full_like(filters, 100.0, dtype=float)

    # We fail without an ObsTable, without indices, or with mismatched indices length.
    with pytest.raises(ValueError, match="ObsTable must be provided"):
        model.apply_noise(bandflux, indices=np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="Indices must be provided"):
        model.apply_noise(bandflux, obs_table=obs_table)
    with pytest.raises(ValueError):
        model.apply_noise(bandflux, obs_table=obs_table, indices=np.array([0, 1]))

    # Check compatibility.
    assert model.check_compatibility(obs_table, fail_on_incompatible=True)

    # Verify that noise is applied using per-row errors from the ObsTable.
    rng_for_model = np.random.default_rng(42)
    flux, flux_err = model.apply_noise(
        bandflux,
        obs_table=obs_table,
        indices=np.arange(len(filters)),
        rng=rng_for_model,
    )

    # Check that we pull the correct error for each row.
    err_arr = np.array([bandflux_error[filt] for filt in filters])
    assert_allclose(flux_err, err_arr)

    rng_for_expected = np.random.default_rng(42)
    expected_flux = rng_for_expected.normal(loc=bandflux, scale=err_arr)
    assert_allclose(flux, expected_flux)


def test_five_sigma_depth_noise_model():
    """Test that the FiveSigmaDepthNoiseModel correctly computes flux errors
    and applies noise to the bandflux."""
    model = FiveSigmaDepthNoiseModel()
    assert set(model.required_values) == {"five_sigma_depth"}

    table_values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
        "five_sigma_depth": 20.0 + np.arange(5),
    }
    ops_data = LookupOnlyObsTable(table_values=table_values)
    assert len(ops_data) == 5

    # Create and apply the noise model.
    noise_model = FiveSigmaDepthNoiseModel()
    assert noise_model.check_compatibility(ops_data, fail_on_incompatible=True)

    bandflux = np.array([1000.0, 2000.0, 1500.0, 2500.0, 3000.0])
    rng_for_model = np.random.default_rng(2024)
    flux, flux_err = noise_model.apply_noise(
        bandflux,
        obs_table=ops_data,
        indices=np.array([0, 1, 2, 3, 4]),
        rng=rng_for_model,
    )
    assert not np.any(bandflux == flux)

    expected_bandflux_error = mag2flux(ops_data["five_sigma_depth"].to_numpy()) / 5.0
    assert np.allclose(flux_err, expected_bandflux_error)


def test_five_sigma_depth_noise_model_nans():
    """Test that the FiveSigmaDepthNoiseModel correctly fails compatibility checks
    when the five_sigma_depth column contains NaNs."""
    model = FiveSigmaDepthNoiseModel()
    assert set(model.required_values) == {"five_sigma_depth"}

    table_values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
        "five_sigma_depth": np.array([20.0, 21.0, np.nan, 23.0, 24.0]),
    }
    ops_data = LookupOnlyObsTable(table_values=table_values)
    assert len(ops_data) == 5

    # Create and apply the noise model.
    noise_model = FiveSigmaDepthNoiseModel()
    assert not noise_model.check_compatibility(ops_data)
