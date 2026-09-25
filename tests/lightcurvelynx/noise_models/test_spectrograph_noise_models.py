import numpy as np
import pytest
from lightcurvelynx.astro_utils.spectrograph import Spectrograph
from lightcurvelynx.noise_models.spectrograph_noise_models import ConstantSpectrographNoiseModel, SNANANoiseModel

# Local helper class.
from lightcurvelynx_main.tests.lightcurvelynx.conftest import test_data_dir
from lookup_only_obstable import LookupOnlyObsTable
from numpy.testing import assert_allclose


def test_constant_spectrograph_noise_model():
    """Test that we can create and apply a ConstantSpectrographNoiseModel with a positive noise level."""
    noise_level = 0.5
    model = ConstantSpectrographNoiseModel(noise_level=noise_level)
    assert np.array_equal(model.required_values, [])
    assert model.spectrograph is None
    assert model.noise_level == pytest.approx(noise_level)

    # Run once with a seeded RNG to verify that the results are deterministic.
    measurements = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rng_for_model1 = np.random.default_rng(1234)
    flux1, flux_err1 = model.apply_noise(measurements, rng=rng_for_model1)
    assert not np.allclose(
        flux1, measurements
    )  # The flux should be different from the original measurements.
    assert np.allclose(flux_err1, np.full_like(measurements, noise_level))

    # Run again with the same seed to verify that we get the same results.
    rng_for_model2 = np.random.default_rng(1234)
    flux2, flux_err2 = model.apply_noise(measurements, rng=rng_for_model2)
    assert_allclose(flux2, flux1)
    assert_allclose(flux_err2, flux_err1)

    # Run with a different seed to verify that we get different results.
    rng_for_model3 = np.random.default_rng(5678)
    flux3, _ = model.apply_noise(measurements, rng=rng_for_model3)
    assert not np.allclose(flux3, flux1)


def test_constant_spectrograph_noise_model_with_spectrograph():
    """Test that we can create and apply a ConstantSpectrographNoiseModel with a Spectrograph."""
    spectrograph = Spectrograph.from_regular_grid(4000.0, 8000.0, 500.0)
    model_with_spectrograph = ConstantSpectrographNoiseModel(noise_level=0.1, spectrograph=spectrograph)
    assert model_with_spectrograph.spectrograph is spectrograph
    assert model_with_spectrograph.noise_level == pytest.approx(0.1)


def test_constant_spectrograph_noise_model_zero_noise():
    """Test that applying noise with a zero noise level returns the original measurements."""
    noise_level = 0.0
    model = ConstantSpectrographNoiseModel(noise_level=noise_level)

    measurements = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rng_for_model = np.random.default_rng(42)
    flux, flux_err = model.apply_noise(measurements, rng=rng_for_model)

    assert_allclose(flux, measurements)
    assert_allclose(flux_err, np.zeros_like(measurements))


def test_constant_spectrograph_noise_model_fail():
    """Test that initializing ConstantSpectrographNoiseModel with a negative noise level
    raises a ValueError.
    """
    with pytest.raises(ValueError, match="non-negative"):
        ConstantSpectrographNoiseModel(noise_level=-1.0)


def test_constant_spectrograph_noise_model_check_compatibility():
    """Use a ConstantSpectrographNoiseModel with overridden required_values to test
    check_compatibility behavior."""
    model = ConstantSpectrographNoiseModel(noise_level=0.5)
    table_values = {
        "some_good_column": np.array([1, 2, 3]),
        "some_bad_column": np.array([np.nan, np.inf, -np.inf]),
    }
    obs_table = LookupOnlyObsTable(table_values)

    # By default everything passes because there are no required values.
    assert model.check_compatibility(obs_table)

    # We add a required value to the model to test the compatibility check.
    model._required_values = ["some_good_column"]
    assert model.check_compatibility(obs_table)

    model._required_values = ["some_bad_column"]
    assert not model.check_compatibility(obs_table)
    with pytest.raises(ValueError):
        model.check_compatibility(obs_table, fail_on_incompatible=True)

    model._required_values = ["some_missing_column"]
    assert not model.check_compatibility(obs_table)
    with pytest.raises(ValueError):
        model.check_compatibility(obs_table, fail_on_incompatible=True)

def test_snana_noise_model_solve(test_data_dir):
    """Test that the solved (ZP, SQSIGSKY) reproduce the file's SNR at both magrefs
    (SNANA's own self-check, sntools_spectrograph.c:748-754)."""
    snana_file = test_data_dir / "fake_snana_spectrograph_gapless.dat"
    model = SNANANoiseModel(snana_file)
    snr = np.array([[50.01, 100.8, 148.36], [3.14, 4.90, 6.55]])[None] + 0.001 * np.arange(8)[:, None, None]

    for m, magref in enumerate(model.magref):
        flux_pe = 10.0 ** (-0.4 * (magref - model.zp_grid))
        assert_allclose(flux_pe / np.sqrt(model.sqsigsky_grid + flux_pe), snr[:, m, :], rtol=1e-8)

def test_snana_spectrograph_noise_model(test_data_dir):
    """Seeded RNG gives reproducible noise; a different seed gives different noise."""

    obs_table = LookupOnlyObsTable({"exptime": [120.0, 600.0, 3600.0]})
    indices = np.array([2, 0])

    snana_file = test_data_dir / "fake_snana_spectrograph_gapless.dat"
    model = SNANANoiseModel(snana_file)
    flux_density = 100
    sed = np.full((len(indices), len(model.spectrograph.query_waves)), flux_density)
    measurements = model.spectrograph.evaluate(sed)

    flux1, err1 = model.apply_noise(measurements, sed=sed, obs_table=obs_table,
                                    indices=indices, rng=np.random.default_rng(1234))
    flux2, err2 = model.apply_noise(measurements, sed=sed, obs_table=obs_table,
                                    indices=indices, rng=np.random.default_rng(1234))
    flux3, _ = model.apply_noise(measurements, sed=sed, obs_table=obs_table,
                                 indices=indices, rng=np.random.default_rng(5678))

    assert flux1.shape == err1.shape == measurements.shape == (2, model.spectrograph.num_bins)
    assert np.all(err1[0] < err1[1])  # row 0 used 3600 s, row 1 used 120 s
    assert not np.allclose(flux1, measurements, atol=0.0)  # fluxes are ~1e-15, so atol must be 0
    assert_allclose(flux2, flux1)
    assert_allclose(err2, err1)
    assert not np.allclose(flux3, flux1, atol=0.0)
