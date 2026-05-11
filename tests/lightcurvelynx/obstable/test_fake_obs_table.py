import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.noise_models.base_noise_models import ConstantFluxNoiseModel
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable


def test_create_fake_obs_table_consts():
    """Create a minimal FakeObsTable object with given defaults."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    bandflux_error = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(
        pdf,
        bandflux_error=bandflux_error,
        fwhm_px=2.0,
        sky_bg_electrons=100.0,
    )
    assert len(ops_data) == 5

    # The FakeObsTable should have no default noise model or passband group.
    assert ops_data.default_noise_model is None
    assert ops_data.default_passband_group is None

    # We use the defaults when we do not provide values in the table. Not all of these
    # will be added as columns, but we can still retrieve an array of values.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["time"])
    assert np.array_equal(ops_data["filter"], values["filter"])
    assert np.allclose(ops_data.get_value_per_row("fwhm_px"), [2.0] * 5)
    assert np.allclose(ops_data.get_value_per_row("sky_bg_electrons"), [100.0] * 5)
    assert np.allclose(ops_data.get_value_per_row("exptime"), [30.0] * 5)  # Default value
    assert np.allclose(ops_data.get_value_per_row("nexposure"), [1] * 5)  # Default value
    assert np.allclose(
        ops_data.get_value_per_row("bandflux_error"),
        [27.0, 26.0, 27.0, 28.0, 26.0],
    )

    # Check default survey values.
    assert ops_data.survey_values["dark_current"] == 0
    assert ops_data.survey_values["nexposure"] == 1
    assert ops_data.survey_values["sky_bg_electrons"] == 100
    assert ops_data.survey_values["radius"] is None
    assert ops_data.survey_values["read_noise"] == 0
    assert ops_data.survey_values["survey_name"] == "FAKE_SURVEY"

    # We can compute noise (using the default GivenNoiseModel).
    fluxes = np.array([100.0, 200.0, 300.0, 400.0, 500.0])  # Fluxes in nJy
    new_fluxes, flux_error = ops_data.noise_model.apply_noise(
        fluxes,
        obs_table=ops_data,
        indices=np.arange(5),  # Indices of the observations
    )
    assert len(flux_error) == 5
    assert np.all(flux_error > 0)
    assert len(np.unique(flux_error)) > 1  # Not all the same
    assert not np.any(new_fluxes == fluxes)  # Noisy fluxes should not be the same as input


def test_create_fake_obs_table_cols_fail():
    """Test that we raise errors when we do not provide required values."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    # We don't have a noise model or a bandflux error.
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, sky_bg_electrons=100.0)

    # But we could use an alternative noise model if we wanted to.
    obs_table = FakeObsTable(
        pdf,
        noise_model=ConstantFluxNoiseModel(10.0),
        sky_bg_electrons=100.0,
    )
    assert isinstance(obs_table.noise_model, ConstantFluxNoiseModel)
