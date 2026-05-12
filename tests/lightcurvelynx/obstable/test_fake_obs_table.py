import numpy as np
import pandas as pd
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
