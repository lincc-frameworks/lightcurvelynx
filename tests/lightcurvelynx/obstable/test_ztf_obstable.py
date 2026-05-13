import numpy as np
import pandas as pd
from lightcurvelynx.noise_models.base_noise_models import PoissonFluxNoiseModel
from lightcurvelynx.obstable.ztf_obstable import ZTFObsTable, create_random_ztf_obs_data


def test_ztf_obstable_init():
    """Test initializing ZTFObsTable."""
    survey_data_table = create_random_ztf_obs_data(100)
    survey_data = ZTFObsTable(table=survey_data_table)
    assert isinstance(survey_data.default_noise_model, PoissonFluxNoiseModel)

    assert "zp" in survey_data
    assert "time" in survey_data

    # We have all the attributes set at their default values.
    assert survey_data.survey_values["dark_current"] == 0.0
    assert survey_data.survey_values["gain"] == 6.2
    assert survey_data.survey_values["pixel_scale"] == 1.01
    assert survey_data.survey_values["radius"] == 3.868
    assert survey_data.survey_values["read_noise"] == 8
    assert survey_data.survey_values["survey_name"] == "ZTF"


def test_create_ztf_obstable_override():
    """Test that we can override the default survey values."""
    survey_data_table = create_random_ztf_obs_data(100)

    survey_data = ZTFObsTable(
        table=survey_data_table,
        dark_current=0.1,
        gain=7.1,
        pixel_scale=0.1,
        radius=1.0,
        read_noise=5.0,
    )

    # We have all the attributes set at their default values.
    assert survey_data.survey_values["dark_current"] == 0.1
    assert survey_data.survey_values["gain"] == 7.1
    assert survey_data.survey_values["pixel_scale"] == 0.1
    assert survey_data.survey_values["radius"] == 1.0
    assert survey_data.survey_values["read_noise"] == 5.0


def test_create_ztf_obstable_no_zp():
    """Create an survey_data without a zeropoint column."""
    dates = [
        "2020-01-01 12:00:00.000",
        "2020-01-02 12:00:00.000",
        "2020-01-03 12:00:00.000",
        "2020-01-04 12:00:00.000",
        "2020-01-05 12:00:00.000",
        "2020-01-06 12:00:00.000",
        "2020-01-07 12:00:00.000",
        "2020-01-08 12:00:00.000",
    ]
    values = {
        "obsdate": dates,
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0, 45.0, 30.0, 15.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
        "filter": np.array(["r", "g", "r", "g", "i", "g", "r", "g"]),
    }

    values["exptime"] = 0.005 * np.ones(8)
    values["maglim"] = 20.0 * np.ones(8)
    values["scibckgnd"] = np.ones(8)
    values["fwhm"] = 2.3 * np.ones(8)

    # Make one of the fwhm values invalid to test that it gets dropped.
    values["fwhm"][4] = np.nan

    survey_data = ZTFObsTable(values)
    assert len(survey_data) == 7  # One dropped value due to invalid fwhm.

    # We derived the zeropoint column.
    assert "zp" in survey_data
    assert np.all(survey_data["zp"] >= 0.0)

    # The indexing, filter list, and size of the spatial data structure are all
    # correct after dropping the invalid value.
    assert np.allclose(
        survey_data.get_value_per_row("ra", indices=[0, 1, 2, 3, 4, 6]),
        np.array([15.0, 30.0, 15.0, 0.0, 45.0, 15.0]),
    )
    assert set(survey_data.filters) == {"g", "r"}
    assert survey_data._spatial_data.n == 7


def test_noise_calculation():
    """Test that the noise calculation is in the right range."""
    mag = np.array([19.0])
    expected_magerr = np.array([0.1])

    flux_nJy = np.power(10.0, -0.4 * (mag - 31.4))
    survey_data = ZTFObsTable(
        table=pd.DataFrame(
            {
                "ra": 0.0,
                "dec": 0.0,
                "scibckgnd": 200.0,
                "maglim": 20.0,
                "fwhm": 2.3,
                "exptime": 30.0,
                "obsdate": "2020-01-01 12:00:00.000",
            },
            index=[0],
        )
    )

    noise_model = PoissonFluxNoiseModel()
    flux, fluxerr_nJy = noise_model.apply_noise(
        flux_nJy,
        obs_table=survey_data,
        indices=[0],
    )
    assert not np.any(flux == flux_nJy)

    magerr = 1.086 * fluxerr_nJy / flux_nJy
    np.testing.assert_allclose(magerr, expected_magerr, rtol=0.2)
