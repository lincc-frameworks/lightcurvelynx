import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.obstable.ztf_obstable import ZTFObsTable, create_random_ztf_obs_data


def test_ztf_obstable_init():
    """Test initializing ZTFObsTable."""
    survey_data_table = create_random_ztf_obs_data(100)
    survey_data = ZTFObsTable(table=survey_data_table)

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
    ]
    values = {
        "obsdate": dates,
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
    }

    # We fail if we do not have the other columns needed:
    # "maglim", "sky", "fwhm", "exptime"
    with pytest.raises(ValueError):
        _ = ZTFObsTable(values)

    values["exptime"] = 0.005 * np.ones(5)
    values["maglim"] = 20.0 * np.ones(5)
    values["scibckgnd"] = np.ones(5)
    values["fwhm"] = 2.3 * np.ones(5)
    survey_data = ZTFObsTable(values)

    assert "zp" in survey_data
    assert np.all(survey_data["zp"] >= 0.0)


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
    fluxerr_nJy = survey_data.bandflux_error_point_source(flux_nJy, 0)
    magerr = 1.086 * fluxerr_nJy / flux_nJy

    np.testing.assert_allclose(magerr, expected_magerr, rtol=0.2)
