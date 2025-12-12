import numpy as np
import pandas as pd
from lightcurvelynx import _LIGHTCURVELYNX_TEST_DATA_DIR
from lightcurvelynx.obstable.roman_obstable import RomanObsTable


def make_random_apt_table():
    """Make random APT table for testing."""
    table = {
        "RA": 0.0,
        "DEC": 0.0,
        "PA": 0.0,
        "BANDPASS": "F129",
        "MA_TABLE_NUMBER": 1001,
        "DURATION": 100,
        "PLAN": 1,
        "PASS": 1,
        "SEGMENT": 1,
        "OBSERVATION": 1,
        "VISIT": 1,
        "EXPOSURE": 1,
        "EXPOSURE_TIME": 60.0,
    }
    return pd.DataFrame(table, index=[0])


def test_roman_obstable_init():
    """Test initializing RomanObsTable."""
    apt_table = make_random_apt_table()
    roman_obstable = RomanObsTable(apt_table)

    assert {"zp", "N_Eff_Pix", "zodi_countrate_min", "thermal_countrate"}.issubset(roman_obstable.columns)


def test_calculate_skynoise():
    """Test calculate_skynoise function."""
    apt_table = make_random_apt_table()
    roman_obstable = RomanObsTable(
        apt_table,
        mat_table_path=_LIGHTCURVELYNX_TEST_DATA_DIR
        / "roman_characterization/roman_wfi_imaging_multiaccum_tables_with_exptime.csv",
    )

    zodi_scale = [1.0, 10.0]
    zodi_countrate = [0.0, 0.1]
    thermal_countrate = [0.1, 0.0]
    exptime = 10.0
    expected_skyvariance = [1.0, 10.0]
    skyvariance = roman_obstable.calculate_skynoise(exptime, zodi_scale, zodi_countrate, thermal_countrate)

    np.testing.assert_allclose(skyvariance, expected_skyvariance)


def test_noise_calculation():
    """Test that the noise calucation returns the right range."""
    apt_table = make_random_apt_table()
    roman_obstable = RomanObsTable(
        apt_table,
        mat_table_path=_LIGHTCURVELYNX_TEST_DATA_DIR
        / "roman_characterization/roman_wfi_imaging_multiaccum_tables_with_exptime.csv",
    )

    roman_obstable.survey_values["zodi_level"] = 2.0

    mag = np.array([24.5])
    expected_magerr = np.array([0.1])

    flux_nJy = np.power(10.0, -0.4 * (mag - 31.4))

    fluxerr_nJy = roman_obstable.bandflux_error_point_source(flux_nJy, 0)
    magerr = 1.086 * fluxerr_nJy / flux_nJy

    np.testing.assert_allclose(magerr, expected_magerr, rtol=0.2)
