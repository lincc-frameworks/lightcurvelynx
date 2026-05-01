import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.obstable.skymapper_obstable import SkyMapperObsTable, SkyMapperPoissonFluxNoiseModel


def test_skymapper_obstable_init():
    """Test initializing SkyMapperObsTable."""
    survey_data = {
        "image_id": [20180101162801, 20171215114730, 20171222125008, 20171222142830, 20171223140648],
        "ccd": [25, 32, 29, 32, 29],
        "ra_deg": [314.32684, 310.79913, 311.33861, 312.05752, 310.9527],
        "dec_deg": [-86.77332, -88.21006, -88.16419, -88.24684, -88.18184],
        "pa_deg": [300.0, 180.0, 90.0, 180.0, 90.0],
        "filter": ["r", "g", "g", "r", "v"],
        "mjd_midpt": [58119.6863, 58102.4915, 58109.5351, 58109.6033, 58110.5882],
        "exp_time": [30.0, 30.0, 30.0, 30.0, 30.0],
        "zeropoint": [26.979, 27.103, 27.176, 27.054, 24.444],
        "zeropoint_stdv": [0.015, 0.012, 0.011, 0.011, 0.047],
        "sb_mag": [17.894, 21.522, 21.506, 20.941, 22.176],
        "fwhm": [2.77, 3.62, 3.34, 2.94, 3.89],
        "elong": [1.109, 1.121, 1.081, 1.07, 1.077],
    }
    survey_data_table = pd.DataFrame(survey_data)
    obs_table = SkyMapperObsTable(table=survey_data_table, make_detector_footprint=True)
    noise_model = SkyMapperPoissonFluxNoiseModel()

    assert "zp" in obs_table
    assert np.allclose(survey_data["ra_deg"], obs_table["ra"])
    assert np.allclose(survey_data["dec_deg"], obs_table["dec"])
    assert np.allclose(survey_data["mjd_midpt"], obs_table["time"])

    # We have all the attributes set at their default values.
    assert obs_table.survey_values["dark_current"] == 0.0
    assert obs_table.survey_values["gain"] == pytest.approx(0.75)
    assert obs_table.survey_values["pixel_scale"] == pytest.approx(0.497)
    assert obs_table.survey_values["read_noise"] == pytest.approx(10.0)
    assert obs_table.survey_values["survey_name"] == "SkyMapper"

    # We successfully built the detector footprint.
    assert obs_table.uses_footprint()

    # We can compute errors.
    clean_flux = mag2flux(np.full(3, 19.0))
    new_vals, err_vals = noise_model.apply_noise(
        clean_flux,
        obs_table=obs_table,
        indices=np.array([0, 1, 2]),  # Index values for the first three rows
    )
    assert not np.any(new_vals == clean_flux)
    assert np.all(err_vals > 0.0)

    # We can build a MOC at level 14
    moc = obs_table.build_moc(max_depth=14)
    assert moc.flatten().size == 5
