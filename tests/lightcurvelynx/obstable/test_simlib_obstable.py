import numpy as np
import pytest
from lightcurvelynx.obstable.simlib_obstable import SIMLIBObsTable


def test_simlib_obstable_init():
    """Test initializing SIMLIBObsTable."""
    data = {
        "RA": [10.0, 20.0],
        "DECL": [30.0, 40.0],
        "MJD": [59000.0, 59001.0],
        "FLT": ["g", "r"],
        "ZPTAVG": [25.0, 25.5],
        "PSF1": [1.0, 1.5],
        "SKYSIG": [0.1, 0.2],
        "CCD_GAIN": [2.0, 2.0],
        "PIXSIZE": [0.5, 0.5],
        "CCD_NOISE": [5.0, 5.0],
    }
    survey_data = SIMLIBObsTable(table=data, radius=1.5)

    assert np.allclose(survey_data["ra"], data["RA"])
    assert np.allclose(survey_data["dec"], data["DECL"])
    assert np.allclose(survey_data["time"], data["MJD"])
    assert np.all(survey_data["filter"] == data["FLT"])
    assert np.allclose(survey_data["zp_mag_adu"], data["ZPTAVG"])
    assert np.allclose(survey_data["fwhm_px"], data["PSF1"])
    assert np.allclose(survey_data["sky"], data["SKYSIG"])
    assert np.allclose(survey_data["gain"], data["CCD_GAIN"])
    assert np.allclose(survey_data["pixel_scale"], data["PIXSIZE"])
    assert np.allclose(survey_data["read_noise"], data["CCD_NOISE"])
    assert "zp" in survey_data  # Values will be different because of the conversion to flux units.

    # We fail if no radius is provided.
    with pytest.raises(ValueError):
        _ = SIMLIBObsTable(table=data)
