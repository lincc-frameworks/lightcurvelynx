import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.obstable.spectrograph_table import SpectrographObsTable


def test_create_spectrograph_obstable():
    """Test initializing SpectrographObsTable."""
    data = {
        "ra": [0.0, 10.0, 0.0, 0.0, 10.0, 0.0],
        "dec": [0.0, -10.0, 0.0, 0.0, -10.0, 0.0],
        "time": 53370.5 + np.array([-5.0, -2.0, 0.0, 5.0, 7.0, 10.0]),
    }
    survey_data_table = pd.DataFrame(data)
    survey_data = SpectrographObsTable(table=survey_data_table)

    assert survey_data.default_noise_model is None
    assert survey_data.default_passband_group is None

    assert np.all(survey_data["ra"] == data["ra"])
    assert np.all(survey_data["dec"] == data["dec"])
    assert np.all(survey_data["time"] == data["time"])

    # We fill in a fake filter column. This is not used for anything.
    assert np.all(survey_data["filter"] == "spectra")

    # We have all the attributes set at their default values.
    assert survey_data.survey_values["radius"] == pytest.approx(10.0 / 3600.0)
