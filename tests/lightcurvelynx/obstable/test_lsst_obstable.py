import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.obstable.lsst_obstable import (
    LSSTObsTable,
    _lsstcam_zeropoint_per_sec_zenith,
)


def test_create_lsst_obstable():
    """Create a minimal LSSTObsTable object and perform basic queries."""
    values = {
        "expMidptMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    ops_data = LSSTObsTable(pdf)
    assert len(ops_data) == 5
    assert len(ops_data.columns) == 4

    # We have all the attributes set at their default values.
    assert ops_data.survey_values["dark_current"] == 0.022
    assert ops_data.survey_values["gain"] == 1.595
    assert ops_data.survey_values["pixel_scale"] == 0.2
    assert ops_data.survey_values["radius"] == 1.75
    assert ops_data.survey_values["read_noise"] == 5.82
    assert ops_data.survey_values["zp_per_sec"] == _lsstcam_zeropoint_per_sec_zenith
    assert ops_data.survey_values["survey_name"] == "LSST"

    # Check that we can extract the time bounds.
    t_min, t_max = ops_data.time_bounds()
    assert t_min == 0.0
    assert t_max == 4.0

    # We can query which columns the LSSTObsTable has by new or old name.
    assert "ra" in ops_data
    assert "dec" in ops_data
    assert "time" in ops_data
    assert "expMidptMJD" in ops_data

    # We can access columns directly as though it was a table.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["expMidptMJD"])
    assert np.allclose(ops_data["expMidptMJD"], values["expMidptMJD"])

    # Without a filters column we cannot access the filters.
    assert len(ops_data.filters) == 0


def test_create_lsst_obstable_override_fail():
    """Test that we fail if we do not have the information needed to create the zeropoints."""
    values = {
        "expMidptMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }

    with pytest.raises(ValueError):
        _ = LSSTObsTable(values, ext_coeff=None)


def _make_fake_data(times):
    """Create 9 fake ccd pointings at each timestep using
    a 3x3 grid centered (like dp1) with grid steps of 0.22 deg.

    Parameters
    ----------
    times : `numpy.ndarray`
        The times of the observations (in MJD).
    """

    # Create a tile of pointings around a random center at each time
    # where the center is chosen from a 1 degree by 1 degree box.
    data = {
        "expMidptMJD": [],
        "ra": [],
        "dec": [],
        "band": [],
    }
    for t in times:
        pointing_ra = np.random.uniform(150.0, 151.0)
        pointing_dec = np.random.uniform(-21.0, -20.0)
        band = np.random.choice(["g", "r", "i"])

        for dra in [-0.22, 0.0, 0.22]:
            for ddec in [-0.22, 0.0, 0.22]:
                data["expMidptMJD"].append(t)
                data["ra"].append(pointing_ra + dra)
                data["dec"].append(pointing_dec + ddec)
                data["band"].append(band)

    # Add the other columns with random values.
    num_samples = len(data["expMidptMJD"])
    data["ccdVisitId"] = np.arange(num_samples)
    data["expTime"] = np.full(num_samples, 30.0)  # seconds
    data["magLim"] = np.random.normal(24.0, 0.5, num_samples)  # mag
    data["seeing"] = np.random.normal(1.0, 0.2, num_samples)  # arcseconds
    data["skyBg"] = np.random.normal(1750, 10.0, num_samples)  # adu
    data["skyNoise"] = np.random.normal(43.0, 1.0, num_samples)  # adu
    data["skyRotation"] = np.zeros(num_samples)  # degrees
    data["pixelScale"] = np.full(num_samples, 0.2)  # arcsec/pixel
    data["xSize"] = np.full(num_samples, 4000)  # pixels
    data["ySize"] = np.full(num_samples, 4000)  # pixels
    data["zeroPoint"] = np.random.normal(31.0, 0.2, num_samples)  # mag

    return pd.DataFrame(data)


def test_lsst_obstable_from_ccdvisit():
    """Test that we can create an LSSTObsTable from a CCD visit database."""
    times = 60623.25 + np.arange(20) * 0.1  # 20 time steps (15 minutes apart)
    ccd_visit_table = _make_fake_data(times)
    obs_table = LSSTObsTable.from_ccdvisit_table(ccd_visit_table)
    assert len(obs_table) == 180  # Number of rows in the test ccdvisit table

    # Check that we have the expected columns.
    assert "time" in obs_table
    assert "ra" in obs_table
    assert "dec" in obs_table
    assert "filter" in obs_table
    assert "zp" in obs_table
    assert "maglim" in obs_table
    assert "seeing" in obs_table
    assert "rotation" in obs_table
    assert "radius" in obs_table

    # Check that we have the expected survey radius values.
    assert np.all(obs_table["radius"] >= 0.15)
    assert np.all(obs_table["radius"] <= 0.16)
    assert np.all(obs_table["pixel_scale"] >= 0.19)
    assert np.all(obs_table["pixel_scale"] <= 0.21)

    # No footprint is created by default.
    assert obs_table._detector_footprint is None

    # If we create a footprint, it is set correctly.
    obs_table_with_footprint = LSSTObsTable.from_ccdvisit_table(
        ccd_visit_table,
        make_detector_footprint=True,
    )
    assert obs_table_with_footprint._detector_footprint is not None
