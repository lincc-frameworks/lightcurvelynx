import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.noise_models.base_noise_models import PoissonFluxNoiseModel
from lightcurvelynx.obstable.lsst_obstable import LSSTObsTable
from lightcurvelynx.utils.io_utils import read_sqlite_table

_expected_gain = 1.595


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
    assert ops_data.survey_values["gain"] == _expected_gain
    assert ops_data.survey_values["pixel_scale"] == 0.2
    assert ops_data.survey_values["radius"] == 1.75
    assert ops_data.survey_values["read_noise"] == 5.82
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

    # We filter out any of the noise columns with NaN.
    ccd_visit_table.loc[5, "seeing"] = np.nan
    ccd_visit_table.loc[7, "zeroPoint"] = np.nan
    ccd_visit_table.loc[9, "skyBg"] = np.nan
    ccd_visit_table.loc[11, "pixelScale"] = np.nan
    obs_table_with_nan = LSSTObsTable.from_ccdvisit_table(ccd_visit_table)
    assert len(obs_table_with_nan) == 176


def test_lsst_obstable_from_sv_visits():
    """Test that we can read an LSSTObsTable from the SV visits table."""
    values = {
        "exp_midpt_mjd": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0, 15.0, 30.0, 15.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
        "sky_bg_median": np.ones(8),
        "zero_point_median": 25.0 * np.ones(8),
        "filter": np.array(["a", "b", "c", "d", "e", "f", "g", "h"]),
    }
    pdf = pd.DataFrame(values)

    # Make one of the sky_bg_median and one of the zero_point_median values NaN
    # to test that we can handle missing data.
    pdf.loc[2, "sky_bg_median"] = np.nan
    pdf.loc[4, "zero_point_median"] = np.nan

    ops_data = LSSTObsTable.from_sv_visits_table(pdf)
    assert len(ops_data) == 6
    assert ops_data.radius == pytest.approx(1.75)

    # Check that we updated the indices and filters correctly.
    assert np.allclose(
        ops_data.get_value_per_row("time", indices=[0, 1, 2, 3, 5]),
        [0.0, 1.0, 3.0, 5.0, 7.0],
    )
    assert set(ops_data.filters) == set(["a", "b", "d", "f", "g", "h"])
    assert ops_data._spatial_data.n == 6


def test_lsst_noise_model_delegation():
    """Test that LSST point-source noise is delegated to the configured noise model."""
    values = {
        "expMidptMJD": np.array([0.0, 1.0, 2.0]),
        "ra": np.array([15.0, 30.0, 15.0]),
        "dec": np.array([-10.0, -5.0, 0.0]),
        "zp": np.array([1.0e12, 1.1e12, 1.2e12]),
        "seeing": np.array([0.9, 1.0, 1.1]),
        "skyBg": np.array([1200.0, 1300.0, 1100.0]),
        "expTime": np.array([30.0, 30.0, 30.0]),
    }
    table = LSSTObsTable(pd.DataFrame(values))

    bandflux = np.array([200.0, 300.0, 400.0])
    indices = np.array([0, 1, 2])

    # We can compute errors.
    noise_model = PoissonFluxNoiseModel()
    new_vals, err_vals = noise_model.apply_noise(
        bandflux,
        obs_table=table,
        indices=indices,
    )
    assert not np.any(new_vals == bandflux)
    assert np.all(err_vals > 0.0)


def test_reading_lsst_obstable_from_ccdvisits(test_data_dir):
    """Test that we can read an LSSTObsTable a CCDVisits parquet."""
    pdf = pd.read_parquet(test_data_dir / "dp1_ccdvisit_subsampled.parquet")
    total_obs = len(pdf)
    assert "zeroPoint" in pdf.columns

    # The derived columns are not in the original table.
    assert "psf_footprint" not in pdf.columns
    assert "sky_bg_e" not in pdf.columns

    obs_table = LSSTObsTable.from_ccdvisit_table(pdf)
    assert total_obs == len(obs_table)
    assert set(obs_table.filters) == {"g", "i", "r", "u", "y", "z"}

    # Check that we have the expected columns.
    assert "time" in obs_table
    assert "ra" in obs_table
    assert "dec" in obs_table
    assert "filter" in obs_table
    assert "zp" in obs_table
    assert "maglim" in obs_table
    assert "seeing" in obs_table
    assert "radius" in obs_table

    # Check the derived columns
    assert "psf_footprint" in obs_table

    assert "sky_bg_e" in obs_table
    assert np.all(obs_table["sky_bg_e"] > 0.0)
    assert np.allclose(obs_table["sky_bg_e"], obs_table["sky_bg_adu"] * _expected_gain)

    # Check that we have everything we need to derive the PoissonFluxNoiseModel.
    noise_model = PoissonFluxNoiseModel()
    assert noise_model.check_compatibility(obs_table, fail_on_incompatible=True)

    # Check that we can search the table for observations at a given location.
    # Over half of the observations in the fake data set are near (ra=53, dec=-28)
    inds = obs_table.range_search(53.0, -28.0, radius=3.0)
    assert 0.5 * total_obs < len(inds) < 0.8 * total_obs

    # If we at time bounds, we can restrict a lot further.
    inds2 = obs_table.range_search(53.0, -28.0, radius=3.0, t_min=60610.0, t_max=60611.0)
    assert len(inds2) < 0.2 * len(inds)


def test_reading_lsst_obstable_from_sv(test_data_dir):
    """Test that we can read an LSSTObsTable a SV database."""
    table = read_sqlite_table(
        test_data_dir / "sv_db_subsampled.db",
        sql_query="SELECT * FROM observations",
    )
    total_obs = len(table)
    assert "zero_point_median" in table.columns

    # The derived columns are not in the original table.
    assert "psf_footprint" not in table.columns
    assert "sky_bg_e" not in table.columns

    obs_table = LSSTObsTable.from_sv_visits_table(table)
    assert total_obs == len(obs_table)
    assert set(obs_table.filters) == {"g", "i", "r", "u", "y", "z"}

    # Check that we have the expected columns.
    assert "time" in obs_table
    assert "ra" in obs_table
    assert "dec" in obs_table
    assert "filter" in obs_table
    assert "zp" in obs_table
    assert "maglim" in obs_table
    assert "seeing" in obs_table
    assert "radius" in obs_table

    # Check the derived columns
    assert "psf_footprint" in obs_table

    # Check that we have everything we need to derive the PoissonFluxNoiseModel.
    noise_model = PoissonFluxNoiseModel()
    assert noise_model.check_compatibility(obs_table, fail_on_incompatible=True)

    # Check that we can search the table for observations at a given location.
    # We should have between 10% and 20% of the observations near (ra=225, dec=-38).
    inds = obs_table.range_search(225.0, -38.0, radius=3.0)
    assert 0.1 * total_obs < len(inds) < 0.2 * total_obs

    # If we at time bounds, we can restrict a lot further.
    inds2 = obs_table.range_search(225.0, -38.0, radius=3.0, t_min=60855.0, t_max=60860.0)
    assert len(inds2) < 0.5 * len(inds)
