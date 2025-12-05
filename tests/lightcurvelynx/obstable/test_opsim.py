import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.obstable.opsim import (
    OpSim,
    _lsstcam_extinction_coeff,
    _lsstcam_zeropoint_per_sec_zenith,
    create_random_opsim,
    opsim_add_random_data,
    oversample_opsim,
)


def test_create_opsim():
    """Create a minimal OpSim object and perform basic queries."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    ops_data = OpSim(pdf)
    assert len(ops_data) == 5
    assert len(ops_data.columns) == 4

    # We have all the attributes set at their default values.
    assert ops_data.survey_values["dark_current"] == 0.2
    assert ops_data.survey_values["ext_coeff"] == _lsstcam_extinction_coeff
    assert ops_data.survey_values["pixel_scale"] == 0.2
    assert ops_data.survey_values["radius"] == 1.75
    assert ops_data.survey_values["read_noise"] == 8.8
    assert ops_data.survey_values["zp_per_sec"] == _lsstcam_zeropoint_per_sec_zenith
    assert ops_data.survey_values["survey_name"] == "LSST"

    # Check that we can extract the time bounds.
    t_min, t_max = ops_data.time_bounds()
    assert t_min == 0.0
    assert t_max == 4.0

    # We can query which columns the OpSim has by new or old name.
    assert "ra" in ops_data
    assert "dec" in ops_data
    assert "time" in ops_data
    assert "fieldRA" in ops_data
    assert "fieldDec" in ops_data
    assert "observationStartMJD" in ops_data
    assert "something_random" not in ops_data

    # We can access columns directly as though it was a table.
    assert np.allclose(ops_data["ra"], values["fieldRA"])
    assert np.allclose(ops_data["dec"], values["fieldDec"])
    assert np.allclose(ops_data["time"], values["observationStartMJD"])
    assert np.allclose(ops_data["fieldRA"], values["fieldRA"])
    assert np.allclose(ops_data["fieldDec"], values["fieldDec"])
    assert np.allclose(ops_data["observationStartMJD"], values["observationStartMJD"])

    # Without a filters column we cannot access the filters.
    assert len(ops_data.filters) == 0

    # We can create an OpSim directly from the dictionary as well.
    ops_data2 = OpSim(pdf)
    assert len(ops_data2) == 5
    assert len(ops_data.columns) == 4
    assert np.allclose(ops_data2["ra"], values["fieldRA"])
    assert np.allclose(ops_data2["dec"], values["fieldDec"])
    assert np.allclose(ops_data2["time"], values["observationStartMJD"])

    # We raise an error if we are missing a required row.
    del values["fieldDec"]
    with pytest.raises(KeyError):
        _ = OpSim(values)


def test_create_opsim_override():
    """Test that we can override the default survey values."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    ops_data = OpSim(
        values,
        dark_current=0.1,
        ext_coeff={"u": 0.1, "g": 0.2, "r": 0.3, "i": 0.4, "z": 0.5, "y": 0.6},
        pixel_scale=0.1,
        radius=1.0,
        read_noise=5.0,
        zp_per_sec={"u": 25.0, "g": 26.0, "r": 27.0, "i": 28.0, "z": 29.0, "y": 30.0},
    )

    # We have loaded the non-default values.
    assert ops_data.survey_values["dark_current"] == 0.1
    assert ops_data.survey_values["ext_coeff"] == {"u": 0.1, "g": 0.2, "r": 0.3, "i": 0.4, "z": 0.5, "y": 0.6}
    assert ops_data.survey_values["pixel_scale"] == 0.1
    assert ops_data.survey_values["radius"] == 1.0
    assert ops_data.survey_values["read_noise"] == 5.0
    assert ops_data.survey_values["zp_per_sec"] == {
        "u": 25.0,
        "g": 26.0,
        "r": 27.0,
        "i": 28.0,
        "z": 29.0,
        "y": 30.0,
    }

    # We can access the filters.
    assert set(ops_data.filters) == {"r", "g", "i"}


def test_create_opsim_override_fail():
    """Test that we fail if we do not have the information needed to create the zeropoints."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }

    with pytest.raises(ValueError):
        _ = OpSim(values, ext_coeff=None)


def test_create_opsim_saturation():
    """Test that we can control whether saturation mags are stored."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    # Without any arguments the default saturation mags are used.
    ops_data = OpSim(pdf)
    assert ops_data.uses_saturation()

    # If we set apply_saturation to False, we do not store the saturation mags.
    ops_data = OpSim(pdf, apply_saturation=False)
    assert ops_data.uses_saturation() is False


def test_create_opsim_no_zp():
    """Create an opsim without a zeropoint column."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
    }

    # We fail if we do not have the other columns needed: filter, airmass, exptime.
    with pytest.raises(ValueError):
        _ = OpSim(values)

    values["filter"] = np.array(["r", "g", "r", "i", "z"])
    values["airmass"] = 0.01 * np.ones(5)
    values["visitExposureTime"] = 0.1 * np.ones(5)
    opsim = OpSim(values)

    assert "zp" in opsim
    assert "zp_nJy" in opsim
    assert np.all(opsim["zp"] >= 0.0)
    assert np.all(opsim["zp_nJy"] >= 0.0)


def test_create_opsim_custom_names():
    """Create a minimal OpSim object from alternate column names."""
    values = {
        "custom_time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "custom_ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "custom_dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }

    # Load fails if we use the default colmap.
    with pytest.raises(KeyError):
        _ = OpSim(values)

    # Load succeeds if we pass in a customer dictionary.
    colmap = {"ra": "custom_ra", "dec": "custom_dec", "time": "custom_time", "zp": "zp_nJy"}
    ops_data = OpSim(values, colmap=colmap)
    assert len(ops_data) == 5


def test_opsim_add_columns():
    """Create a minimal OpSim object and perform basic queries."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    ops_data = OpSim(pdf)
    assert len(ops_data) == 5
    assert len(ops_data.columns) == 4

    # We can add a column of constant values from a scalar.
    assert "new_column1" not in ops_data.columns
    ops_data.add_column("new_column1", 10)
    assert "new_column1" in ops_data.columns
    assert np.allclose(ops_data["new_column1"], [10, 10, 10, 10, 10])

    # We can add a column of data from a list or array.
    assert "new_column2" not in ops_data.columns
    ops_data.add_column("new_column2", [1, 2, 3, 4, 5])
    assert "new_column2" in ops_data.columns
    assert np.allclose(ops_data["new_column2"], [1, 2, 3, 4, 5])

    # We fail if we add a column with the incorrect number of values.
    with pytest.raises(ValueError):
        ops_data.add_column("new_column3", [1, 2, 3, 4, 5, 6])

    # We fail if we try to overwrite a current column unless we
    # set overwrite=True.
    with pytest.raises(KeyError):
        ops_data.add_column("new_column1", 12)
    assert np.allclose(ops_data["new_column1"], [10, 10, 10, 10, 10])

    ops_data.add_column("new_column2", 12, overwrite=True)
    assert np.allclose(ops_data["new_column2"], [12, 12, 12, 12, 12])

    # We can add random data.
    opsim_add_random_data(ops_data, "new_column3", min_val=1.0, max_val=5.0)
    values = ops_data["new_column3"]
    assert len(np.unique(values)) == 5
    assert np.all(values >= 1.0)
    assert np.all(values <= 5.0)


def test_opsim_filter_rows():
    """Test that the user can filter out OpSim rows."""
    times = np.arange(0.0, 10.0, 1.0)
    values = {
        "observationStartMJD": times,
        "fieldRA": 15.0 * (times + 1.0),
        "fieldDec": -1.0 * times,
        "zp_nJy": np.ones(10),
        "filter": np.tile(["r", "g"], 5),
    }
    ops_data = OpSim(values)
    assert len(ops_data) == 10
    assert len(ops_data.columns) == 5

    # We can filter the OpSim to specific rows by index.
    inds = [0, 1, 2, 3, 4, 5, 7, 8]
    ops_data = ops_data.filter_rows(inds)
    assert isinstance(ops_data, OpSim)
    assert len(ops_data) == 8
    assert len(ops_data.columns) == 5

    expected_times = np.array(inds)
    assert np.allclose(ops_data["time"], expected_times)
    assert np.allclose(ops_data["ra"], 15.0 * (expected_times + 1.0))
    assert np.allclose(ops_data["dec"], -1.0 * expected_times)
    assert np.array_equal(ops_data["filter"], values["filter"][inds])

    # Check that the size of the internal KD-tree has changed.
    assert ops_data._kd_tree.n == 8

    # We can filter the OpSim to specific rows by mask.
    ops_data = ops_data.filter_rows(ops_data["filter"] == "r")
    assert len(ops_data) == 4

    expected_times = np.array([0.0, 2.0, 4.0, 8.0])
    assert np.allclose(ops_data["time"], expected_times)
    assert np.allclose(ops_data["ra"], 15.0 * (expected_times + 1.0))
    assert np.allclose(ops_data["dec"], -1.0 * expected_times)
    assert np.all(ops_data["filter"] == "r")

    # Check that the size of the internal KD-tree has changed (again).
    assert ops_data._kd_tree.n == 4

    # We throw an error if the mask is the wrong size.
    bad_mask = [True] * (len(ops_data) - 1)
    with pytest.raises(ValueError):
        _ = ops_data.filter_rows(bad_mask)


def test_read_small_opsim(opsim_small):
    """Read in a small OpSim file from the testing data directory."""
    ops_data = OpSim.from_db(opsim_small)
    assert len(ops_data) == 300


def test_write_read_opsim_db():
    """Create a minimal opsim data frame, test that we can write it as a database table,
    and test that we can correctly read it back in."""

    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    ops_data = OpSim(pd.DataFrame(values))

    with tempfile.TemporaryDirectory() as dir_name:
        file_path = Path(dir_name, "test_write_read_opsim.db")

        # The opsim does not exist until we write it.
        assert not file_path.is_file()
        with pytest.raises(FileNotFoundError):
            _ = OpSim.from_db(file_path)

        # We can write the opsim db.
        ops_data.write_db(file_path)
        assert file_path.is_file()

        # We can reread the opsim db.
        ops_data2 = OpSim.from_db(file_path)
        assert len(ops_data2) == 5
        assert np.allclose(values["observationStartMJD"], ops_data2["time"].to_numpy())
        assert np.allclose(values["fieldRA"], ops_data2["ra"].to_numpy())
        assert np.allclose(values["fieldDec"], ops_data2["dec"].to_numpy())

        # We cannot overwrite unless we set overwrite=True
        with pytest.raises(ValueError):
            ops_data.write_db(file_path, overwrite=False)
        ops_data.write_db(file_path, overwrite=True)


def test_write_read_opsim_parquet():
    """Create a minimal opsim data frame, test that we can write it as a parquet file,
    and test that we can correctly read it back in."""

    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    pdf = pd.DataFrame(values)
    pdf.attrs["lightcurvelynx_survey_data"] = {"pixel_scale": 0.001}
    ops_data = OpSim(pdf, radius=10.0)

    with tempfile.TemporaryDirectory() as dir_name:
        file_path = Path(dir_name, "test_write_read_opsim.parquet")

        # The opsim does not exist until we write it.
        assert not file_path.is_file()
        with pytest.raises(FileNotFoundError):
            _ = OpSim.from_parquet(file_path)

        # We can write the opsim db.
        ops_data.write_parquet(file_path)
        assert file_path.is_file()

        # We can reread the opsim db.
        ops_data2 = OpSim.from_parquet(file_path)
        assert len(ops_data2) == 5
        assert np.allclose(values["observationStartMJD"], ops_data2["time"].to_numpy())
        assert np.allclose(values["fieldRA"], ops_data2["ra"].to_numpy())
        assert np.allclose(values["fieldDec"], ops_data2["dec"].to_numpy())
        assert ops_data2.survey_values["pixel_scale"] == 0.001
        assert ops_data2.survey_values["radius"] == 10.0
        assert ops_data2.survey_values["survey_name"] == "LSST"

        # We cannot overwrite unless we set overwrite=True
        with pytest.raises(FileExistsError):
            ops_data.write_parquet(file_path, overwrite=False)
        ops_data.write_parquet(file_path, overwrite=True)


@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore radius warnings
def test_opsim_range_search():
    """Test that we can extract the time, ra, and dec from an opsim data frame."""
    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "fieldRA": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "fieldDec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp_nJy": np.ones(8),
    }
    ops_data = OpSim(values)

    # Test single queries.
    assert set(ops_data.range_search(15.0, 10.0, radius=0.5)) == set([1, 2, 3])
    assert set(ops_data.range_search(25.0, 10.0, radius=0.5)) == set([4, 5])
    assert set(ops_data.range_search(15.0, 10.0, radius=100.0)) == set([0, 1, 2, 3, 4, 5, 6, 7])
    assert set(ops_data.range_search(15.0, 10.0, radius=1e-6)) == set([1])
    assert set(ops_data.range_search(15.02, 10.0, radius=1e-6)) == set()

    # Test that we can filter by time.
    assert set(ops_data.range_search(15.0, 10.0, radius=0.5, t_min=1.0, t_max=3.0)) == set([1, 2, 3])
    assert set(ops_data.range_search(15.0, 10.0, radius=0.5, t_min=2.0, t_max=4.0)) == set([2, 3])
    assert set(ops_data.range_search(15.0, 10.0, radius=0.5, t_min=0.0, t_max=1.0)) == set([1])
    assert set(ops_data.range_search(15.0, 10.0, radius=0.5, t_min=4.0, t_max=5.0)) == set()

    # With no radius provided, it should default to 1.75.
    assert set(ops_data.range_search(15.0, 10.0)) == set([1, 2, 3])
    assert set(ops_data.range_search(25.0, 10.0)) == set([4, 5])

    # Test is_observed() with single queries.
    assert ops_data.is_observed(15.0, 10.0, radius=0.5)
    assert not ops_data.is_observed(15.02, 10.0, radius=1e-6)
    assert ops_data.is_observed(15.0, 10.0, radius=0.5, t_min=1.0, t_max=3.0)
    assert not ops_data.is_observed(15.0, 10.0, radius=0.5, t_min=40.0, t_max=50.0)

    # Test a batched query.
    query_ra = np.array([15.0, 25.0, 15.0])
    query_dec = np.array([10.0, 10.0, 5.0])
    neighbors = ops_data.range_search(query_ra, query_dec, radius=0.5)
    assert len(neighbors) == 3
    assert set(neighbors[0]) == set([1, 2, 3])
    assert set(neighbors[1]) == set([4, 5])
    assert set(neighbors[2]) == set()

    # Do the same query with time filtering.
    t_min = np.array([0.0, 5.0, 0.0])
    t_max = np.array([2.0, 11.0, 1.0])
    neighbors = ops_data.range_search(query_ra, query_dec, radius=0.5, t_min=t_min, t_max=t_max)
    assert len(neighbors) == 3
    assert set(neighbors[0]) == set([1, 2])
    assert set(neighbors[1]) == set([5])
    assert set(neighbors[2]) == set()

    # Test is_observed() with batched queries.
    assert np.array_equal(
        ops_data.is_observed(query_ra, query_dec, radius=0.5),
        np.array([True, True, False]),
    )

    # Test that we fail if bad query arrays are provided.
    with pytest.raises(ValueError):
        _ = ops_data.range_search(None, None, radius=0.5)
    with pytest.raises(ValueError):
        _ = ops_data.range_search([1.0, 2.3], 4.5, radius=0.5)
    with pytest.raises(ValueError):
        _ = ops_data.range_search([1.0, 2.3], [4.5, 6.7, 8.9], radius=0.5)
    with pytest.raises(ValueError):
        _ = ops_data.range_search([1.0, 2.3], [4.5, None], radius=0.5)


@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore radius warnings
def test_opsim_get_observations():
    """Test that we can extract the time, ra, and dec from an opsim data frame."""
    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "fieldRA": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "fieldDec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp_nJy": np.ones(8),
    }
    ops_data = OpSim(values)

    # Test basic queries (all columns).
    obs = ops_data.get_observations(15.0, 10.0, radius=0.5)
    assert len(obs) == 4
    assert np.allclose(obs["time"], [1.0, 2.0, 3.0])

    obs = ops_data.get_observations(25.0, 10.0, radius=0.5)
    assert np.allclose(obs["time"], [4.0, 5.0])

    obs = ops_data.get_observations(15.0, 10.0, radius=100.0)
    assert np.allclose(obs["time"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    obs = ops_data.get_observations(15.0, 10.0, radius=1e-6)
    assert np.allclose(obs["time"], [1.0])

    obs = ops_data.get_observations(15.02, 10.0, radius=1e-6)
    assert len(obs["time"]) == 0

    # Test we can get a subset of columns.
    obs = ops_data.get_observations(15.0, 10.0, radius=0.5, cols=["time", "zp"])
    assert len(obs) == 2
    assert np.allclose(obs["time"], [1.0, 2.0, 3.0])

    # Test we can use the colmap names.
    obs = ops_data.get_observations(15.0, 10.0, radius=0.5, cols=["time", "ra"])
    assert len(obs) == 2
    assert np.allclose(obs["time"], [1.0, 2.0, 3.0])

    # Test we fail with an unrecognized column name.
    with pytest.raises(KeyError):
        _ = ops_data.get_observations(15.0, 10.0, radius=0.5, cols=["time", "custom_col"])


def test_opsim_docstring():
    """Test if OpSim class has a docstring"""
    assert OpSim.__doc__ is not None
    assert len(OpSim.__doc__) > 100


def test_read_opsim_shorten(opsim_shorten):
    """Read in a shorten OpSim file from the testing data directory."""
    ops_data = OpSim.from_db(opsim_shorten)
    assert len(ops_data) == 100


def test_opsim_flux_err_point_source(opsim_shorten):
    """Check if OpSim.flux_err_point_source is consistent with fiveSigmaDepth."""
    ops_data = OpSim.from_db(opsim_shorten)
    # fiveSigmaDepth is the 5-sigma limiting magnitude.
    flux = mag2flux(ops_data["fiveSigmaDepth"])
    expected_flux_err = flux / 5.0

    flux_err = ops_data.bandflux_error_point_source(flux, index=np.arange(len(ops_data)))

    # Tolerance is very high, we should investigate why the values are so different.
    np.testing.assert_allclose(flux_err, expected_flux_err, rtol=0.2)


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


def test_opsim_from_ccdvisit():
    """Test that we can create an OpSim from a CCD visit database."""
    times = 60623.25 + np.arange(20) * 0.1  # 20 time steps (15 minutes apart)
    ccd_visit_table = _make_fake_data(times)
    opsim = OpSim.from_ccdvisit_table(ccd_visit_table)
    assert len(opsim) == 180  # Number of rows in the test ccdvisit table

    # Check that we have the expected columns.
    assert "time" in opsim
    assert "ra" in opsim
    assert "dec" in opsim
    assert "filter" in opsim
    assert "zp" in opsim
    assert "fiveSigmaDepth" in opsim
    assert "seeing" in opsim
    assert "skybrightness" in opsim
    assert "rotation" in opsim
    assert "radius" in opsim

    # Check that we have the expected survey radius values.
    assert np.all(opsim["radius"] >= 0.15)
    assert np.all(opsim["radius"] <= 0.16)
    assert np.all(opsim["pixel_scale"] >= 0.19)
    assert np.all(opsim["pixel_scale"] <= 0.21)

    # No footprint is created by default.
    assert opsim._detector_footprint is None

    # If we create a footprint, it is set correctly.
    opsim_with_footprint = OpSim.from_ccdvisit_table(
        ccd_visit_table,
        make_detector_footprint=True,
    )
    assert opsim_with_footprint._detector_footprint is not None


def test_create_random_opsim():
    """Test that we can create a complete random OpSim."""
    opsim = create_random_opsim(1000)
    assert len(opsim) == 1000


def test_oversample_opsim(opsim_shorten):
    """Test that we can oversample an OpSim file."""
    opsim = OpSim.from_db(opsim_shorten)

    bands = ["g", "r"]
    ra, dec = 205.0, -57.0
    time_range = 60_000.0, 60_010.0
    delta_t = 0.01

    for strategy in ["darkest_sky", "random"]:
        oversampled = oversample_opsim(
            opsim,
            pointing=(ra, dec),
            time_range=time_range,
            delta_t=delta_t,
            bands=bands,
            strategy=strategy,
        )
        assert set(opsim.columns) == set(oversampled.columns), "columns are not the same"
        np.testing.assert_allclose(np.diff(oversampled["time"]), delta_t, err_msg="delta_t is not correct")
        np.testing.assert_allclose(oversampled["ra"], ra, err_msg="RA is not correct")
        np.testing.assert_allclose(oversampled["dec"], dec, err_msg="Dec is not correct")
        assert np.all(oversampled["time"] >= time_range[0]), "time range is not correct"
        assert np.all(oversampled["time"] <= time_range[1]), "time range is not correct"
        assert set(oversampled["filter"]) == set(bands), "oversampled table has the wrong bands"

        n_skybright = oversampled["skybrightness"].unique().size
        n_filters = oversampled["filter"].unique().size
        assert n_skybright >= n_filters, "there should be at least as many skybrightness values as bands"
        assert oversampled["skybrightness"].isna().sum() == 0, "skybrightness has NaN values"

    # Oversampling fails if there are no observations in the time range.
    with pytest.raises(ValueError):
        _ = oversample_opsim(
            opsim,
            pointing=(5.0, 67.0),
            time_range=(0.0, 1.0),
            delta_t=delta_t,
            bands=bands,
            strategy=strategy,
        )

    # Oversampling fails with an invalid strategy.
    with pytest.raises(ValueError):
        _ = oversample_opsim(
            opsim,
            pointing=(ra, dec),
            time_range=(0.0, 1.0),
            delta_t=delta_t,
            bands=bands,
            strategy="invalid",
        )


def test_fixture_oversampled_observations(oversampled_observations):
    """Test the fixture oversampled_observations."""
    assert len(oversampled_observations) == 7_300
    assert set(oversampled_observations["filter"]) == {"g", "r"}
    assert oversampled_observations["skybrightness"].isna().sum() == 0
    assert oversampled_observations["skybrightness"].unique().size >= 2
    assert np.all(oversampled_observations["time"] >= 61406.0)
    assert np.all(oversampled_observations["time"] <= 61771.0)
    np.testing.assert_allclose(oversampled_observations["ra"], 0.0)
    np.testing.assert_allclose(oversampled_observations["dec"], 0.0)
    np.testing.assert_allclose(np.diff(oversampled_observations["time"]), 0.05)
