import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint
from lightcurvelynx.obstable.obs_table import ObsTable
from regions import CircleSkyRegion


def test_create_obs_table():
    """Create a minimal ObsTable object and perform basic queries."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    ops_data = ObsTable(pdf)
    assert len(ops_data) == 5
    assert len(ops_data.columns) == 4
    assert not ops_data.uses_footprint()
    assert not ops_data.uses_saturation()

    # We have all the attributes set at their default values (which are None for the base class).
    assert ops_data.survey_values["dark_current"] is None
    assert ops_data.survey_values["ext_coeff"] is None
    assert ops_data.survey_values["pixel_scale"] is None
    assert ops_data.survey_values["radius"] is None
    assert ops_data.survey_values["read_noise"] is None
    assert ops_data.survey_values["zp_per_sec"] is None
    assert ops_data.survey_values["survey_name"] == "Unknown"

    # Check that we can extract the time bounds.
    t_min, t_max = ops_data.time_bounds()
    assert t_min == 0.0
    assert t_max == 4.0

    # We can query which columns the ObsTable has.
    assert set(ops_data.columns) == {"time", "ra", "dec", "zp"}
    assert "time" in ops_data
    assert "ra" in ops_data
    assert "dec" in ops_data
    assert "is_good_obs" not in ops_data

    # We can query which parameter values the ObsTable has.
    assert "dark_current" not in ops_data
    assert "survey_name" in ops_data
    assert "random_param" not in ops_data

    # We can access columns directly as though it was a table.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["time"])

    # Without a filters column we cannot access the filters.
    assert len(ops_data.filters) == 0

    # We can create an ObsTable directly from the dictionary as well.
    ops_data2 = ObsTable(pdf)
    assert len(ops_data2) == 5
    assert len(ops_data.columns) == 4
    assert np.allclose(ops_data2["ra"], values["ra"])
    assert np.allclose(ops_data2["dec"], values["dec"])
    assert np.allclose(ops_data2["time"], values["time"])

    # We raise an error if we are missing a required row.
    del values["dec"]
    with pytest.raises(KeyError):
        _ = ObsTable(values)


def test_create_obs_table_override():
    """Test that we can override the default obs table values."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    ops_data = ObsTable(
        values,
        dark_current=0.1,
        ext_coeff={"u": 0.1, "g": 0.2, "r": 0.3, "i": 0.4, "z": 0.5, "y": 0.6},
        pixel_scale=0.1,
        radius=1.0,
        read_noise=5.0,
        zp_per_sec={"u": 25.0, "g": 26.0, "r": 27.0, "i": 28.0, "z": 29.0, "y": 30.0},
    )

    # We have loaded the non-default values.
    assert ops_data["dark_current"] == 0.1
    assert ops_data["ext_coeff"] == {"u": 0.1, "g": 0.2, "r": 0.3, "i": 0.4, "z": 0.5, "y": 0.6}
    assert ops_data["pixel_scale"] == 0.1
    assert ops_data["radius"] == 1.0
    assert ops_data["read_noise"] == 5.0
    assert ops_data["zp_per_sec"] == {
        "u": 25.0,
        "g": 26.0,
        "r": 27.0,
        "i": 28.0,
        "z": 29.0,
        "y": 30.0,
    }

    # We can check if parameters are set.
    assert "dark_current" in ops_data
    assert "zp_per_sec" in ops_data
    assert "random_param" not in ops_data

    # We can access the filters.
    assert set(ops_data.filters) == {"r", "g", "i"}

    # Check that we can read in the defaults from the pandas metadata.
    pdf = pd.DataFrame(values)
    pdf.attrs["lightcurvelynx_survey_data"] = {
        "survey_name": "test",
        "pixel_scale": 0.1,
        "dark_current": 0.2,
        "radius": 2.0,
    }
    ops_data2 = ObsTable(pdf, dark_current=0.5, radius=1.0)

    # Check that we use use the updated defaults, preferring the keyword arguments
    # to the table's metadata.
    assert ops_data2.survey_values["survey_name"] == "test"
    assert ops_data2.survey_values["pixel_scale"] == 0.1
    assert ops_data2.survey_values["dark_current"] == 0.5
    assert ops_data2.survey_values["radius"] == 1.0

    # We can use the radius to get and set the property.
    assert ops_data2.radius == 1.0
    ops_data2.radius = 5.0
    assert ops_data2.survey_values["radius"] == 5.0

    # If we delete the radius survey value, we get None and can reset it.
    del ops_data2.survey_values["radius"]
    assert ops_data2.radius is None
    ops_data2.radius = 5.0
    assert ops_data2.survey_values["radius"] == 5.0


def test_create_obs_table_custom_names():
    """Create a minimal ObsTable object from alternate column names."""
    values = {
        "custom_time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "custom_ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "custom_dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
        "other": np.zeros(5),
    }

    # Load fails if we use the default colmap.
    with pytest.raises(KeyError):
        _ = ObsTable(values)

    # Load succeeds if we pass in a customer dictionary.
    colmap = {"ra": "custom_ra", "dec": "custom_dec", "time": "custom_time"}
    ops_data = ObsTable(values, colmap=colmap)
    assert len(ops_data) == 5

    # Test that we fail if we try to map a column onto an existing column (mapping the given
    # "other" column onto the existing "zp" column).
    colmap = {"ra": "custom_ra", "dec": "custom_dec", "time": "custom_time", "zp": "other"}
    with pytest.raises(ValueError):
        _ = ObsTable(values, colmap=colmap)


def test_obs_table_add_columns():
    """Create a minimal ObsTable object and perform basic queries."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    ops_data = ObsTable(pdf)
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


def test_create_obs_table_saturation():
    """Test that we can set saturation magnitudes."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    saturation_mags = {"r": 16.0, "g": 17.0, "i": 18.0}

    ops_data = ObsTable(pdf, saturation_mags=saturation_mags)
    assert ops_data.uses_saturation()

    # If we set apply_saturation to False, we do not store the saturation mags.
    ops_data = ObsTable(pdf, saturation_mags=saturation_mags, apply_saturation=False)
    assert ops_data.uses_saturation() is False


def test_obs_table_filter_rows():
    """Test that the user can filter out ObsTable rows."""
    times = np.arange(0.0, 10.0, 1.0)
    values = {
        "time": times,
        "ra": 15.0 * (times + 1.0),
        "dec": -1.0 * times,
        "zp": np.ones(10),
        "filter": np.tile(["r", "g"], 5),
        "custom_column": np.zeros(10),
    }
    ops_data = ObsTable(values, dark_current=0.1, colmap={"custom_column": "my_col"})
    assert len(ops_data) == 10
    assert len(ops_data.columns) == 6
    assert ops_data.survey_values["dark_current"] == 0.1

    # We can filter the ObsTable to specific rows by index.
    inds = [0, 1, 2, 3, 4, 5, 7, 8]
    ops_data = ops_data.filter_rows(inds)
    assert len(ops_data) == 8
    assert len(ops_data.columns) == 6

    # Check that we propagate the survey values and the column mapping.
    assert ops_data.survey_values["dark_current"] == 0.1
    assert "custom_column" in ops_data
    assert "my_col" in ops_data

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


def test_read_small_obs_table(opsim_small):
    """Read in a small ObsTable file from the testing data directory."""
    colmap = {
        "airmass": "airmass",
        "dec": "fieldDec",
        "exptime": "visitExposureTime",
        "filter": "filter",
        "ra": "fieldRA",
        "time": "observationStartMJD",
        "zp": "zp_nJy",  # We add this column to the table
        "seeing": "seeingFwhmEff",
        "skybrightness": "skyBrightness",
        "nexposure": "numExposures",
    }
    ops_data = ObsTable.from_db(opsim_small, colmap=colmap)
    assert len(ops_data) == 300


def test_write_read_obs_table_parquet():
    """Create a minimal observation table data frame, test that we can write it as
    a parquet file, and test that we can correctly read it back in."""

    # Create a fake obstable data frame with just time, RA, dec, and zp. Add two
    # pieces of metadata (survey name and pixel scale).
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
    }
    ops_data = ObsTable(pd.DataFrame(values), survey_name="test", pixel_scale=0.1)

    with tempfile.TemporaryDirectory() as dir_name:
        file_path = Path(dir_name, "test_write_read_obs_table.parquet")

        # The obstable does not exist until we write it.
        assert not file_path.is_file()
        with pytest.raises(FileNotFoundError):
            _ = ObsTable.from_parquet(file_path)

        # We can write the obstable parquet.
        ops_data.write_parquet(file_path)
        assert file_path.is_file()

        # We can reread the obstable parquet.
        ops_data2 = ObsTable.from_parquet(file_path)
        assert len(ops_data2) == 5
        assert np.allclose(values["time"], ops_data2["time"].to_numpy())
        assert np.allclose(values["ra"], ops_data2["ra"].to_numpy())
        assert np.allclose(values["dec"], ops_data2["dec"].to_numpy())
        assert ops_data2.survey_values["survey_name"] == "test"
        assert ops_data2.survey_values["pixel_scale"] == 0.1

        # We cannot overwrite unless we set overwrite=True
        with pytest.raises(FileExistsError):
            ops_data.write_parquet(file_path, overwrite=False)
        ops_data.write_parquet(file_path, overwrite=True)


def test_write_read_obs_table():
    """Create a minimal observation table data frame, test that we can write it,
    and test that we can correctly read it back in."""

    # Create a fake obstable data frame with just time, RA, and dec.
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
    }
    ops_data = ObsTable(pd.DataFrame(values))

    with tempfile.TemporaryDirectory() as dir_name:
        file_path = Path(dir_name, "test_write_read_obs_table.db")

        # The obstable does not exist until we write it.
        assert not file_path.is_file()
        with pytest.raises(FileNotFoundError):
            _ = ObsTable.from_db(file_path)

        # We can write the obstable db.
        ops_data.write_db(file_path)
        assert file_path.is_file()

        # We can reread the obstable db.
        ops_data2 = ObsTable.from_db(file_path)
        assert len(ops_data2) == 5
        assert np.allclose(values["time"], ops_data2["time"].to_numpy())
        assert np.allclose(values["ra"], ops_data2["ra"].to_numpy())
        assert np.allclose(values["dec"], ops_data2["dec"].to_numpy())

        # We cannot overwrite unless we set overwrite=True
        with pytest.raises(ValueError):
            ops_data.write_db(file_path, overwrite=False)
        ops_data.write_db(file_path, overwrite=True)


def test_obs_table_range_search():
    """Test that we can extract the time, ra, and dec from an obs table data frame."""
    # Create a fake obs table data frame with just time, RA, and dec.
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "ra": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "dec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp": np.ones(8),
    }
    ops_data = ObsTable(values)

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

    # With no radius provided (and no default), the query fails.
    with pytest.raises(ValueError):
        _ = ops_data.range_search(15.0, 10.0)

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


def test_obs_table_get_observations():
    """Test that we can extract the time, ra, and dec from an obs table data frame."""
    # Create a fake obs table data frame with just time, RA, and dec.
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "ra": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "dec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp": np.ones(8),
        "airmass_data": np.array([1, 2, 3, 4, 5, 6, 7, 8]),
    }
    ops_data = ObsTable(values, colmap={"airmass_data": "airmass"})

    # Test basic queries (all columns).
    obs = ops_data.get_observations(15.0, 10.0, radius=0.5)
    assert len(obs) == 5
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

    # Test we can use the mapped or original column names.
    obs = ops_data.get_observations(15.0, 10.0, radius=0.5, cols="airmass")
    assert np.allclose(obs["airmass"], [2, 3, 4])
    obs = ops_data.get_observations(15.0, 10.0, radius=0.5, cols="airmass_data")
    assert np.allclose(obs["airmass_data"], [2, 3, 4])

    # Test we fail with an unrecognized column name.
    with pytest.raises(KeyError):
        _ = ops_data.get_observations(15.0, 10.0, radius=0.5, cols=["time", "custom_col"])


def test_obs_table_range_search_detector_footprint():
    """Test that we can extract the time, ra, and dec from an obs table data frame
    with a detector footprint."""
    # Create a fake obs table data frame with just time, RA, and dec.
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "ra": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "dec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp": np.ones(8),
    }
    # Add a circular footprint with radius 0.5 deg.
    detector_footprint = CircleSkyRegion(center=SkyCoord(ra=0.0, dec=0.0, unit="deg"), radius=0.5 * u.deg)
    ops_data = ObsTable(values, detector_footprint=detector_footprint, pixel_scale=360.0)  # 0.1 deg/pix
    assert ops_data.uses_footprint()

    # Check that the ObsTable radius was increased to account for the bounding box of the
    # detector footprint.
    assert ops_data.radius > 0.5

    # We throw a warning if we manually set the radius smaller than the detector footprint bounding box.
    with pytest.warns(UserWarning):
        ops_data.radius = 0.51

    # Test single queries. Despite the given radius, we only find points within the detector footprint.
    assert set(ops_data.range_search(15.0, 10.0, radius=2.0)) == set([1, 2, 3])
    assert set(ops_data.range_search(25.0, 10.0, radius=2.0)) == set([4, 5])
    assert set(ops_data.range_search(15.0, 10.0, radius=100.0)) == set([1, 2, 3])

    # If we use radius smaller than they detector footprint bounding box, we get a warning
    # and only find points within the intersection of the radius and the detector footprint.
    with pytest.warns(Warning):
        assert set(ops_data.range_search(15.0, 10.0, radius=1e-6)) == set([1])
    with pytest.warns(Warning):
        assert set(ops_data.range_search(15.02, 10.0, radius=1e-6)) == set()

    # Test a batched queries and that the detector footprint is applied to all of them.
    query_ra = np.array([15.0, 25.0, 15.0])
    query_dec = np.array([10.0, 10.0, 5.0])
    neighbors = ops_data.range_search(query_ra, query_dec, radius=10.0)
    assert len(neighbors) == 3
    assert set(neighbors[0]) == set([1, 2, 3])
    assert set(neighbors[1]) == set([4, 5])
    assert set(neighbors[2]) == set()

    # We can shut off the detector footprint to save time and use the approximate radius.
    ops_data.clear_detector_footprint()
    assert set(ops_data.range_search(15.0, 10.0, radius=100.0)) == set([0, 1, 2, 3, 4, 5, 6, 7])

    # We can re-add the detector footprint to save time and use the approximate radius.
    ops_data.set_detector_footprint(detector_footprint)
    assert set(ops_data.range_search(15.0, 10.0, radius=2.0)) == set([1, 2, 3])
    assert set(ops_data.range_search(25.0, 10.0, radius=2.0)) == set([4, 5])
    assert set(ops_data.range_search(15.0, 10.0, radius=100.0)) == set([1, 2, 3])

    # If we create a new ObsTable with no radius, it is filled in by the detector footprint.
    ops_data = ObsTable(values, detector_footprint=detector_footprint, pixel_scale=360.0)  # 0.1 deg/pix
    assert ops_data.radius > 0.5
    assert ops_data.radius < 1.5

    # If we manually provide a radius larger than the detector footprint, it is used.
    ops_data = ObsTable(
        values, detector_footprint=detector_footprint, radius=2.0, pixel_scale=360.0
    )  # 0.1 deg/pix
    assert ops_data.radius == 2.0


def test_obs_table_docstring():
    """Test if ObsTable class has a docstring"""
    assert ObsTable.__doc__ is not None
    assert len(ObsTable.__doc__) > 100


def test_read_obs_table_shorten(opsim_shorten):
    """Read in a shorten ObsTable file from the testing data directory."""
    colmap = {
        "airmass": "airmass",
        "dec": "fieldDec",
        "exptime": "visitExposureTime",
        "filter": "filter",
        "ra": "fieldRA",
        "time": "observationStartMJD",
        "zp": "zp_nJy",  # We add this column to the table
        "seeing": "seeingFwhmEff",
        "skybrightness": "skyBrightness",
        "nexposure": "numExposures",
    }
    ops_data = ObsTable.from_db(opsim_shorten, colmap=colmap)
    assert len(ops_data) == 100


def test_build_moc():
    """Test building a Multi-Order Coverage Map (MOC) from the ObsTable."""
    # Create an ObsTable with three pointings.
    values = {
        "time": np.array([0.0, 1.0, 2.0]),
        "ra": np.array([10.0, 11.0, 45.0]),
        "dec": np.array([0.0, 0.0, -10.0]),
        "zp": np.ones(3),
    }
    ops_data = ObsTable(values)

    # We fail if no radius is given (including no default.)
    with pytest.raises(ValueError):
        _ = ops_data.build_moc()

    moc = ops_data.build_moc(radius=1.5)
    assert moc is not None
    assert moc.max_order == 10

    # Test that the MOC covers the correct area.
    assert moc.contains_skycoords(SkyCoord(ra=10.0, dec=0.0, unit="deg"))
    assert moc.contains_skycoords(SkyCoord(ra=11.0, dec=1.0, unit="deg"))
    assert moc.contains_skycoords(SkyCoord(ra=10.5, dec=0.5, unit="deg"))
    assert moc.contains_skycoords(SkyCoord(ra=45.0, dec=-10.0, unit="deg"))
    assert moc.contains_skycoords(SkyCoord(ra=45.01, dec=-9.99, unit="deg"))
    assert moc.contains_skycoords(SkyCoord(ra=44.98, dec=-10.01, unit="deg"))
    assert not moc.contains_skycoords(SkyCoord(ra=46.00, dec=0.0, unit="deg"))
    assert not moc.contains_skycoords(SkyCoord(ra=14.00, dec=0.0, unit="deg"))
    assert not moc.contains_skycoords(SkyCoord(ra=52.00, dec=-10.0, unit="deg"))

    # Although we have 3 pointings of ~7 deg^2 each, there is overlap between two of them.
    # So we check approximate bounds.
    coverage = ops_data.estimate_coverage(radius=1.5)
    assert coverage > 16.0
    assert coverage < 21.0


def test_build_moc_footprint():
    """Test building a Multi-Order Coverage Map (MOC) from the ObsTable with a
    rectangular detector footprint."""
    # Create an ObsTable with three pointings and a 1.0 degree wide (RA) by 0.5 degree high (Dec)
    # detector footprint. Pixel scale is not needed in this test.
    footprint = DetectorFootprint.from_sky_rect(1.0, 0.5, pixel_scale=36.0)
    values = {
        "time": np.array([0.0, 1.0, 2.0]),
        "ra": np.array([10.0, 15.0, 45.0]),
        "dec": np.array([0.0, 0.0, -10.0]),
        "zp": np.ones(3),
    }
    ops_data = ObsTable(values, detector_footprint=footprint)

    moc = ops_data.build_moc(use_footprint=True, max_depth=14)
    assert moc is not None
    assert moc.max_order == 14

    # Test that the MOC covers the correct area.
    for ra, dec in zip(values["ra"], values["dec"], strict=False):
        print(f"Testing pointing at RA={ra}, Dec={dec}")
        # Center is always included.
        assert moc.contains_skycoords(SkyCoord(ra=ra, dec=dec, unit="deg"))

        # Include near corners (We don't do exact corners because of spherical geometry).
        assert moc.contains_skycoords(SkyCoord(ra=ra + 0.4, dec=dec + 0.2, unit="deg"))
        assert moc.contains_skycoords(SkyCoord(ra=ra + 0.4, dec=dec - 0.2, unit="deg"))
        assert moc.contains_skycoords(SkyCoord(ra=ra - 0.4, dec=dec + 0.2, unit="deg"))
        assert moc.contains_skycoords(SkyCoord(ra=ra - 0.4, dec=dec - 0.2, unit="deg"))

        # Exclude just outside the box in each direction.
        assert not moc.contains_skycoords(SkyCoord(ra=ra, dec=dec + 0.3, unit="deg"))
        assert not moc.contains_skycoords(SkyCoord(ra=ra, dec=dec - 0.3, unit="deg"))
        assert not moc.contains_skycoords(SkyCoord(ra=ra + 0.6, dec=dec, unit="deg"))
        assert not moc.contains_skycoords(SkyCoord(ra=ra - 0.6, dec=dec, unit="deg"))

    coverage = ops_data.estimate_coverage(use_footprint=True, max_depth=14)
    assert coverage == pytest.approx(1.5, 0.1)  # 3 pointings x 0.5 deg^2 each
