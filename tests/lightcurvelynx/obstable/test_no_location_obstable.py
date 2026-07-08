import numpy as np
import pytest
from lightcurvelynx.obstable.no_location_obstable import NoLocationObsTable


def test_create_no_location_obstable():
    """Create a minimal NoLocationObsTable object and perform basic queries."""
    num_points = 50
    values = {
        "time": 59000.0 + np.arange(num_points),
        "filter": np.random.choice(["g", "r", "i"], size=num_points),
    }
    obs_table = NoLocationObsTable(values)
    assert len(obs_table) == num_points

    # The query should return all indices since the NoLocationObsTable covers the entire sky.
    query_ra = np.random.uniform(0.0, 360.0, size=10)
    query_dec = np.random.uniform(-90.0, 90.0, size=10)
    inds = obs_table.range_search(query_ra, query_dec)
    assert len(inds) == 10
    for ind in inds:
        assert np.array_equal(ind, np.arange(num_points))

    # We can filter on time though.
    t_min = 59010.0
    t_max = 59020.0
    inds = obs_table.range_search(query_ra, query_dec, t_min=t_min, t_max=t_max)
    for ind in inds:
        assert np.all(obs_table._table["time"].to_numpy()[ind] >= t_min)
        assert np.all(obs_table._table["time"].to_numpy()[ind] <= t_max)

    # We can filter on times that are arrays.
    t_min = 59010.0 + np.arange(10)
    t_max = 59020.0 + np.arange(10)
    inds = obs_table.range_search(query_ra, query_dec, t_min=t_min, t_max=t_max)
    for i, ind in enumerate(inds):
        assert np.all(obs_table._table["time"].to_numpy()[ind] >= t_min[i])
        assert np.all(obs_table._table["time"].to_numpy()[ind] <= t_max[i])

    # We fail if we are missing required columns.
    with pytest.raises(KeyError):
        _ = NoLocationObsTable({"time": 59000.0 + np.arange(num_points)})
    with pytest.raises(KeyError):
        _ = NoLocationObsTable({"filter": np.random.choice(["g", "r", "i"], size=num_points)})


def test_no_location_obstable_build_moc():
    """Test that the MOC built from an NoLocationObsTable covers the entire sky."""
    num_points = 50
    values = {
        "time": 59000.0 + np.arange(num_points),
        "filter": np.random.choice(["g", "r", "i"], size=num_points),
    }
    obs_table = NoLocationObsTable(values)
    moc = obs_table.build_moc(max_depth=5)
    assert moc is not None
    assert moc.sky_fraction == pytest.approx(1.0, rel=1e-6)
