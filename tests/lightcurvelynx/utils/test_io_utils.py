import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.utils.io_utils import (
    read_grid_data,
    read_lclib_data,
    read_numpy_data,
    write_numpy_data,
    write_results_as_hats,
)
from lsdb import read_hats
from nested_pandas import NestedFrame


def test_read_write_numpy_data(tmp_path):
    """Test reading and writing numpy data."""
    data = np.arange(100, dtype=float).reshape((25, 4))

    for fmt in [".npy", ".npz", ".csv", ".ecsv", ".txt"]:
        file_path = tmp_path / f"test{fmt}"
        assert not file_path.exists()

        write_numpy_data(file_path, data)
        assert file_path.exists()

        loaded_data = read_numpy_data(file_path)
        np.testing.assert_allclose(data, loaded_data)

    # Test a file that does not exist.
    with pytest.raises(FileNotFoundError):
        _ = read_numpy_data(tmp_path / "no_such_file_here.npy")

    # Test an unsupported file format.
    with pytest.raises(ValueError):
        write_numpy_data(tmp_path / "test.invalid", data)


def test_read_write_lsdb(tmp_path):
    """Test reading and writing HATS data via LSDB."""
    outer_dict = {
        "id": [0, 1, 2],
        "ra": [10.0, 10.0001, 10.0002],
        "dec": [-10.0, -9.999, -10.0001],
        "nobs": [3, 2, 1],
        "z": [0.1, 0.2, 0.3],
    }
    inner_dict = {
        "mjd": [59000, 59001, 59002, 59000, 59001, 59000],
        "flux": [10.0, 12.0, 11.0, 15.0, 14.0, 13.0],
        "fluxerr": [1.0, 1.0, 1.0, 1.5, 1.5, 1.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
    }
    nested_inds = [0, 0, 0, 1, 1, 2]
    results = NestedFrame(data=outer_dict, index=[0, 1, 2])
    nested_1 = pd.DataFrame(data=inner_dict, index=nested_inds)
    results = results.join_nested(nested_1, "lightcurve")

    # Write out the results to a temporary directory.
    out_dir = tmp_path / "lsdb_output"
    write_results_as_hats(out_dir, results, overwrite=True)
    assert out_dir.exists()

    # Check that we can read the data back in.
    loaded_results = read_hats(out_dir).compute()
    assert len(loaded_results) == len(results)
    for i in range(len(results)):
        assert results["ra"].iloc[i] == loaded_results["ra"].iloc[i]
        assert results["dec"].iloc[i] == loaded_results["dec"].iloc[i]
        pd.testing.assert_frame_equal(results["lightcurve"].iloc[i], loaded_results["lightcurve"].iloc[i])


def test_read_grid_data_good(grid_data_good_file):
    """Test that we can read a well formatted grid data file."""
    x0, x1, values = read_grid_data(grid_data_good_file, format="ascii.csv")
    x0_expected = np.array([0.0, 1.0, 2.0])
    x1_expected = np.array([1.0, 1.5])
    values_expected = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    np.testing.assert_allclose(x0, x0_expected, atol=1e-5)
    np.testing.assert_allclose(x1, x1_expected, atol=1e-5)
    np.testing.assert_allclose(values, values_expected, atol=1e-5)


def test_read_grid_data_bad(grid_data_bad_file):
    """Test that we correctly handle a badly formatted grid data file."""
    # We load without a problem is validation is off.
    x0, x1, values = read_grid_data(grid_data_bad_file, format="ascii")
    assert values.shape == (3, 2)

    with pytest.raises(ValueError):
        _, _, _ = read_grid_data(grid_data_bad_file, format="ascii", validate=True)

    # We fail when loading a nonexistent file.
    with pytest.raises(FileNotFoundError):
        _, _, _ = read_grid_data("no_such_file_here", format="ascii", validate=True)


def test_read_lclib_data(test_data_dir):
    """Test reading a SNANA LCLIB data from a text file."""
    lc_file = test_data_dir / "test_lclib_data.TEXT"
    curves = read_lclib_data(lc_file)
    assert len(curves) == 3

    expected_cols = ["type", "time", "u", "g", "r", "i", "z"]
    expected_len = [20, 20, 15]
    expected_param = [
        {"TYPE": "1", "OTHER": "1"},
        {"TYPE": "1", "OTHER": "2"},
        {"TYPE": "6", "OTHER": "7"},
    ]
    for idx, curve in enumerate(curves):
        assert len(curve) == expected_len[idx]
        assert int(curve.meta["id"]) == idx
        assert curve.meta["RECUR_CLASS"] == "NON-RECUR"
        for key, value in expected_param[idx].items():
            assert curve.meta["PARVAL"][key] == value

        # We did not pick up anything in the documentation block.
        assert "PURPOSE" not in curve.meta

        # Check that the expected columns are present and type is "S" or "T"
        for col in expected_cols:
            assert col in curve.colnames
        assert np.all((curve["type"].data == "S") | (curve["type"].data == "T"))
