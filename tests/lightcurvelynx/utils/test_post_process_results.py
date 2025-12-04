import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.astro_utils.mag_flux import flux2mag
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable
from lightcurvelynx.utils.post_process_results import (
    augment_single_lightcurve,
    concat_results,
    lightcurve_compute_mag,
    lightcurve_compute_snr,
    results_append_obstable_data,
    results_append_param_as_col,
    results_augment_lightcurves,
    results_drop_empty,
)
from nested_pandas import NestedFrame


def test_concat_results():
    """Test the concat_results function."""
    # Create two NestedFrames to concatenate.
    outer_dict_1 = {
        "id": [0, 1, 2],
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, 0.0, 10.0],
        "nobs": [3, 2, 1],
        "z": [0.1, 0.2, 0.3],
    }
    inner_dict_1 = {
        "mjd": [59000, 59001, 59002, 59000, 59001, 59000],
        "flux": [10.0, 12.0, 11.0, 15.0, 14.0, 13.0],
        "fluxerr": [1.0, 1.0, 1.0, 1.5, 1.5, 1.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
    }
    nested_inds_1 = [0, 0, 0, 1, 1, 2]
    res1 = NestedFrame(data=outer_dict_1, index=[0, 1, 2])
    nested_1 = pd.DataFrame(data=inner_dict_1, index=nested_inds_1)
    res1 = res1.join_nested(nested_1, "lightcurve")

    outer_dict_2 = {
        "id": [3, 4],
        "ra": [40.0, 50.0],
        "dec": [-40.0, -50.0],
        "nobs": [1, 2],
        "z": [0.4, 0.5],
    }
    inner_dict_2 = {
        "mjd": [59000, 59001, 59001],
        "flux": [20.0, 22.0, 21.0],
        "fluxerr": [2.0, 2.0, 2.0],
        "filter": ["g", "r", "i"],
    }
    nested_inds_2 = [0, 1, 1]
    res2 = NestedFrame(data=outer_dict_2, index=[0, 1])
    nested_2 = pd.DataFrame(data=inner_dict_2, index=nested_inds_2)
    res2 = res2.join_nested(nested_2, "lightcurve")

    # Concatenate the results.
    results = concat_results([res1, res2])
    assert len(results) == 5
    assert results.columns.tolist() == ["id", "ra", "dec", "nobs", "z", "lightcurve"]

    # Check the outer columns.
    assert results["id"].tolist() == [0, 1, 2, 3, 4]
    assert np.array_equal(results["ra"], [10.0, 20.0, 30.0, 40.0, 50.0])
    assert np.array_equal(results["dec"], [-10.0, 0.0, 10.0, -40.0, -50.0])
    assert np.array_equal(results["z"], [0.1, 0.2, 0.3, 0.4, 0.5])

    # Check one of the inner columns.
    assert results["lightcurve"].nest.columns == ["mjd", "flux", "fluxerr", "filter"]
    assert np.allclose(results["lightcurve"][0]["flux"], [10.0, 12.0, 11.0])
    assert np.allclose(results["lightcurve"][1]["flux"], [15.0, 14.0])
    assert np.allclose(results["lightcurve"][2]["flux"], [13.0])
    assert np.allclose(results["lightcurve"][3]["flux"], [20.0])
    assert np.allclose(results["lightcurve"][4]["flux"], [22.0, 21.0])


def test_results_append_param_as_col():
    """Test the results_append_param_as_col function."""
    # Create two NestedFrames to concatenate.
    outer_dict = {
        "id": [0, 1, 2],
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, 0.0, 10.0],
        "nobs": [3, 2, 1],
        "z": [0.1, 0.2, 0.3],
        "params": [
            {"salt2.c": 0.1, "salt2.x1": 0.5},
            {"salt2.c": 0.2, "salt2.x1": 0.6},
            {"salt2.c": 0.3, "salt2.x1": 0.7},
        ],
    }
    inner_dict = {
        "mjd": [59000, 59001, 59002, 59000, 59001, 59000],
        "flux": [10.0, 12.0, 11.0, 15.0, 14.0, 13.0],
        "fluxerr": [1.0, 1.0, 1.0, 1.5, 1.5, 1.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
    }
    nested_inds = [0, 0, 0, 1, 1, 2]
    res1 = NestedFrame(data=outer_dict, index=[0, 1, 2])
    nested = pd.DataFrame(data=inner_dict, index=nested_inds)
    res1 = res1.join_nested(nested, "lightcurve")

    assert "salt2_c" not in res1.columns
    assert "salt2_x1" not in res1.columns

    res1 = results_append_param_as_col(res1, "salt2.c")
    assert "salt2_c" in res1.columns
    assert np.array_equal(res1["salt2_c"], [0.1, 0.2, 0.3])
    assert "salt2_x1" not in res1.columns

    # We warn if we are overwriting an existing column.
    with pytest.warns(UserWarning):
        res1 = results_append_param_as_col(res1, "salt2.c")


def test_results_append_obstable_data():
    """Test the results_append_param_as_col function."""
    # Create two NestedFrames to concatenate.
    outer_dict = {
        "id": [0, 1, 2],
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, 0.0, 10.0],
        "nobs": [3, 2, 1],
        "z": [0.1, 0.2, 0.3],
        "params": [
            {"salt2.c": 0.1, "salt2.x1": 0.5},
            {"salt2.c": 0.2, "salt2.x1": 0.6},
            {"salt2.c": 0.3, "salt2.x1": 0.7},
        ],
    }
    inner_dict = {
        "mjd": [59000, 59001, 59002, 59000, 59001.5, 59003],
        "flux": [10.0, 12.0, 11.0, 15.0, 14.0, 13.0],
        "fluxerr": [1.0, 1.0, 1.0, 1.5, 1.5, 1.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
        "survey_idx": [0, 0, 0, 1, 0, 1],
        "obs_idx": [0, 1, 3, 0, 2, 3],
    }
    nested_inds = [0, 0, 0, 1, 1, 2]
    res1 = NestedFrame(data=outer_dict, index=[0, 1, 2])
    nested = pd.DataFrame(data=inner_dict, index=nested_inds)
    res1 = res1.join_nested(nested, "lightcurve")
    assert "test_col" not in res1.columns
    assert "test_col" not in res1["lightcurve"].nest.columns

    # Create two FakeObsTable instances with a test column to add.
    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data_1 = {
        "time": np.array([59000.0, 59001.0, 59001.5, 59002.0, 59003.0]),
        "ra": np.array([10.0, 10.0, 20.0, 10.0, 10.0]),
        "dec": np.array([-10.0, -10.0, 0.0, -10.0, -10.0]),
        "filter": np.array(["g", "r", "g", "i", "g"]),
        "test_col": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    ops_table_1 = FakeObsTable(
        pd.DataFrame(ops_data_1),
        noise_strategy="exhaustive",
        zp_per_band=zp_per_band,
        fwhm_px=2.0,
        sky_bg_electrons=100.0,
    )

    ops_data_2 = {
        "time": np.array([59000.0, 59001.0, 59002.0, 59003.0]),
        "ra": np.array([20.0, 20.0, 30.0, 30.0]),
        "dec": np.array([0.0, 0.0, 0.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i"]),
        "test_col": np.array([11.0, 12.0, 13.0, 14.0]),
    }
    ops_table_2 = FakeObsTable(
        pd.DataFrame(ops_data_2),
        noise_strategy="exhaustive",
        zp_per_band=zp_per_band,
        fwhm_px=2.0,
        sky_bg_electrons=100.0,
    )

    res1 = results_append_obstable_data(res1, "test_col", [ops_table_1, ops_table_2])
    assert "test_col" in res1["lightcurve"].nest.columns
    assert np.array_equal(res1["lightcurve.test_col"][0], [1.0, 2.0, 4.0])
    assert np.array_equal(res1["lightcurve.test_col"][1], [11.0, 3.0])
    assert np.array_equal([res1["lightcurve.test_col"][2]], [14.0])


def _allclose(a, b, rtol=1e-05, atol=1e-08):
    """Helper function to compare two arrays, treating NaNs and Nones as equal.

    Parameters
    ----------
    a : array-like
        First array to compare.
    b : array-like
        Second array to compare.
    rtol : float, optional
        Relative tolerance. Default is 1e-5.
    atol : float, optional
        Absolute tolerance. Default is 1e-8.
    """
    a = np.ma.asarray(a)
    b = np.ma.asarray(b)

    # Check whether the same entries are None.
    a_valid = (a is not None) & (a is not np.ma.masked)
    b_valid = (b is not None) & (b is not np.ma.masked)
    if not np.array_equal(a_valid, b_valid):
        return False

    a_float = a[a_valid].astype(float)
    b_float = b[b_valid].astype(float)
    return np.allclose(a_float, b_float, rtol=rtol, atol=atol, equal_nan=True)


def test_results_drop_empty():
    """Test the results_drop_empty function."""
    # Create a NestedFrame with some empty lightcurves.
    source_data = {
        "object_id": [0, 1, 2, 3, 4],
        "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
        "dec": [-10.0, -20.0, -30.0, -40.0, -50.0],
        "nobs": [3, 0, 1, 2, 0],
        "z": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    results = NestedFrame(data=source_data, index=[0, 1, 2, 3, 4])

    # Create a nested DataFrame with lightcurves, some of which are empty.
    nested_data = {
        "mjd": [59000, 59001, 59002, 59003, 59004, 59005],
        "flux": [10.0, 12.0, 11.0, 20.0, 22.0, 30.0],
        "fluxerr": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
    }
    nested_frame = pd.DataFrame(data=nested_data, index=[0, 0, 0, 2, 3, 3])

    # Add the nested DataFrame to the results.
    results = results.join_nested(nested_frame, "lightcurve")
    assert len(results) == 5
    assert results["object_id"].tolist() == [0, 1, 2, 3, 4]

    # Apply the drop_empty function.
    filtered_results = results_drop_empty(results)
    assert len(filtered_results) == 3
    assert filtered_results["object_id"].tolist() == [0, 2, 3]

    assert len(filtered_results["lightcurve"][0]) == 3
    assert len(filtered_results["lightcurve"][2]) == 1
    assert len(filtered_results["lightcurve"][3]) == 2


def test_lightcurve_compute_snr():
    """Test the lightcurve_compute_snr function."""
    flux = [10.0, 12.0, 0.1, 0.2, 5.0, 6.0, -1.0, 1.0, 10.0]
    fluxerr = [1.0, 1.0, 1.0, 20.0, 2.0, 1.0, 1.0, -1.0, 0.0]
    snr = lightcurve_compute_snr(flux, fluxerr)
    assert _allclose(snr, [10.0, 12.0, 0.1, 0.01, 2.5, 6.0, None, None, None])


def test_lightcurve_compute_mag():
    """Test the lightcurve_compute_mag function."""
    flux = [10.0, 12.0, 0.1, 0.2, 5.0, 6.0, -1.0, 0.0]
    fluxerr = [1.0, 1.0, 1.0, 20.0, 2.0, 1.0, 1.0, 10.0]
    mag, magerr = lightcurve_compute_mag(flux, fluxerr)
    assert _allclose(mag, [flux2mag(f) for f in flux[:-2]] + [None, None])
    assert _allclose(
        magerr,
        [(2.5 / np.log(10)) * (f_err / f_val) for f_val, f_err in zip(flux[:-2], fluxerr[:-2], strict=False)]
        + [None, None],
    )


def test_results_augment_lightcurves():
    """Test the results_augment_lightcurves function."""
    # Create a NestedFrame with some empty lightcurves.
    source_data = {
        "object_id": [0, 1, 2],
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, -20.0, -30.0],
        "nobs": [3, 0, 1],
        "z": [0.1, 0.2, 0.3],
    }
    results = NestedFrame(data=source_data, index=[0, 1, 2])

    with pytest.raises(ValueError):
        # Ensure that the function raises an error if 'lightcurve' is not present.
        results_augment_lightcurves(results, min_snr=5)

    # Create a nested DataFrame with lightcurves, some of which are empty.
    nested_data = {
        "mjd": [59000, 59001, 59002, 59003, 59004, 59005],
        "flux": [10.0, 12.0, 0.1, 0.2, 5.0, 6.0],
        "fluxerr": [1.0, 1.0, 1.0, 20.0, 2.0, 1.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
    }
    nested_frame = pd.DataFrame(data=nested_data, index=[0, 0, 1, 1, 2, 2])

    # Add the nested DataFrame to the results.
    results = results.join_nested(nested_frame, "lightcurve")
    assert len(results) == 3
    assert "snr" not in results["lightcurve"].nest.columns
    assert "detection" not in results["lightcurve"].nest.columns
    assert "mag" not in results["lightcurve"].nest.columns
    assert "magerr" not in results["lightcurve"].nest.columns
    assert results["object_id"].tolist() == [0, 1, 2]

    # Augmenting the lightcurves should add the new columns.
    results_augment_lightcurves(results, min_snr=5)
    assert len(results) == 3

    # Check the SNR and detection markings.
    assert "snr" in results["lightcurve"].nest.columns
    assert _allclose(results["lightcurve.snr"][0].values, [10.0, 12.0])
    assert _allclose(results["lightcurve.snr"][1].values, [0.1, 0.01])
    assert _allclose(results["lightcurve.snr"][2].values, [2.5, 6.0])

    assert "detection" in results["lightcurve"].nest.columns
    assert results["lightcurve.detection"][0].tolist() == [True, True]
    assert results["lightcurve.detection"][1].tolist() == [False, False]
    assert results["lightcurve.detection"][2].tolist() == [False, True]

    # Check the AB magnitudes and magnitude errors.
    assert "mag" in results["lightcurve"].nest.columns
    assert _allclose(results["lightcurve.mag"][0].values, [flux2mag(10.0), flux2mag(12.0)])
    assert _allclose(results["lightcurve.mag"][1].values, [flux2mag(0.1), flux2mag(0.2)])
    assert _allclose(results["lightcurve.mag"][2].values, [flux2mag(5.0), flux2mag(6.0)])

    assert "magerr" in results["lightcurve"].nest.columns
    for i in range(3):
        assert _allclose(
            results["lightcurve.magerr"][i].values,
            (2.5 / np.log(10)) * 1.0 / results["lightcurve.snr"][i].values,
        )

    # Without providing a t0, we do not compute relative time.
    assert "time_rel" not in results["lightcurve"].nest.columns

    # If we provide an array with None, we do not compute relative time.
    results["t0"] = np.array([None, None, None])
    results_augment_lightcurves(results, min_snr=5)
    assert "time_rel" not in results["lightcurve"].nest.columns

    # Try with a valid array of t0.
    results["t0"] = np.array([59000, 59001, 59002])
    results_augment_lightcurves(results, min_snr=5)
    assert "time_rel" in results["lightcurve"].nest.columns
    assert _allclose(results["lightcurve.time_rel"][0].values, [0, 1])
    assert _allclose(results["lightcurve.time_rel"][1].values, [1, 2])
    assert _allclose(results["lightcurve.time_rel"][2].values, [2, 3])


def test_results_augment_lightcurves_empty():
    """Test the results_augment_lightcurves function with a completely empty frame."""
    # Create an empty NestedFrame.
    source_data = {
        "object_id": [],
        "ra": [],
        "dec": [],
        "nobs": [],
        "z": [],
    }
    results = NestedFrame(data=source_data, index=[])

    nested_data = {
        "mjd": [],
        "flux": [],
        "fluxerr": [],
        "filter": [],
    }
    nested_frame = pd.DataFrame(data=nested_data, index=[])
    results = results.join_nested(nested_frame, "lightcurve")
    assert len(results) == 0

    # Augmenting the lightcurves should add the new columns.
    results_augment_lightcurves(results, min_snr=5)
    assert len(results) == 0
    assert "snr" in results["lightcurve"].nest.columns
    assert "detection" in results["lightcurve"].nest.columns
    assert "mag" in results["lightcurve"].nest.columns
    assert "magerr" in results["lightcurve"].nest.columns


def test_results_augment_lightcurves_single():
    """Test the results_augment_lightcurves function with a non-nested frame."""
    # Create a DataFrame with a lightcurve.
    source_data = {
        "mjd": [59000, 59001, 59002, 59003, 59004, 59005, 59006],
        "flux": [10.0, 12.0, 0.1, 0.2, 5.0, 6.0, -1.0],
    }
    results = pd.DataFrame(data=source_data)

    with pytest.raises(ValueError):
        # Ensure that the function raises an error if 'flux' or 'fluxerr' is not present.
        results_augment_lightcurves(results, min_snr=5)

    # Create a nested DataFrame with lightcurves, some of which are empty.
    results["fluxerr"] = [1.0, 1.0, 1.0, 20.0, 2.0, 1.0, 1.0]

    # Add the nested DataFrame to the results.
    assert "snr" not in results.columns
    assert "detection" not in results.columns
    assert "mag" not in results.columns
    assert "magerr" not in results.columns
    assert "time_rel" not in results.columns

    # Augmenting the lightcurves should add the new columns.
    augment_single_lightcurve(results, min_snr=5)

    # Check we have added the columns.
    assert "snr" in results.columns
    assert "detection" in results.columns
    assert "mag" in results.columns
    assert "magerr" in results.columns
    assert "time_rel" not in results.columns
    assert _allclose(results["snr"].values, [10.0, 12.0, 0.1, 0.01, 2.5, 6.0, None])
    assert np.array_equal(results["detection"].values, [True, True, False, False, False, True, False])
    assert _allclose(
        results["mag"].values,
        [flux2mag(10.0), flux2mag(12.0), flux2mag(0.1), flux2mag(0.2), flux2mag(5.0), flux2mag(6.0), None],
    )
    assert _allclose(results["magerr"].values[:6], (2.5 / np.log(10)) * (1.0 / results["snr"].values[:6]))

    # Try with a t0.
    augment_single_lightcurve(results, min_snr=5, t0=59000)
    assert "time_rel" in results.columns
    assert _allclose(results["time_rel"].values, [0, 1, 2, 3, 4, 5, 6])


def test_results_augment_lightcurves_single_empty():
    """Test the results_augment_lightcurves function with an empty non-nested frame."""
    # Create a DataFrame with a lightcurve.
    source_data = {
        "mjd": [],
        "flux": [],
        "fluxerr": [],
    }
    results = pd.DataFrame(data=source_data)

    # Augmenting the lightcurves should add the new columns.
    augment_single_lightcurve(results, min_snr=5)
    assert "snr" in results.columns
    assert "detection" in results.columns
    assert "mag" in results.columns
    assert "magerr" in results.columns


def test_results_augment_lightcurves_single_invalid():
    """Test the results_augment_lightcurves function with only invalid entries."""
    # Create a DataFrame with a lightcurve.
    source_data = {
        "mjd": [1.0, 2.0, 3.0],
        "flux": [-1.0, -1.0, 1.0],
        "fluxerr": [-1.0, 1.0, -1.0],
    }
    results = pd.DataFrame(data=source_data)

    # Augmenting the lightcurves should add the new columns.
    augment_single_lightcurve(results, min_snr=5)
    assert "snr" in results.columns
    assert "detection" in results.columns
    assert "mag" in results.columns
    assert "magerr" in results.columns


def test_augment_lightcurves_with_existing_detection_column():
    """Check that we have fixed the error from Issue 638. This failed before the fix."""
    # Create a results data frame with some invalid flux/fluxerr entries.
    data = {
        "id": [0, 1],
        "t0": [59000.0, 59001.0],
    }
    results = NestedFrame(data=data, index=[0, 1])

    nested_data = {
        "mjd": [59000.0, 59001.0, 59002.0, 59003.0, 59004.0, 59005.0],
        "flux": [100.0, 200.0, -50.0, 150.0, 0.0, 180.0],
        "fluxerr": [10.0, 15.0, 5.0, 0.0, 10.0, 12.0],
    }
    nested_index = [0, 0, 0, 1, 1, 1]
    nested_df = pd.DataFrame(data=nested_data, index=nested_index)
    results = results.join_nested(nested_df, "lightcurve")

    # Pre-populate the detection column with None/null values to set up PyArrow null type
    # This simulates a scenario where the column exists but has no data
    results["lightcurve.detection"] = None

    # Run the augmentation function, which previously failed
    results_augment_lightcurves(results, min_snr=5.0)
