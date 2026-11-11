import numpy as np
import pandas as pd
import pytest

from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable
from lightcurvelynx.utils.results_builder import SimulationResultBuilder

_LC_COLS = ["mjd", "filter", "flux", "fluxerr", "flux_perfect", "survey_idx", "obs_idx", "is_saturated"]
_SPECTRA_COLS = [
    "mjd",
    "waves",
    "measured_flux",
    "measured_flux_perfect",
    "measured_flux_error",
    "instrument",
]


def _create_basic_builder(num_rows):
    ra = np.arange(num_rows).astype(float)
    dec = np.arange(num_rows).astype(float)
    t0 = 10.0 * np.arange(num_rows).astype(float)
    redshift = 0.1 * np.arange(num_rows).astype(float)
    return SimulationResultBuilder(ra, dec, t0, redshift)


def test_basic_simulation_results_builder():
    """Test that we can build and manipulate simulation results."""
    builder = _create_basic_builder(3)

    # Check that we build a data frame with all the expected columns.
    df = builder.build_results()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    for col in ["id", "ra", "dec", "t0", "z", "nobs", "lightcurve", "spectra"]:
        assert col in df.columns
    assert np.allclose(df["id"], [0.0, 1.0, 2.0])
    assert np.allclose(df["ra"], [0.0, 1.0, 2.0])
    assert np.allclose(df["dec"], [0.0, 1.0, 2.0])
    assert np.allclose(df["t0"], [0.0, 10.0, 20.0])
    assert np.allclose(df["z"], [0.0, 0.1, 0.2])
    assert np.array_equal(df["nobs"], [0, 0, 0])
    assert df["lightcurve"].iloc[0] is None
    assert df["spectra"].iloc[0] is None

    # We fail if we try to add light curve information without starting a new object.
    with pytest.raises(RuntimeError):
        builder.add_light_curve_observations(
            mjd=[0.0],
            filter=["g"],
            flux=[1.0],
            fluxerr=[0.1],
            flux_perfect=[1.0],
            survey_idx=[0],
            obs_idx=[0],
            is_saturated=[False],
        )

    # We can add some basic information.
    builder.start_new_object(0)
    builder.add_light_curve_observations(
        mjd=[0.0, 1.0, 2.0],
        filter=["g", "g", "g"],
        flux=[1.0, 1.0, 1.0],
        fluxerr=[0.1, 0.1, 0.1],
        flux_perfect=[1.0, 1.0, 1.0],
        survey_idx=[0, 0, 0],
        obs_idx=[0, 1, 2],
        is_saturated=[False, False, False],
    )

    # We fail if we try to start a new object before appending the current one.
    with pytest.raises(RuntimeError):
        builder.start_new_object(0)

    builder.append_current_object()

    # Check that we build a data frame with all the expected columns.
    df = builder.build_results()
    lc0 = df["lightcurve"].iloc[0]
    assert len(lc0) == 3
    for col in _LC_COLS:
        assert col in lc0
    assert df["spectra"].iloc[0] is None
    assert df["nobs"].iloc[0] == 3

    # We fail if we try to start an object with an index out of bounds.
    with pytest.raises(ValueError):
        builder.start_new_object(10)

    # We can add a second observation to the same object.
    builder.start_new_object(0)
    builder.add_light_curve_observations(
        mjd=[0.0, 1.0, 2.0],
        filter=["g", "g", "g"],
        flux=[1.0, 1.0, 1.0],
        fluxerr=[0.1, 0.1, 0.1],
        flux_perfect=[1.0, 1.0, 1.0],
        survey_idx=[1, 1, 1],
        obs_idx=[0, 1, 2],
        is_saturated=[False, False, False],
    )
    builder.append_current_object()
    df = builder.build_results()
    assert len(df["lightcurve"].iloc[0]) == 6
    assert df["nobs"].iloc[0] == 6

    # We can append the current object even if no new observations were added.
    builder.start_new_object(0)
    builder.append_current_object()
    df = builder.build_results()
    assert len(df["lightcurve"].iloc[0]) == 6
    assert df["nobs"].iloc[0] == 6

    # We fail if we try to add without all the columns.
    builder.start_new_object(0)
    with pytest.raises(KeyError):
        builder.add_light_curve_observations(mjd=[0.0])

    # We fail if we try to add light curve observations with different array lengths.
    with pytest.raises(ValueError):
        builder.add_light_curve_observations(
            mjd=[0.0, 1.0],
            filter=["g"],
            flux=[1.0, 1.0],
            fluxerr=[0.1, 0.1],
            flux_perfect=[1.0, 1.0],
            survey_idx=[0, 0],
            obs_idx=[0, 1],
            is_saturated=[False, False],
        )


def test_simulation_results_builder_build_results_with_no_objects():
    """Test that build_results works correctly when no objects have been added."""
    builder = _create_basic_builder(5)

    df = builder.build_results()
    assert len(df) == 5
    assert all(df["lightcurve"].isna())
    assert all(df["spectra"].isna())
    assert all(df["nobs"] == 0)


def test_simulation_results_builder_fail():
    """Test failures when constructing a new object"""
    ra = np.array([0.0])
    dec = np.array([0.0])
    t0 = np.array([0.0])
    redshift = np.array([0.0])

    # One of the arrays has too many elements compared to the others.
    with pytest.raises(ValueError):
        _ = SimulationResultBuilder(np.array([0.0, 0.0]), dec, t0, redshift)
    with pytest.raises(ValueError):
        _ = SimulationResultBuilder(ra, np.array([0.0, 0.0]), t0, redshift)
    with pytest.raises(ValueError):
        _ = SimulationResultBuilder(ra, dec, np.array([0.0, 0.0]), redshift)
    with pytest.raises(ValueError):
        _ = SimulationResultBuilder(ra, dec, t0, np.array([0.0, 0.0]))


def test_simulation_results_builder_add_column():
    """Test that we can add a new column to the simulation results builder."""
    builder = _create_basic_builder(1)

    # Fail if we try to add a column with the wrong number of observations.
    with pytest.raises(ValueError):
        builder.add_result_column("wrong_length_column", [1, 2])

    # Fail if we try to add a column that already exists.
    with pytest.raises(KeyError):
        builder.add_result_column("ra", [1])

    # Succeed if we add a column with the correct number of observations.
    builder.add_result_column("correct_length_column", ["str"])
    assert "correct_length_column" in builder.results_dict
    assert builder.results_dict["correct_length_column"] == ["str"]
    assert builder.results_dtypes["correct_length_column"] is str

    # Check that the column was added correctly.
    df = builder.build_results()
    assert "correct_length_column" in df.columns
    assert df["correct_length_column"].iloc[0] == "str"
    assert isinstance(df["correct_length_column"].dtype, pd.StringDtype)

    # Check that we can override the dtype.
    builder.add_result_column("overridden_dtype_column", ["str"], dtype=object)
    assert builder.results_dtypes["overridden_dtype_column"] == object

    df = builder.build_results()
    assert df["overridden_dtype_column"].dtype == object


def test_simulation_results_builder_add_mixed_column():
    """Test that we can add a new column with a mix of types."""
    builder = _create_basic_builder(3)
    builder.add_result_column("uniform_column", [1, 2, 3])
    assert builder.results_dtypes["uniform_column"] == int

    builder.add_result_column("mixed_column", [1, None, 3])
    assert builder.results_dtypes["mixed_column"] == object


def test_simulation_results_builder_add_param_column():
    """Test that we can add a new column to the simulation results builder."""
    builder = _create_basic_builder(1)

    # Check that we can build a column automatically from a GraphState.
    state = GraphState()
    state.set("a", "v1", 1.0)
    builder.add_param_column("a.v1", state)
    assert "a_v1" in builder.results_dict
    assert builder.results_dict["a_v1"] == [1.0]
    assert builder.results_dtypes["a_v1"] == float

    df = builder.build_results()
    assert "a_v1" in df.columns
    assert df["a_v1"].iloc[0] == 1.0

    # We fail if we try to add a parameter that does not exist in the GraphState.
    with pytest.raises(KeyError):
        builder.add_param_column("a.v2", state)


def test_simulation_results_builder_sort_light_curves():
    """Test that we sort the light curves for each object by MJD."""
    builder = _create_basic_builder(5)

    # Add observations from two interleaved surveys.
    builder.start_new_object(0)
    builder.add_light_curve_observations(
        mjd=[0.0, 1.0, 2.0],
        filter=["g", "g", "g"],
        flux=[1.0, 1.0, 1.0],
        fluxerr=[0.1, 0.1, 0.1],
        flux_perfect=[1.0, 1.0, 1.0],
        survey_idx=[0, 0, 0],
        obs_idx=[0, 1, 2],
        is_saturated=[False, False, False],
    )
    builder.add_light_curve_observations(
        mjd=[0.5, 1.5, 2.5, 3.5],
        filter=["r", "r", "r", "r"],
        flux=[2.0, 3.0, 4.0, 5.0],
        fluxerr=[0.1, 0.1, 0.1, 0.1],
        flux_perfect=[2.0, 3.0, 4.0, 5.0],
        survey_idx=[1, 1, 1, 1],
        obs_idx=[0, 1, 2, 3],
        is_saturated=[False, False, False, False],
    )
    builder.append_current_object()

    # Add observations for a second object where the observations are in batches.
    builder.start_new_object(2)
    builder.add_light_curve_observations(
        mjd=[10.0, 11.0],
        filter=["g", "g"],
        flux=[1.0, 1.0],
        fluxerr=[0.1, 0.1],
        flux_perfect=[1.0, 1.0],
        survey_idx=[0, 0],
        obs_idx=[0, 1],
        is_saturated=[False, False],
    )
    builder.add_light_curve_observations(
        mjd=[0.0, 1.0],
        filter=["r", "r"],
        flux=[2.0, 3.0],
        fluxerr=[0.1, 0.1],
        flux_perfect=[2.0, 3.0],
        survey_idx=[1, 1],
        obs_idx=[0, 1],
        is_saturated=[False, False],
    )
    builder.add_light_curve_observations(
        mjd=[3.0, 4.0],
        filter=["i", "i"],
        flux=[4.0, 5.0],
        fluxerr=[0.1, 0.1],
        flux_perfect=[4.0, 5.0],
        survey_idx=[2, 2],
        obs_idx=[0, 1],
        is_saturated=[False, False],
    )
    builder.append_current_object()

    # Check that we build a data frame with 5 objects, but only two sets of observations.
    df = builder.build_results()
    assert len(df) == 5
    assert np.array_equal(df["nobs"], [7, 0, 6, 0, 0])
    assert df["lightcurve"].iloc[1] is None
    assert df["lightcurve"].iloc[3] is None
    assert df["lightcurve"].iloc[4] is None

    lc0 = df["lightcurve"].iloc[0]
    assert np.array_equal(lc0["mjd"], np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]))
    assert np.array_equal(lc0["filter"], np.array(["g", "r", "g", "r", "g", "r", "r"]))
    assert np.array_equal(lc0["flux"], np.array([1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 5.0]))
    assert np.array_equal(lc0["fluxerr"], np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    assert np.array_equal(lc0["flux_perfect"], np.array([1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 5.0]))
    assert np.array_equal(lc0["survey_idx"], np.array([0, 1, 0, 1, 0, 1, 1]))
    assert np.array_equal(lc0["obs_idx"], np.array([0, 0, 1, 1, 2, 2, 3]))
    assert np.array_equal(lc0["is_saturated"], np.array([False, False, False, False, False, False, False]))

    lc2 = df["lightcurve"].iloc[2]
    assert np.array_equal(lc2["mjd"], np.array([0.0, 1.0, 3.0, 4.0, 10.0, 11.0]))
    assert np.array_equal(lc2["filter"], np.array(["r", "r", "i", "i", "g", "g"]))
    assert np.array_equal(lc2["flux"], np.array([2.0, 3.0, 4.0, 5.0, 1.0, 1.0]))
    assert np.array_equal(lc2["fluxerr"], np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    assert np.array_equal(lc2["flux_perfect"], np.array([2.0, 3.0, 4.0, 5.0, 1.0, 1.0]))
    assert np.array_equal(lc2["survey_idx"], np.array([1, 1, 2, 2, 0, 0]))
    assert np.array_equal(lc2["obs_idx"], np.array([0, 1, 0, 1, 0, 1]))
    assert np.array_equal(lc2["is_saturated"], np.array([False, False, False, False, False, False]))


def test_basic_simulation_results_add_spectra():
    """Test that we can build and manipulate simulation results."""
    builder = _create_basic_builder(3)

    # We fail if we try to add spectra information without starting a new object.
    with pytest.raises(RuntimeError):
        builder.add_spectra_observations(
            mjd=0.0,
            waves=[4000.0, 5000.0, 6000.0],
            measured_flux=[1.0, 1.0, 1.0],
            measured_flux_perfect=[1.0, 1.0, 1.0],
            measured_flux_error=[0.1, 0.1, 0.1],
            instrument="inst1",
        )

    # We can add some basic information with two spectra observations
    # for object 0, 0 for object 1, and 1 for object 2.
    builder.start_new_object(0)
    builder.add_spectra_observations(
        mjd=0.0,
        waves=[4000.0, 5000.0, 6000.0],
        measured_flux=[1.0, 1.0, 1.0],
        measured_flux_perfect=[1.0, 1.0, 1.0],
        measured_flux_error=[0.1, 0.1, 0.1],
        instrument="inst1",
    )
    builder.add_spectra_observations(
        mjd=10.0,
        waves=[5000.0, 6000.0, 7000.0],
        measured_flux=[2.1, 2.2, 2.3],
        measured_flux_perfect=[2.0, 2.0, 2.0],
        measured_flux_error=[0.1, 0.2, 0.3],
        instrument="inst2",
    )
    builder.append_current_object()

    builder.start_new_object(2)
    builder.add_spectra_observations(
        mjd=20.0,
        waves=[4000.0, 5000.0, 6000.0],
        measured_flux=[1.0, 1.0, 1.0],
        measured_flux_perfect=[1.0, 1.0, 1.0],
        measured_flux_error=[0.1, 0.1, 0.1],
        instrument="inst1",
    )
    builder.append_current_object()

    # Check that we build a data frame with all the expected columns.
    df = builder.build_results()
    assert len(df) == 3
    assert np.array_equal(df["nobs"], [2, 0, 1])

    spectra0 = df["spectra"].iloc[0]
    print(spectra0)
    assert spectra0 is not None
    for col in _SPECTRA_COLS:
        assert col in spectra0
        assert len(spectra0[col]) == 2

    assert np.array_equal(spectra0["mjd"], [0.0, 10.0])
    assert np.array_equal(spectra0["waves"][0], [4000.0, 5000.0, 6000.0])
    assert np.array_equal(spectra0["waves"][1], [5000.0, 6000.0, 7000.0])
    assert np.array_equal(spectra0["measured_flux"][0], [1.0, 1.0, 1.0])
    assert np.array_equal(spectra0["measured_flux"][1], [2.1, 2.2, 2.3])
    assert np.array_equal(spectra0["measured_flux_perfect"][0], [1.0, 1.0, 1.0])
    assert np.array_equal(spectra0["measured_flux_perfect"][1], [2.0, 2.0, 2.0])
    assert np.array_equal(spectra0["measured_flux_error"][0], [0.1, 0.1, 0.1])
    assert np.array_equal(spectra0["measured_flux_error"][1], [0.1, 0.2, 0.3])
    assert np.array_equal(spectra0["instrument"], ["inst1", "inst2"])

    spectra1 = df["spectra"].iloc[1]
    assert spectra1 is None

    spectra2 = df["spectra"].iloc[2]
    assert spectra2 is not None
    for col in _SPECTRA_COLS:
        assert col in spectra2
        assert len(spectra2[col]) == 1

    assert np.array_equal(spectra2["mjd"], [20.0])
    assert np.array_equal(spectra2["waves"][0], [4000.0, 5000.0, 6000.0])
    assert np.array_equal(spectra2["measured_flux"][0], [1.0, 1.0, 1.0])
    assert np.array_equal(spectra2["measured_flux_perfect"][0], [1.0, 1.0, 1.0])
    assert np.array_equal(spectra2["measured_flux_error"][0], [0.1, 0.1, 0.1])
    assert np.array_equal(spectra2["instrument"], ["inst1"])


def test_simulation_results_builder_add_light_curve_column():
    """Test that we can add columns for the light curves."""
    builder = _create_basic_builder(10)

    # We add a column with a given dtype.
    builder.add_light_curve_column("num_trees_in_fov", dtype=int)
    assert "num_trees_in_fov" in builder.lightcurve_dict
    assert builder.lightcurve_dtypes["num_trees_in_fov"] == int

    # If given and ObsTable, we can infer the dtype from the observation tables.
    obstable1 = FakeObsTable(
        {
            "time": [3.0, 4.0, 5.0],
            "ra": [0.0, 0.0, 0.0],
            "dec": [10.0, 10.0, 10.0],
            "filter": ["r", "r", "r"],
            "zp": [0.5, 0.5, 0.5],
            "custom_col1": [1, 2, 3],
            "custom_col2": ["a", "b", "c"],
            "custom_col3": [1.1, 1.2, 1.3],
        }
    )
    builder.add_light_curve_column("custom_col1", obs_table_list=[obstable1])
    assert "custom_col1" in builder.lightcurve_dict
    assert builder.lightcurve_dtypes["custom_col1"] == int

    # We can't add a column twice.
    with pytest.raises(KeyError):
        builder.add_light_curve_column("custom_col1", obs_table_list=[obstable1])

    # If we have different dtypes for the same column across different observation tables,
    # we should default to object.
    obstable2 = FakeObsTable(
        {
            "time": [3.0],
            "ra": [0.0],
            "dec": [10.0],
            "filter": ["r"],
            "zp": [0.5],
            "custom_col2": [1],
        }
    )
    builder.add_light_curve_column("custom_col2", obs_table_list=[obstable1, obstable2])
    assert "custom_col2" in builder.lightcurve_dict
    assert builder.lightcurve_dtypes["custom_col2"] == object

    # If we add a column that is only present in one observation table, it should default
    # to 'object' to correctly handle None.
    builder.add_light_curve_column("custom_col3", obs_table_list=[obstable1, obstable2])
    assert "custom_col3" in builder.lightcurve_dict
    assert builder.lightcurve_dtypes["custom_col3"] == object

    # If we add a column that is not present in any observation table, it should produce a warning
    # and not add the column.
    with pytest.warns(UserWarning):
        builder.add_light_curve_column("custom_col4", obs_table_list=[obstable1, obstable2])
    assert "custom_col4" not in builder.lightcurve_dict


def test_simulation_results_builder_add_light_curve_column_in_progress():
    """Test that we cannot add columns for the light curves after we have started
    constructing the data set."""
    builder = _create_basic_builder(4)

    builder.start_new_object(0)
    builder.add_light_curve_observations(
        mjd=[10.0, 11.0],
        filter=["g", "g"],
        flux=[1.0, 1.0],
        fluxerr=[0.1, 0.1],
        flux_perfect=[1.0, 1.0],
        survey_idx=[0, 0],
        obs_idx=[0, 1],
        is_saturated=[False, False],
    )
    builder.append_current_object()

    with pytest.raises(RuntimeError):
        builder.add_light_curve_column("new_column", dtype=int)
