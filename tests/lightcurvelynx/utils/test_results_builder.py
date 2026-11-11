import numpy as np
import pandas as pd
import pytest

from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.utils.results_builder import SimulationResultBuilder

_LC_COLS = ["mjd", "filter", "flux", "fluxerr", "flux_perfect", "survey_idx", "obs_idx", "is_saturated"]
_SPECTRA_COLS = ["mjd", "wavelength", "flux", "fluxerr", "survey_idx", "obs_idx", "is_saturated"]


def test_basic_simulation_results_builder():
    """Test that we can build and manipulate simulation results."""
    ra = np.array([0.0])
    dec = np.array([0.0])
    t0 = np.array([0.0])
    redshift = np.array([0.0])
    builder = SimulationResultBuilder(ra, dec, t0, redshift)

    # Check that we build a data frame with all the expected columns.
    df = builder.build_nested_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    for col in ["id", "ra", "dec", "t0", "redshift", "nobs", "lightcurve", "spectra"]:
        assert col in df.columns
    assert df["id"].iloc[0] == 0
    assert df["ra"].iloc[0] == 0.0
    assert df["dec"].iloc[0] == 0.0
    assert df["t0"].iloc[0] == 0.0
    assert df["redshift"].iloc[0] == 0.0
    assert df["nobs"].iloc[0] == 0
    assert df["lightcurve"].iloc[0] is None
    assert df["spectra"].iloc[0] is None

    # We fail if we try to add light curve information without starting a new object.
    with pytest.raises(ValueError):
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
    builder.append_current_object()

    # Check that we build a data frame with all the expected columns.
    df = builder.build_nested_frame()
    lc0 = df["lightcurve"].iloc[0]
    assert len(lc0) == 3
    for col in _LC_COLS:
        assert col in lc0
    assert df["spectra"].iloc[0] is None
    assert df["nobs"].iloc[0] == 3

    # We fail if we try to start an object with an index out of bounds.
    with pytest.raises(ValueError):
        builder.start_new_object(1)


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
    ra = np.array([0.0])
    dec = np.array([0.0])
    t0 = np.array([0.0])
    redshift = np.array([0.0])
    builder = SimulationResultBuilder(ra, dec, t0, redshift)

    # Fail if we try to add a column with the wrong number of observations.
    with pytest.raises(ValueError):
        builder.add_result_column("wrong_length_column", [1, 2])

    # Succeed if we add a column with the correct number of observations.
    builder.add_result_column("correct_length_column", ["str"])
    assert "correct_length_column" in builder.results_dict
    assert builder.results_dict["correct_length_column"] == ["str"]
    assert builder.results_dtypes["correct_length_column"] == str

    # Check that the column was added correctly.
    df = builder.build_nested_frame()
    assert "correct_length_column" in df.columns
    assert df["correct_length_column"].iloc[0] == "str"
    assert isinstance(df["correct_length_column"].dtype, pd.StringDtype)

    # Check that we can override the dtype.
    builder.add_result_column("overridden_dtype_column", ["str"], dtype=object)
    assert builder.results_dtypes["overridden_dtype_column"] == object

    df = builder.build_nested_frame()
    assert df["overridden_dtype_column"].dtype == object

    # Check that we can build a column automatically from a GraphState.
    state = GraphState()
    state.set("a", "v1", 1.0)
    builder.add_param_column("a.v1", state)
    assert "a_v1" in builder.results_dict
    assert builder.results_dict["a_v1"] == [1.0]
    assert builder.results_dtypes["a_v1"] == float

    df = builder.build_nested_frame()
    assert "a_v1" in df.columns
    assert df["a_v1"].iloc[0] == 1.0


def test_simulation_results_builder_sort_light_curves():
    """Test that we sort the light curves for each object by MJD."""
    ra = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dec = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    t0 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    redshift = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    builder = SimulationResultBuilder(ra, dec, t0, redshift)

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
    df = builder.build_nested_frame()
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
    ra = np.array([0.0, 1.0, 2.0])
    dec = np.array([0.0, 1.0, 2.0])
    t0 = np.array([0.0, 1.0, 2.0])
    redshift = np.array([0.0, 1.0, 2.0])
    builder = SimulationResultBuilder(ra, dec, t0, redshift)

    # We fail if we try to add spectra information without starting a new object.
    with pytest.raises(ValueError):
        builder.add_spectra_observations(
            mjd=[0.0, 0.0, 0.0],
            waves=[4000.0, 5000.0, 6000.0],
            measured_flux=[1.0, 1.0, 1.0],
            measured_flux_perfect=[1.0, 1.0, 1.0],
            measured_flux_error=[0.1, 0.1, 0.1],
            instrument=["inst1", "inst1", "inst1"],
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
    df = builder.build_nested_frame()
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
