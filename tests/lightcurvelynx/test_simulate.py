from multiprocessing import Pool

import numpy as np
import pytest
from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.basic_models import ConstantSEDModel, StepModel
from lightcurvelynx.models.static_sed_model import StaticBandfluxModel
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import (
    SimulationInfo,
    compute_noise_free_lightcurves,
    compute_single_noise_free_lightcurve,
    get_time_windows,
    simulate_lightcurves,
)


def test_get_time_windows():
    """Test the get_time_windows function with various inputs."""
    assert get_time_windows(None, None) == (None, None)
    assert get_time_windows(0.0, None) == (None, None)
    assert get_time_windows(None, (1.0, 2.0)) == (None, None)

    result = get_time_windows(0.0, (1.0, 2.0))
    assert np.array_equal(result[0], np.array([-1.0]))
    assert np.array_equal(result[1], np.array([2.0]))

    result = get_time_windows(1.0, (None, 2.0))
    assert result[0] is None
    assert np.array_equal(result[1], np.array([3.0]))

    result = get_time_windows(-10.0, (1.0, None))
    assert np.array_equal(result[0], np.array([-11.0]))
    assert result[1] is None

    result = get_time_windows(np.array([0.0, 1.0, 2.0]), (1.0, 2.0))
    assert np.array_equal(result[0], np.array([-1.0, 0.0, 1.0]))
    assert np.array_equal(result[1], np.array([2.0, 3.0, 4.0]))

    with pytest.raises(ValueError):
        get_time_windows(0.0, (1.0, 2.0, 3.0))


def test_simulation_info():
    """Test that we can create and split SimulationInfo objects."""
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0, node_label="source")

    # Create a completely fake passband group and obstable for testing.
    pb_group = PassbandGroup(
        [
            Passband(np.array([[100, 0.5], [200, 0.75], [300, 0.25]]), "my_survey", "a"),
            Passband(np.array([[250, 0.25], [300, 0.5], [350, 0.75]]), "my_survey", "b"),
        ]
    )

    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(values, zp_per_band=zp_per_band, fwhm_px=2.0, sky=100.0)

    sim_info = SimulationInfo(
        model=model,
        num_samples=100,
        obstable=ops_data,
        passbands=pb_group,
        time_window_offset=(-5.0, 10.0),  # A keyword argument to test
        rng=np.random.default_rng(12345),
    )
    assert sim_info.num_samples == 100
    assert sim_info.time_window_offset == (-5.0, 10.0)

    # Test splitting into batches.
    batches = sim_info.split(num_batches=3)
    assert len(batches) == 3
    assert batches[0].num_samples == 34
    assert batches[1].num_samples == 34
    assert batches[2].num_samples == 32

    # We propagate all the keyword parameters.
    assert batches[0].time_window_offset == (-5.0, 10.0)
    assert batches[1].time_window_offset == (-5.0, 10.0)
    assert batches[2].time_window_offset == (-5.0, 10.0)

    # They have different RNGs with different sampling states.
    assert batches[0].rng is not batches[1].rng
    assert batches[0].rng is not batches[2].rng
    assert batches[1].rng is not batches[2].rng
    sample0 = batches[0].rng.integers(0, 100000)
    sample1 = batches[1].rng.integers(0, 100000)
    sample2 = batches[2].rng.integers(0, 100000)
    assert sample0 != sample1
    assert sample0 != sample2
    assert sample1 != sample2


def test_simulate_lightcurves(test_data_dir):
    """Test an end to end run of simulating the light curves."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSEDModel(
        brightness=GivenValueList(given_brightness),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].values[0:5]),
        dec=GivenValueList(opsim_db["dec"].values[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        5,
        opsim_db,
        passband_group,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
    )
    assert len(results) == 5
    assert np.all(results["nobs"].values >= 1)
    assert np.allclose(results["ra"].values, opsim_db["ra"].values[0:5])
    assert np.allclose(results["dec"].values, opsim_db["dec"].values[0:5])
    assert np.allclose(results["z"].values, 0.0)
    assert np.allclose(results["t0"].values, 0.0)

    for idx in range(5):
        num_obs = results["nobs"][idx]
        assert len(results.loc[idx]["lightcurve"]["flux"]) == num_obs

        # Check that we pulled the metadata from the opsim.
        assert len(results.loc[idx]["lightcurve"]["observationId"]) == num_obs
        assert len(np.unique(results.loc[idx]["lightcurve"]["observationId"])) == num_obs
        assert np.all(results.loc[idx]["lightcurve"]["observationId"] >= 0)
        assert len(results.loc[idx]["lightcurve"]["zp_nJy"]) == num_obs

        # Check that we extract one of the parameters.
        assert results["source_brightness"][idx] == given_brightness[idx]

    # Check that we saved and can reassemble the GraphStates
    assert "params" in results
    state = GraphState.from_list(results["params"].values)
    assert state.num_samples == 5
    assert np.allclose(state["source.ra"], opsim_db["ra"].values[0:5])
    assert np.allclose(state["source.dec"], opsim_db["dec"].values[0:5])

    # Check that we can extract a single GraphState from the results.
    single_state = GraphState.from_dict(results["params"][2])
    assert single_state.num_samples == 1
    assert single_state["source.ra"] == opsim_db["ra"].values[2]
    assert single_state["source.dec"] == opsim_db["dec"].values[2]

    # Check that we fail if we try to save a parameter column that doesn't exist.
    # And that the error gives the existing options.
    source2 = ConstantSEDModel(
        brightness=10.0,
        t0=0.0,
        ra=1.0,
        dec=-1.0,
        redshift=0.0,
        node_label="source2",
    )
    with pytest.raises(KeyError) as excinfo:
        _ = simulate_lightcurves(
            source2,
            1,
            opsim_db,
            passband_group,
            param_cols=["source2.unknown_parameter"],
        )
    assert "Available parameters are:" in str(excinfo.value)
    assert "source2.ra" in str(excinfo.value)
    assert "source2.dec" in str(excinfo.value)


def test_simulate_bandfluxes(test_data_dir):
    """Test an end to end run of simulating a bandflux model."""
    # Create a toy observation table with two pointings.
    num_obs = 6
    obsdata = {
        "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "ra": [0.0, 0.0, 180.0, 0.0, 180.0, 0.0],
        "dec": [10.0, 10.0, -10.0, 10.0, -10.0, 10.0],
        "filter": ["g", "r", "g", "r", "z", "z"],
        "zp": [1.0] * num_obs,
        "seeing": [1.12] * num_obs,
        "skybrightness": [20.0] * num_obs,
        "exptime": [29.2] * num_obs,
        "nexposure": [2] * num_obs,
    }
    obstable = OpSim(obsdata)

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a static bandflux model and simulate 2 runs.
    model = StaticBandfluxModel({"g": 1.0, "i": 2.0, "r": 3.0, "y": 4.0, "z": 5.0}, ra=0.0, dec=10.0)
    results = simulate_lightcurves(model, 2, obstable, passband_group)
    assert len(results) == 2
    assert np.allclose(results["lightcurve"][0]["flux_perfect"], [1.0, 3.0, 3.0, 5.0])
    assert np.allclose(results["lightcurve"][1]["flux_perfect"], [1.0, 3.0, 3.0, 5.0])


def test_simulate_parallel(test_data_dir):
    """Test an end to end run of simulating the light curves."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    ra0 = opsim_db["ra"].values[0]
    dec0 = opsim_db["dec"].values[0]
    source = ConstantSEDModel(
        brightness=NumpyRandomFunc("uniform", low=100.0, high=500.0),
        t0=0.0,
        ra=NumpyRandomFunc("uniform", low=ra0 - 0.5, high=ra0 + 0.5),
        dec=NumpyRandomFunc("uniform", low=dec0 - 0.5, high=dec0 + 0.5),
        redshift=0.0,
        node_label="source",
    )

    with Pool(processes=2) as pool:
        results = simulate_lightcurves(
            source,
            100,
            opsim_db,
            passband_group,
            obstable_save_cols=["observationId", "zp_nJy"],
            param_cols=["source.brightness"],
            pool=pool,
            batch_size=10,
        )
    assert len(results) == 100
    assert np.all(results["nobs"].values >= 1)
    assert np.all(results["ra"].values >= ra0 - 0.5)
    assert np.all(results["ra"].values <= ra0 + 0.5)
    assert np.all(results["dec"].values >= dec0 - 0.5)
    assert np.all(results["dec"].values <= dec0 + 0.5)

    for idx in range(100):
        num_obs = results["nobs"][idx]
        assert num_obs >= 1
        assert len(results["lightcurve"][idx]["flux"]) == num_obs


def test_simulate_single_lightcurve(test_data_dir):
    """Test an end to end run of simulating a single light curve."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSEDModel(
        brightness=GivenValueList(given_brightness),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].values[0:5]),
        dec=GivenValueList(opsim_db["dec"].values[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        1,
        opsim_db,
        passband_group,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
    )
    assert len(results) == 1

    # Check that we saved and can reassemble the GraphStates
    assert "params" in results
    state = GraphState.from_list(results["params"].values)
    assert state.num_samples == 1
    assert state["source.ra"] == opsim_db["ra"].values[0]
    assert state["source.dec"] == opsim_db["dec"].values[0]


def test_simulate_with_time_window(test_data_dir):
    """Test an end to end run of simulating with a limited time window."""
    # Create a toy OpSim database with two pointings over a series of tiems.
    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.arange(50.0),
        "fieldRA": np.array([15.0 if i % 2 == 0 else 180.0 for i in range(50)]),
        "fieldDec": np.array([10.0 if i % 2 == 0 else -10.0 for i in range(50)]),
        "filter": np.full(50, "g"),
        # We add the remaining values so the OpSim can compute noise, but they are
        # arbitrary and not tested in this test.
        "zp_nJy": np.ones(50),
        "seeingFwhmEff": np.ones(50) * 0.7,
        "visitExposureTime": np.ones(50) * 30.0,
        "numExposures": np.ones(50) * 1,
        "skyBrightness": np.full(50, 20.0),
    }
    opsim_db = OpSim(values)

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    source = ConstantSEDModel(
        brightness=1000.0,
        t0=GivenValueList([20.0, 15.0]),
        ra=15.0,
        dec=10.0,
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        2,
        opsim_db,
        passband_group,
        time_window_offset=(5.0, 10.0),
    )
    assert len(results) == 2

    # We should simulate the observations that are only within the time window, (15.0, 30.0) for the
    # first samples and (10.0, 25.0) for the second sample, and at the matching RA/Dec (the even indices).
    assert np.array_equal(
        results["lightcurve"][0]["mjd"],
        np.array([16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]),
    )
    assert np.array_equal(
        results["lightcurve"][1]["mjd"],
        np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]),
    )


def test_simulate_multiple_surveys(test_data_dir):
    """Test an end to end run of simulating a single light curve from multiple surveys."""
    # The first survey points at two locations in the sky in the "g" and "r" bands.
    obsdata1 = {
        "time": [0.0, 1.0, 2.0, 3.0],
        "ra": [0.0, 0.0, 180.0, 180.0],
        "dec": [10.0, 10.0, -10.0, -10.0],
        "filter": ["g", "r", "g", "r"],
        "zp": [0.4, 0.5, 0.6, 0.7],
        "seeing": [1.12, 1.12, 1.12, 1.12],
        "skybrightness": [20.0, 20.0, 20.0, 20.0],
        "exptime": [29.2, 29.2, 29.2, 29.2],
        "nexposure": [2, 2, 2, 2],
        "custom_col": [1, 1, 1, 1],
    }
    obstable1 = OpSim(obsdata1)
    passband_group1 = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r"],
    )

    # The second survey points at two locations on the sky in the "r" and "z" bands.
    obsdata2 = {
        "time": [0.5, 1.5, 2.5, 3.5],
        "ra": [0.0, 90.0, 0.0, 90.0],
        "dec": [10.0, -10.0, 10.0, -10.0],
        "filter": ["r", "z", "r", "z"],
        "zp": [0.05, 0.1, 0.2, 0.3],
        "seeing": [1.12, 1.12, 1.12, 1.12],
        "skybrightness": [20.0, 20.0, 20.0, 20.0],
        "exptime": [29.2, 29.2, 29.2, 29.2],
        "nexposure": [2, 2, 2, 2],
    }
    obstable2 = OpSim(obsdata2)
    passband_group2 = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["r", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec values that
    # match the (0.0, 10.0) pointing.
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=10.0, redshift=0.0, node_label="source")
    results = simulate_lightcurves(
        model,
        1,
        [obstable1, obstable2],
        [passband_group1, passband_group2],
        obstable_save_cols=["zp", "custom_col"],
    )
    assert len(results) == 1
    assert results["nobs"][0] == 4

    # Check that the light curve was simulated correctly, including saving the zeropoint information
    # from each ObsTable.
    lightcurve = results["lightcurve"][0]
    assert np.allclose(lightcurve["mjd"], np.array([0.0, 1.0, 0.5, 2.5]))
    assert np.allclose(lightcurve["zp"], np.array([0.4, 0.5, 0.05, 0.2]))
    assert np.array_equal(lightcurve["filter"], np.array(["g", "r", "r", "r"]))
    assert np.array_equal(lightcurve["survey_idx"], np.array([0, 0, 1, 1]))

    # The custom column should only exist for observations from one of the surveys.
    assert np.all(lightcurve["custom_col"][0:2] == 1)
    assert np.all(np.isnan(lightcurve["custom_col"][2:4]))

    # We fail if we pass in lists of different lengths.
    with pytest.raises(ValueError):
        simulate_lightcurves(
            model,
            1,
            [obstable1, obstable2],
            passband_group1,
        )

    # We fail if we try to use a bandflux only model with multiple surveys.
    model2 = StaticBandfluxModel({"g": 1.0, "i": 1.0, "r": 1.0, "y": 1.0, "z": 1.0})
    with pytest.raises(ValueError):
        simulate_lightcurves(
            model2,
            1,
            [obstable1, obstable2],
            [passband_group1, passband_group2],
        )


def test_compute_noise_free_lightcurves_single(test_data_dir):
    """Test computing noise free light curves for a set of times and filters."""
    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a step model that changes brightness at t=10.0
    model = StepModel(brightness=100.0, t0=10.0, t1=30.0, node_label="source")
    graph_state = model.sample_parameters(num_samples=1)

    lightcurves = compute_single_noise_free_lightcurve(
        model,
        graph_state,
        passband_group,
        rest_frame_phase_min=-40.0,  # 40 days before t0
        rest_frame_phase_max=60.0,  # 60 days after t0
        rest_frame_phase_step=1.0,  # 1 sample per day
    )
    rest_phase = np.arange(-40.0, 60.0, 1.0)
    obs_times = rest_phase + 10.0
    assert np.array_equal(lightcurves["times"], obs_times)
    assert np.array_equal(lightcurves["rest_phase"], rest_phase)

    mask = (obs_times >= 10.0) & (obs_times <= 30.0)
    for filter in ["g", "r", "i", "z"]:
        assert filter in lightcurves
        bandfluxes = lightcurves[filter]
        assert len(bandfluxes) == len(obs_times)
        assert np.allclose(bandfluxes[~mask], 0.0)
        assert np.allclose(bandfluxes[mask], 100.0)


def test_compute_noise_free_lightcurves_multiple(test_data_dir):
    """Test computing noise free light curves for a set of times and filters
    and multiple sample states"""
    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a step model that changes brightness at t0
    start_times = [10.0, 15.0, 20.0, 25.0]
    end_times = [30.0, 30.0, 40.0, 40.0]
    model = StepModel(
        brightness=100.0,
        t0=GivenValueList(start_times),
        t1=GivenValueList(end_times),
        ra=0.0,
        dec=0.0,
        redshift=0.0,
        node_label="source",
    )
    graph_state = model.sample_parameters(num_samples=4)

    lightcurves = compute_noise_free_lightcurves(
        model,
        graph_state,
        passband_group,
        rest_frame_phase_min=-40.0,  # 40 days before t0
        rest_frame_phase_max=60.0,  # 60 days after t0
        rest_frame_phase_step=1.0,  # 1 sample per day
    )
    assert len(lightcurves) == 4

    rest_phase = np.arange(-40.0, 60.0, 1.0)
    for idx in range(4):
        assert lightcurves["id"][idx] == idx
        lc_df = lightcurves["lightcurve"][idx]
        assert np.array_equal(lc_df["rest_phase"], rest_phase)
        assert np.array_equal(lc_df["times"], rest_phase + start_times[idx])

        mask = (lc_df["times"] >= start_times[idx]) & (lc_df["times"] <= end_times[idx])
        for filter in ["g", "r", "i", "z"]:
            assert filter in lc_df
            bandfluxes = lc_df[filter]
            assert len(bandfluxes) == len(lc_df["times"])
            assert np.allclose(bandfluxes[~mask], 0.0)
            assert np.allclose(bandfluxes[mask], 100.0)
