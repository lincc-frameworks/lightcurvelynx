import copy
import gc
import pickle
import tempfile
import warnings
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from nested_pandas import NestedFrame, read_parquet

import lightcurvelynx.simulate as simulate_module
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.astro_utils.spectrograph import Spectrograph
from lightcurvelynx.astro_utils.unit_utils import fnu_to_flam
from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.given_sampler import (
    GivenValueList,
    GivenValueSampler,
    TableSampler,
)
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.state_expansion_node import StateExpansionNode
from lightcurvelynx.models.basic_models import ConstantSEDModel, SinWaveModel, StepModel
from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.models.static_sed_model import StaticBandfluxModel
from lightcurvelynx.noise_models.base_noise_models import GivenNoiseModel
from lightcurvelynx.noise_models.spectrograph_noise_models import ConstantSpectrographNoiseModel
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.obstable.spectrograph_table import SpectrographObsTable
from lightcurvelynx.obstable.ztf_obstable import ZTFObsTable, create_random_ztf_obs_data
from lightcurvelynx.simulate import (
    SimulationInfo,
    WorkerPreloadWarning,
    compute_noise_free_lightcurves,
    compute_single_noise_free_lightcurve,
    get_time_windows,
    make_worker_initializer,
    simulate_lightcurves,
)
from lightcurvelynx.survey_info import SurveyInfo
from lightcurvelynx.utils.worker_preload import (
    load_preloaded_data,
    reset_preload_state,
    serialize_simulation_data,
)


def test_get_time_windows():
    """Test the get_time_windows function with various inputs."""
    # No bounds
    assert get_time_windows(None, None, None, None) == (None, None)

    # No t0
    assert get_time_windows(0.0, None, None, None) == (None, None)
    assert get_time_windows(None, None, (1.0, 2.0), None) == (None, None)
    assert get_time_windows(None, None, None, (1.0, 2.0)) == (None, None)
    assert get_time_windows(0.0, 2.0, None, None) == (None, None)
    assert get_time_windows(None, 2.0, (1.0, 2.0), None) == (None, None)
    assert get_time_windows(None, 2.0, None, (1.0, 2.0)) == (None, None)

    # t0 = 0.0, z=1.0, time_window = (1.0, 2.0) in observer frame
    result = get_time_windows(0.0, 1.0, (-1.0, 2.0), None)
    assert np.array_equal(result[0], np.array([-1.0]))
    assert np.array_equal(result[1], np.array([2.0]))

    # t0 = 0.0, z=1.0, time_window = (inf, 2.0) in observer frame
    result = get_time_windows(1.0, 1.0, (None, 2.0), None)
    assert result[0] is None
    assert np.array_equal(result[1], np.array([3.0]))

    # t0 = -10.0, z=1.0, time_window = (-1.0, inf) in observer frame
    result = get_time_windows(-10.0, 1.0, (-1.0, None), None)
    assert np.array_equal(result[0], np.array([-11.0]))
    assert result[1] is None

    # t0 = array, z=1.0, time_window = (-1.0, 2.0) in observer frame
    result = get_time_windows(np.array([0.0, 1.0, 2.0]), 1.0, (-1.0, 2.0), None)
    assert np.array_equal(result[0], np.array([-1.0, 0.0, 1.0]))
    assert np.array_equal(result[1], np.array([2.0, 3.0, 4.0]))

    # Wrong size bounds tuple.
    with pytest.raises(ValueError):
        get_time_windows(0.0, None, (1.0, 2.0, 3.0), None)

    # t0 = array, z=1.5
    # time_window = (inf, 2.0) in rest frame = (inf, 5.0) in rest frame
    result = get_time_windows(np.array([0.0, 1.0, 2.0]), 1.5, None, (None, 2.0))
    assert result[0] is None
    assert np.array_equal(result[1], np.array([5.0, 6.0, 7.0]))

    # t0 = array, z=2.0
    # time_window = (-1.0, inf) in rest frame = (-3.0, inf) in rest frame
    result = get_time_windows(np.array([0.0, 1.0, 2.0]), 2.0, None, (-1.0, None))
    assert np.array_equal(result[0], np.array([-3.0, -2.0, -1.0]))
    assert result[1] is None

    # t0 = array, z=1.0
    # time_window = (-1.0, 2.0) in rest frame = (-2.0, 4.0) in rest frame
    result = get_time_windows(np.array([0.0, 1.0, 2.0]), 1.0, None, (-1.0, 2.0))
    assert np.array_equal(result[0], np.array([-2.0, -1.0, 0.0]))
    assert np.array_equal(result[1], np.array([4.0, 5.0, 6.0]))

    # t0 = array, z = array, time_window = (-1.0, 2.0) in rest frame.
    # The obs frame windows get larger as z increases.
    result = get_time_windows(
        np.array([0.0, 1.0, 2.0]),  # Different t0 values
        np.array([1.0, 1.5, 2.0]),  # Different z values
        None,
        (-1.0, 2.0),
    )
    assert np.array_equal(result[0], np.array([-2.0, -1.5, -1.0]))
    assert np.array_equal(result[1], np.array([4.0, 6.0, 8.0]))

    # t0 = 1.0, z=0.1
    result = get_time_windows(1.0, 0.1, None, (-100.0, 102.0))
    assert result[0] == pytest.approx(-109.0)  # 1.0 + (-100.0 * (1 + 0.1))
    assert result[1] == pytest.approx(113.2)  # 1.0 + (102.0 * (1 + 0.1))

    with pytest.raises(ValueError):
        # No z value provided.
        result = get_time_windows(np.array([0.0, 1.0, 2.0]), None, None, (-1.0, 2.0))

    with pytest.raises(ValueError):
        # Wrong size bounds.
        result = get_time_windows(np.array([0.0, 1.0, 2.0]), None, None, (-1.0, 2.0, 3.0))

    with pytest.raises(ValueError):
        # t0 is a scalar but z is an array.
        result = get_time_windows(0.01, np.array([0.1, 0.2]), None, (-1.0, 2.0))

    with pytest.raises(ValueError):
        # t0 and z are incompatible shapes
        result = get_time_windows(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2]), None, (-1.0, 2.0))


def test_simulation_info():
    """Test that we can create and split SimulationInfo objects."""
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0, node_label="source")

    # Create a completely fake passband group and obstable for testing.
    pb_group = PassbandGroup(
        [
            Passband(np.array([[100, 0.5], [200, 0.75], [300, 0.25]]), "my_survey", "r"),
            Passband(np.array([[250, 0.25], [300, 0.5], [350, 0.75]]), "my_survey", "g"),
            Passband(np.array([[250, 0.25], [300, 0.5], [350, 0.75]]), "my_survey", "i"),
        ]
    )

    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    bandflux_error = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(
        values,
        bandflux_error=bandflux_error,
        fwhm_px=2.0,
        sky_bg_electrons=100.0,
    )

    survey_info = SurveyInfo(
        obstable=ops_data,
        passbands=pb_group,
        noise_model=GivenNoiseModel(),  # Uses the bandflux_error from the obs table
    )

    sim_info = SimulationInfo(
        model=model,
        num_samples=100,
        survey_info=survey_info,
        obs_time_window_offset=(-5.0, 10.0),  # A keyword argument to test
        rng=np.random.default_rng(12345),
    )
    assert sim_info.num_samples == 100
    assert sim_info.obs_time_window_offset == (-5.0, 10.0)
    assert sim_info.progress_bar is True
    assert sim_info.graph_state is None

    # Test splitting into batches.
    batches = sim_info.split(num_batches=3)
    assert len(batches) == 3
    assert batches[0].num_samples == 34
    assert batches[0].sample_offset == 0
    assert batches[0].graph_state is None
    assert batches[1].num_samples == 34
    assert batches[1].sample_offset == 34
    assert batches[1].graph_state is None
    assert batches[2].num_samples == 32
    assert batches[2].sample_offset == 68
    assert batches[2].graph_state is None

    # We propagate all the keyword parameters.
    assert batches[0].obs_time_window_offset == (-5.0, 10.0)
    assert batches[1].obs_time_window_offset == (-5.0, 10.0)
    assert batches[2].obs_time_window_offset == (-5.0, 10.0)

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

    with pytest.raises(ValueError):
        # Neither num_samples nor num_batches is specified.
        sim_info.split()
    with pytest.raises(ValueError):
        # num_samples and batch_size are both specified.
        sim_info.split(batch_size=10, num_batches=2)
    with pytest.raises(ValueError):
        # Negative num_batches.
        sim_info.split(num_batches=-1)
    with pytest.raises(ValueError):
        # Negative batch_size.
        sim_info.split(batch_size=-10)

    # Fail with a negative number of samples.
    with pytest.raises(ValueError):
        _ = SimulationInfo(
            model=model,
            num_samples=-10,
            survey_info=survey_info,
            obs_time_window_offset=(-5.0, 10.0),  # A keyword argument to test
            rng=np.random.default_rng(12345),
        )


def test_simulation_info_random_split():
    """Test that is we split two SimulationInfo objects with the same seed,
    each result has the same set of RNG states."""
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0, node_label="source")

    # Create a completely fake passband group and obstable for testing.
    pb_group = PassbandGroup(
        [
            Passband(np.array([[100, 0.5], [200, 0.75], [300, 0.25]]), "my_survey", "r"),
            Passband(np.array([[250, 0.25], [300, 0.5], [350, 0.75]]), "my_survey", "g"),
            Passband(np.array([[250, 0.25], [300, 0.5], [350, 0.75]]), "my_survey", "i"),
        ]
    )

    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    bandflux_error = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(
        values,
        bandflux_error=bandflux_error,
        fwhm_px=2.0,
        sky_bg_electrons=100.0,
    )

    survey_info = SurveyInfo(
        obstable=ops_data,
        passbands=pb_group,
        noise_model=GivenNoiseModel(),  # Uses the bandflux_error from the obs table
    )

    # Create a SimulationInfo object with a given seed.
    sim_info1 = SimulationInfo(
        model=model,
        num_samples=100,
        survey_info=survey_info,
        rng=np.random.default_rng(12345),
    )
    batches1 = sim_info1.split(num_batches=5)
    assert len(batches1) == 5

    # Create a second SimulationInfo object with the same given seed.
    sim_info2 = SimulationInfo(
        model=model,
        num_samples=100,
        survey_info=survey_info,
        rng=np.random.default_rng(12345),
    )
    batches2 = sim_info2.split(num_batches=5)
    assert len(batches2) == 5

    # For each batch, test for each batch's random number generator is the same
    # for both SimulationInfo objects (since they had the same seed). But the random
    # number generators vary within the batch.
    random_samples1 = [batch.rng.integers(0, 100000) for batch in batches1]
    random_samples2 = [batch.rng.integers(0, 100000) for batch in batches2]
    assert np.all(random_samples1 == random_samples2)
    assert len(np.unique(random_samples1)) == 5


def test_simulation_info_graph_state():
    """Test that we can create and split SimulationInfo objects with given graph state."""
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0, node_label="source")
    state = model.sample_parameters(num_samples=100)

    # Create a completely fake passband group and obstable for testing.
    pb_group = PassbandGroup(
        [
            Passband(np.array([[100, 0.5], [200, 0.75], [300, 0.25]]), "my_survey", "r"),
            Passband(np.array([[250, 0.25], [300, 0.5], [350, 0.75]]), "my_survey", "g"),
            Passband(np.array([[250, 0.25], [300, 0.5], [350, 0.75]]), "my_survey", "i"),
        ]
    )

    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    bandflux_error = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(
        values,
        bandflux_error=bandflux_error,
        fwhm_px=2.0,
        sky_bg_electrons=100.0,
    )

    survey_info = SurveyInfo(
        obstable=ops_data,
        passbands=pb_group,
        noise_model=GivenNoiseModel(),  # Uses the bandflux_error from the obs table
    )

    sim_info = SimulationInfo(
        model=model,
        num_samples=100,
        survey_info=survey_info,
        obs_time_window_offset=(-5.0, 10.0),  # A keyword argument to test
        rng=np.random.default_rng(12345),
        graph_state=state,
    )
    assert sim_info.num_samples == 100
    assert sim_info.obs_time_window_offset == (-5.0, 10.0)
    assert sim_info.progress_bar is True
    assert sim_info.graph_state == state

    # Test splitting into batches.
    batches = sim_info.split(num_batches=3)
    assert len(batches) == 3
    assert batches[0].num_samples == 34
    assert batches[0].sample_offset == 0
    assert batches[0].graph_state.sample_offset == 0
    assert batches[0].graph_state.num_samples == 34
    assert batches[1].num_samples == 34
    assert batches[1].sample_offset == 34
    assert batches[1].graph_state.sample_offset == 34
    assert batches[1].graph_state.num_samples == 34
    assert batches[2].num_samples == 32
    assert batches[2].sample_offset == 68
    assert batches[2].graph_state.sample_offset == 68
    assert batches[2].graph_state.num_samples == 32

    # Check that we fail if we try to create a SimulationInfo with a graph state that has a
    # different number of samples than num_samples.
    with pytest.raises(ValueError):
        _ = SimulationInfo(
            model=model,
            num_samples=50,  # Wrong number
            survey_info=survey_info,
            obs_time_window_offset=(-5.0, 10.0),  # A keyword argument to test
            rng=np.random.default_rng(12345),
            graph_state=state,
        )


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

    # Use the default noise model.
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSEDModel(
        brightness=GivenValueList(given_brightness),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].to_numpy()[0:5]),
        dec=GivenValueList(opsim_db["dec"].to_numpy()[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        5,
        survey_info,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results) == 5
    assert "lightcurve" in results
    assert "spectra" not in results

    assert np.all(results["nobs"].to_numpy() >= 1)
    assert np.allclose(results["ra"].to_numpy(), opsim_db["ra"].to_numpy()[0:5])
    assert np.allclose(results["dec"].to_numpy(), opsim_db["dec"].to_numpy()[0:5])
    assert np.allclose(results["z"].to_numpy(), 0.0)
    assert np.allclose(results["t0"].to_numpy(), 0.0)

    for idx in range(5):
        num_obs = results["nobs"][idx]
        assert len(results.loc[idx]["lightcurve"]["flux"]) == num_obs

        # Check that we pulled the metadata from the opsim.
        assert len(results.loc[idx]["lightcurve"]["observationId"]) == num_obs
        assert len(np.unique(results.loc[idx]["lightcurve"]["observationId"])) == num_obs
        assert np.all(results.loc[idx]["lightcurve"]["observationId"] >= 0)
        assert len(results.loc[idx]["lightcurve"]["zp_nJy"]) == num_obs

        # Check that we have the survey and obs indices. All observations come from the first
        # survey, and the obs indices are unique.
        assert len(results.loc[idx]["lightcurve"]["survey_idx"]) == num_obs
        assert np.all(results.loc[idx]["lightcurve"]["survey_idx"] == 0)
        assert len(results.loc[idx]["lightcurve"]["obs_idx"]) == num_obs
        assert len(np.unique(results.loc[idx]["lightcurve"]["obs_idx"])) == num_obs

        # We do not include the full filter name unless that option is selected.
        assert "full_filter_name" not in results.loc[idx]["lightcurve"]

        # Check that we extract one of the parameters.
        assert results["source_brightness"][idx] == given_brightness[idx]

    # We can access a specific set of parameters for one sample as a dict or
    # all parameters for all samples as a StructArray.
    assert "params" in results
    assert isinstance(results["params"][0], dict)
    assert len(results["params"].struct.field("source.brightness")) == 5

    # Check that we can safely write the results to a file.
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "test_simulate.parquet"
        assert not filename.exists()

        results.to_parquet(filename)
        assert filename.exists()

    # Check that we saved and can reassemble the GraphStates
    assert "params" in results
    state = GraphState.from_list(results["params"].to_numpy())
    assert state.num_samples == 5
    assert np.allclose(state["source.ra"], opsim_db["ra"].to_numpy()[0:5])
    assert np.allclose(state["source.dec"], opsim_db["dec"].to_numpy()[0:5])

    # Check that we can extract a single GraphState from the results.
    single_state = GraphState.from_dict(results["params"][2])
    assert single_state.num_samples == 1
    assert single_state["source.ra"] == opsim_db["ra"].to_numpy()[2]
    assert single_state["source.dec"] == opsim_db["dec"].to_numpy()[2]

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
            survey_info,
            param_cols=["source2.unknown_parameter"],
            progress_bar=False,  # Disable progress bar for testing
        )
    assert "Available parameters are:" in str(excinfo.value)
    assert "source2.ra" in str(excinfo.value)
    assert "source2.dec" in str(excinfo.value)


def test_replay_simulate_lightcurves(test_data_dir):
    """Test an end to end run of simulating the light curves using a given state."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Use the default noise model.
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSEDModel(
        brightness=GivenValueList(given_brightness),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].to_numpy()[0:5]),
        dec=GivenValueList(opsim_db["dec"].to_numpy()[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        5,
        survey_info,
        progress_bar=False,  # Disable progress bar for testing
    )

    # Test that we can replay the simulation using the saved GraphState.
    state = GraphState.from_list(results["params"].to_numpy())
    replay_results = simulate_lightcurves(
        source,
        5,
        survey_info,
        progress_bar=False,  # Disable progress bar for testing
        graph_state=state,
    )
    assert np.allclose(replay_results["ra"].to_numpy(), results["ra"].to_numpy())
    assert np.allclose(replay_results["dec"].to_numpy(), results["dec"].to_numpy())
    assert np.allclose(replay_results["z"].to_numpy(), results["z"].to_numpy())
    assert np.allclose(replay_results["t0"].to_numpy(), results["t0"].to_numpy())
    assert np.allclose(
        replay_results["lightcurve.flux_perfect"].to_numpy(),
        results["lightcurve.flux_perfect"].to_numpy(),
    )

    # The noise realization will be different.
    assert not np.allclose(
        replay_results["lightcurve.flux"].to_numpy(),
        results["lightcurve.flux"].to_numpy(),
    )


def test_simulate_lightcurves_uses_single_progress_bar(test_data_dir, monkeypatch):
    """Test that serial and threaded runs share one progress bar instance."""
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    class DummyTqdm:
        instances = []

        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable
            self.total = kwargs.get("total")
            self.disabled = kwargs.get("disable", False)
            self.updates = []
            if not self.disabled:
                DummyTqdm.instances.append(self)

        def __iter__(self):
            return iter(self._iterable) if self._iterable is not None else iter([])

        def update(self, n=1):
            self.updates.append(n)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, exc_tb):
            return False

    monkeypatch.setattr(simulate_module, "tqdm", DummyTqdm)

    source1 = ConstantSEDModel(
        brightness=GivenValueList(
            [1000.0, 2000.0, 5000.0, 1000.0, 100.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0]
        ),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].to_numpy()[0:10]),
        dec=GivenValueList(opsim_db["dec"].to_numpy()[0:10]),
        redshift=0.0,
        node_label="source",
    )
    _ = simulate_lightcurves(source1, 5, survey_info, progress_bar=True)
    assert len(DummyTqdm.instances) == 1
    assert DummyTqdm.instances[0].total == 5

    DummyTqdm.instances.clear()

    class FakeMapExecutor:
        def map(self, fn, iterable):
            return [fn(item) for item in iterable]

    source2 = ConstantSEDModel(
        brightness=GivenValueList(
            [1000.0, 2000.0, 5000.0, 1000.0, 100.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0]
        ),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].to_numpy()[0:10]),
        dec=GivenValueList(opsim_db["dec"].to_numpy()[0:10]),
        redshift=0.0,
        node_label="source",
    )
    # The executor was not preloaded, so the run falls back with a WorkerPreloadWarning.
    with pytest.warns(WorkerPreloadWarning):
        _ = simulate_lightcurves(
            source2,
            5,
            survey_info,
            executor=FakeMapExecutor(),
            batch_size=2,
            progress_bar=True,  # Should only create one progress bar for the whole run.
        )
    assert len(DummyTqdm.instances) == 1
    assert DummyTqdm.instances[0].total == 5
    assert sum(DummyTqdm.instances[0].updates) == 5


def test_simulate_lightcurves_to_file(test_data_dir):
    """Test an end to end run of simulating the light curves, saving them to a file."""
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
        ra=GivenValueList(opsim_db["ra"].to_numpy()[0:5]),
        dec=GivenValueList(opsim_db["dec"].to_numpy()[0:5]),
        redshift=0.0,
        node_label="source",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_simulate_lightcurves.parquet"
        assert not file_path.exists()

        survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)
        results = simulate_lightcurves(
            source,
            5,
            survey_info,
            obstable_save_cols=["observationId", "zp_nJy"],
            param_cols=["source.brightness"],
            output_file_path=file_path,
            progress_bar=False,  # Disable progress bar for testing
        )
        assert str(results) == str(file_path)
        assert file_path.exists()

        results_df = read_parquet(file_path)
        assert len(results_df) == 5
        assert np.all(results_df["nobs"].to_numpy() >= 1)
        assert np.allclose(results_df["ra"].to_numpy(), opsim_db["ra"].to_numpy()[0:5])
        assert np.allclose(results_df["dec"].to_numpy(), opsim_db["dec"].to_numpy()[0:5])
        assert np.allclose(results_df["z"].to_numpy(), 0.0)
        assert np.allclose(results_df["t0"].to_numpy(), 0.0)


def test_simulate_lightcurves_reproduce(test_data_dir):
    """Test two end to end runs of simulation produce the same
    results when given the same random seed."""
    num_samples = 10_000

    # We load the OpSim data and passband data once.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    # Create a basic sinwave model with random parameters. RA and dec values
    # should match the opsim so we get at least one observation for each sample.
    source = SinWaveModel(
        amplitude=NumpyRandomFunc("uniform", low=100.0, high=200.0),
        frequency=NumpyRandomFunc("uniform", low=0.5, high=2.0),
        t0=0.0,
        ra=GivenValueSampler(opsim_db["ra"].to_numpy()[0:5]),
        dec=GivenValueSampler(opsim_db["dec"].to_numpy()[0:5]),
        redshift=NumpyRandomFunc("uniform", low=0.01, high=0.1),
        node_label="source",
    )

    # Simulate once with a given seed.
    rng1 = np.random.default_rng(54321)
    results1 = simulate_lightcurves(
        source,
        num_samples,
        survey_info,
        progress_bar=False,  # Disable progress bar for testing
        rng=rng1,
    )
    assert len(results1) == num_samples

    # Simulate again with the same seed.
    rng2 = np.random.default_rng(54321)
    results2 = simulate_lightcurves(
        source,
        num_samples,
        survey_info,
        progress_bar=False,  # Disable progress bar for testing
        rng=rng2,
    )
    assert len(results2) == num_samples

    # Check the main columns are the same. We don't check RA, because all 5
    # options for RA are 0.0. So they are always the same.
    assert np.allclose(results1["dec"].to_numpy(), results2["dec"].to_numpy())
    assert np.allclose(results1["z"].to_numpy(), results2["z"].to_numpy())

    # Check that the light curve columns are equal.
    for idx in range(num_samples):
        lc1 = results1.loc[idx]["lightcurve"]
        lc2 = results2.loc[idx]["lightcurve"]

        assert np.allclose(lc1["mjd"], lc2["mjd"])
        assert np.allclose(lc1["flux_perfect"], lc2["flux_perfect"])
        assert np.allclose(lc1["fluxerr"], lc2["fluxerr"])
        assert np.allclose(lc1["flux"], lc2["flux"])

    # Using a different seed should give different results.
    rng3 = np.random.default_rng(12345)
    results3 = simulate_lightcurves(
        source,
        num_samples,
        survey_info,
        progress_bar=False,  # Disable progress bar for testing
        rng=rng3,
    )
    assert len(results3) == num_samples

    # We don't check RA, because all 5 options for RA are 0.0.
    # So they are always the same.
    assert not np.allclose(results1["dec"].to_numpy(), results3["dec"].to_numpy())
    assert not np.allclose(results1["z"].to_numpy(), results3["z"].to_numpy())


def test_simulate_bandfluxes(test_data_dir):
    """Test an end to end run of simulating a bandflux model."""
    # Create a toy observation table with two pointings.
    num_obs = 8
    obsdata = {
        "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 6.0],
        "ra": [0.0, 0.0, 180.0, 0.0, 180.0, 0.0, 90.0, 90.0],
        "dec": [10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -20.0, -20.0],
        "filter": ["g", "r", "g", "r", "z", "z", "r", "g"],
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

    # Create a static bandflux model and simulate 2 runs. Check that we correctly extract per-observation
    # information, such as times, filters, and data indices.
    model = StaticBandfluxModel({"g": 1.0, "i": 2.0, "r": 3.0, "y": 4.0, "z": 5.0}, ra=0.0, dec=10.0)
    survey_info = SurveyInfo(obstable=obstable, passbands=passband_group)
    results = simulate_lightcurves(model, 2, survey_info, progress_bar=False)
    assert len(results) == 2
    for idx in range(2):
        assert np.allclose(results["lightcurve"][idx]["mjd"], [0.0, 1.0, 3.0, 5.0])
        assert np.allclose(results["lightcurve"][idx]["flux_perfect"], [1.0, 3.0, 3.0, 5.0])
        assert np.array_equal(results["lightcurve"][idx]["filter"], ["g", "r", "r", "z"])
        assert np.array_equal(results["lightcurve"][idx]["obs_idx"], [0, 1, 3, 5])
        assert np.array_equal(results["lightcurve"][idx]["survey_idx"], [0, 0, 0, 0])

    # Simulate a model with out of order times.
    model = StaticBandfluxModel({"g": 1.0, "i": 2.0, "r": 3.0, "y": 4.0, "z": 5.0}, ra=90.0, dec=-20.0)
    results = simulate_lightcurves(model, 1, survey_info, progress_bar=False)
    assert len(results) == 1
    assert np.allclose(results["lightcurve"][0]["mjd"], [6.0, 7.0])
    assert np.allclose(results["lightcurve"][0]["flux_perfect"], [1.0, 3.0])
    assert np.array_equal(results["lightcurve"][0]["filter"], ["g", "r"])
    assert np.array_equal(results["lightcurve"][0]["obs_idx"], [7, 6])
    assert np.array_equal(results["lightcurve"][0]["survey_idx"], [0, 0])


def test_simulate_bandfluxes_dups(test_data_dir):
    """Test that we warn when duplicate observation times are found."""
    # Create a toy observation table with two pointings.
    num_obs = 6
    obsdata = {
        "time": [0.0, 1.0, 1.0, 4.0, 5.0, 6.0],
        "ra": [0.0, 0.0, 0.0, 180.0, 0.0, 180.0],
        "dec": [10.0, 10.0, 10.0, -10.0, 10.0, -10.0],
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

    # Check that we have a warning for duplicate observation times.
    model = StaticBandfluxModel({"g": 1.0, "i": 2.0, "r": 3.0, "y": 4.0, "z": 5.0}, ra=0.0, dec=10.0)
    survey_info = SurveyInfo(obstable=obstable, passbands=passband_group)
    with pytest.warns(UserWarning):
        results = simulate_lightcurves(model, 1, survey_info, progress_bar=False)
    assert np.allclose(results["lightcurve"][0]["mjd"], [0.0, 1.0, 1.0, 5.0])
    assert np.array_equal(results["lightcurve"][0]["obs_idx"], [0, 1, 2, 4])


def test_simulate_parallel_threads(test_data_dir):
    """Test an end to end run of simulating the light curves parallel with threads."""
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
    ra0 = opsim_db["ra"].to_numpy()[0]
    dec0 = opsim_db["dec"].to_numpy()[0]
    source = ConstantSEDModel(
        brightness=NumpyRandomFunc("uniform", low=100.0, high=500.0),
        t0=0.0,
        ra=NumpyRandomFunc("uniform", low=ra0 - 0.5, high=ra0 + 0.5),
        dec=NumpyRandomFunc("uniform", low=dec0 - 0.5, high=dec0 + 0.5),
        redshift=0.0,
        node_label="source",
    )

    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
            results = simulate_lightcurves(
                source,
                100,
                survey_info,
                obstable_save_cols=["observationId", "zp_nJy"],
                param_cols=["source.brightness"],
                executor=executor,
                batch_size=10,
                save_full_filter_names=True,
                progress_bar=False,  # Disable progress bar for testing
            )
    assert len(results) == 100
    assert np.all(results["nobs"].to_numpy() >= 1)
    assert np.all(results["ra"].to_numpy() >= ra0 - 0.5)
    assert np.all(results["ra"].to_numpy() <= ra0 + 0.5)
    assert np.all(results["dec"].to_numpy() >= dec0 - 0.5)
    assert np.all(results["dec"].to_numpy() <= dec0 + 0.5)

    for idx in range(100):
        num_obs = results["nobs"][idx]
        assert num_obs >= 1
        assert len(results["lightcurve"][idx]["flux"]) == num_obs

        # Check that we got the full filter names correct.
        for filter_name, full_name in zip(
            results["lightcurve"][idx]["filter"],
            results["lightcurve"][idx]["full_filter_name"],
            strict=False,
        ):
            assert full_name == f"LSST_{filter_name}"

    # Extract the graph state, rerun and check that we get the same pre-noise results.
    state = GraphState.from_list(results["params"].to_numpy())
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
            results2 = simulate_lightcurves(
                source,
                100,
                survey_info,
                obstable_save_cols=["observationId", "zp_nJy"],
                param_cols=["source.brightness"],
                executor=executor,
                batch_size=10,
                save_full_filter_names=True,
                progress_bar=False,  # Disable progress bar for testing
                graph_state=state,
            )
    assert np.allclose(results["ra"].to_numpy(), results2["ra"].to_numpy())
    assert np.allclose(results["dec"].to_numpy(), results2["dec"].to_numpy())
    assert np.allclose(
        results["lightcurve.flux_perfect"].to_numpy(),
        results2["lightcurve.flux_perfect"].to_numpy(),
    )
    # Different noise realizations
    assert not np.allclose(
        results["lightcurve.flux"].to_numpy(),
        results2["lightcurve.flux"].to_numpy(),
    )


def test_simulate_parallel_processes(test_data_dir):
    """Test an end to end run of simulating the light curves paralle with processes."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    ra0 = opsim_db["ra"].to_numpy()[0]
    dec0 = opsim_db["dec"].to_numpy()[0]
    num_samples = 5_000
    table_data = {
        "t0": np.arange(num_samples, dtype=float),
        "ra": np.random.uniform(ra0 - 0.5, ra0 + 0.5, size=num_samples),
        "dec": np.random.uniform(dec0 - 0.5, dec0 + 0.5, size=num_samples),
    }
    table_sampler = TableSampler(table_data, in_order=True)

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    source = ConstantSEDModel(
        brightness=NumpyRandomFunc("uniform", low=100.0, high=500.0),
        t0=table_sampler.t0,
        ra=table_sampler.ra,
        dec=table_sampler.dec,
        redshift=0.0,
        node_label="source",
    )

    with ProcessPoolExecutor(max_workers=2) as executor:
        with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
            results = simulate_lightcurves(
                source,
                500,
                survey_info,
                obstable_save_cols=["observationId", "zp_nJy"],
                param_cols=["source.brightness"],
                executor=executor,
                batch_size=10,
                progress_bar=False,  # Disable progress bar for testing
            )
    assert len(results) == 500
    assert np.all(results["nobs"].to_numpy() >= 1)
    assert np.all(results["ra"].to_numpy() >= ra0 - 0.5)
    assert np.all(results["ra"].to_numpy() <= ra0 + 0.5)
    assert np.all(results["dec"].to_numpy() >= dec0 - 0.5)
    assert np.all(results["dec"].to_numpy() <= dec0 + 0.5)

    # Make sure that we get different parameter values across the processes.
    assert np.unique(results["ra"].to_numpy()).size > 475
    assert np.unique(results["dec"].to_numpy()).size > 475
    assert np.unique(results["source_brightness"].to_numpy()).size > 475

    # Check that we did not duplicate any t0 values even though they came
    # from a TableSampler. We should get each value [0, 499] exactly once.
    assert np.unique(results["t0"].to_numpy()).size == 500
    assert np.all(results["t0"].to_numpy() >= 0)
    assert np.all(results["t0"].to_numpy() < 500)

    # We can use the default (ProcessPoolExecutor) by giving a number of jobs.
    results2 = simulate_lightcurves(
        source,
        100,
        survey_info,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
        batch_size=10,
        num_jobs=2,
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results2) == 100

    # We can write the results to files.
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "test_df.parquet"
        ind_paths = []
        for i in range(4):
            curr_path = Path(tmpdir) / f"test_df_part{i}.parquet"
            assert not curr_path.exists()
            ind_paths.append(curr_path)

        results3 = simulate_lightcurves(
            source,
            200,
            survey_info,
            obstable_save_cols=["observationId", "zp_nJy"],
            param_cols=["source.brightness"],
            output_file_path=base_path,
            batch_size=50,
            num_jobs=4,
            progress_bar=False,  # Disable progress bar for testing
        )
        assert len(results3) == 4
        for i in range(4):
            assert str(results3[i]) == str(ind_paths[i])
            assert ind_paths[i].exists()

            results_df = read_parquet(ind_paths[i])
            assert len(results_df) == 50


def test_simulate_parallel_processes_fixed_params(test_data_dir):
    """Test an end to end run of simulating the light curves in parallel with processes
    when the model uses a FunctionNode with fixed_params."""
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    # Clip the sampled brightness to [100, 500] using fixed clipping bounds.
    brightness = FunctionNode(
        np.clip,
        a=NumpyRandomFunc("uniform", low=0.0, high=1000.0),
        fixed_params={"a_min": 100.0, "a_max": 500.0},
    )
    source = ConstantSEDModel(
        brightness=brightness,
        t0=0.0,
        ra=opsim_db["ra"].to_numpy()[0],
        dec=opsim_db["dec"].to_numpy()[0],
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        20,
        survey_info,
        param_cols=["source.brightness"],
        batch_size=10,
        num_jobs=2,
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results) == 20
    assert np.all(results["source_brightness"].to_numpy() >= 100.0)
    assert np.all(results["source_brightness"].to_numpy() <= 500.0)


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
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSEDModel(
        brightness=GivenValueList(given_brightness),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].to_numpy()[0:5]),
        dec=GivenValueList(opsim_db["dec"].to_numpy()[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        1,
        survey_info,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results) == 1

    # Check that we saved and can reassemble the GraphStates
    assert "params" in results
    state = GraphState.from_list(results["params"].to_numpy())
    assert state.num_samples == 1
    assert state["source.ra"] == opsim_db["ra"].to_numpy()[0]
    assert state["source.dec"] == opsim_db["dec"].to_numpy()[0]


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

    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    source = ConstantSEDModel(
        brightness=1000.0,
        t0=GivenValueList([20.0, 15.0, 20.0, 15.0, 20.0, 15.0, 20.0, 15.0]),
        ra=15.0,
        dec=10.0,
        redshift=1.5,
        node_label="source",
    )

    # Define the time window in the observer frame as (-5, 10) days from t0.
    results1 = simulate_lightcurves(
        source,
        2,
        survey_info,
        obs_time_window_offset=(-5.0, 10.0),
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results1) == 2

    # We should simulate the observations that are only within the time window, (15.0, 30.0) for the
    # first samples and (10.0, 25.0) for the second sample, and at the matching RA/Dec (the even indices).
    assert np.array_equal(
        results1["lightcurve"][0]["mjd"],
        np.array([16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]),
    )
    assert np.array_equal(
        results1["lightcurve"][1]["mjd"],
        np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]),
    )

    # Define the time window in the rest frame as (5, 15) days from t0.
    results2 = simulate_lightcurves(
        source,
        2,
        survey_info,
        rest_time_window_offset=(-5.0, 5.0),
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results2) == 2

    # We should simulate the observations that are only within the time window:
    # First sample (t0=20.0, z=1.5) =>
    # (20.0 - 5.0 * (1.0 + 1.5), 20 + 5.0 * (1.0 + 1.5)) = (7.5, 32.5)
    # Second sample (t0=15.0, z=1.5) =>
    # (15.0 - 5.0 * (1.0 + 1.5), 15 + 5.0 * (1.0 + 1.5)) = (2.5, 27.5)
    # We only sample evens because of the RA/Dec match.
    assert np.array_equal(results2["lightcurve"][0]["mjd"].to_numpy(), np.arange(8.0, 33.0, 2.0))
    assert np.array_equal(results2["lightcurve"][1]["mjd"].to_numpy(), np.arange(4.0, 28.0, 2.0))

    # Everything still works if t0 is a float.
    results3 = simulate_lightcurves(
        source,
        1,
        survey_info,
        rest_time_window_offset=(-5.0, 10.0),
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results3) == 1


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
    survey_info1 = SurveyInfo(obstable=obstable1, passbands=passband_group1)

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
    survey_info2 = SurveyInfo(obstable=obstable2, passbands=passband_group2)

    # Create a constant SED model with known brightnesses and RA, dec values that
    # match the (0.0, 10.0) pointing.
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=10.0, redshift=0.0, node_label="source")
    results = simulate_lightcurves(
        model,
        1,
        [survey_info1, survey_info2],
        obstable_save_cols=["zp", "custom_col"],
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results) == 1
    assert results["nobs"][0] == 4
    assert "lightcurve" in results
    assert "spectra" not in results

    # Check that the light curve was simulated correctly, including saving the zeropoint information
    # from each ObsTable. All entries for the light curve should be sorted by time.
    lightcurve = results["lightcurve"][0]
    assert np.allclose(lightcurve["mjd"], np.array([0.0, 0.5, 1.0, 2.5]))
    assert np.allclose(lightcurve["zp"], np.array([0.4, 0.05, 0.5, 0.2]))
    assert np.array_equal(lightcurve["filter"], np.array(["g", "r", "r", "r"]))
    assert np.array_equal(lightcurve["survey_idx"], np.array([0, 1, 0, 1]))

    # The custom column should only exist for observations from one of the surveys
    # and should be NaN for the other survey.
    assert lightcurve["custom_col"][0] == 1
    assert np.isnan(lightcurve["custom_col"][1])
    assert lightcurve["custom_col"][2] == 1
    assert np.isnan(lightcurve["custom_col"][3])


def test_simulate_multiple_surveys_diff_filters():
    """Test an end to end run of simulating a single object's light curve
    from multiple surveys with different filters (using the same name)."""
    # The first survey points at a single location on the sky in the "r" band.
    # It's filter overlaps the SED of the object.
    obsdata1 = {
        "time": [0.0, 1.0, 2.0],
        "ra": [0.0, 0.0, 0.0],
        "dec": [10.0, 10.0, 10.0],
        "filter": ["r", "r", "r"],
        "zp": [0.5, 0.5, 0.5],
        "seeing": [1.12, 1.12, 1.12],
        "skybrightness": [20.0, 20.0, 20.0],
        "exptime": [29.2, 29.2, 29.2],
        "nexposure": [2, 2, 2],
    }
    obstable1 = OpSim(obsdata1)

    waves = np.arange(5000, 8000, 100)
    transmission1 = np.where((waves >= 6000) & (waves <= 7000), 1.0, 0.0)
    r_passband1 = Passband(np.column_stack((waves, transmission1)), "survey1", "r")
    passband_group1 = PassbandGroup([r_passband1])
    survey_info1 = SurveyInfo(obstable=obstable1, passbands=passband_group1)

    # The second survey points at the same location at different times. But uses
    # a shifted passband for the "r" filter that only partially overlaps the SED.
    obsdata2 = {
        "time": [3.0, 4.0, 5.0],
        "ra": [0.0, 0.0, 0.0],
        "dec": [10.0, 10.0, 10.0],
        "filter": ["r", "r", "r"],
        "zp": [0.5, 0.5, 0.5],
        "seeing": [1.12, 1.12, 1.12],
        "skybrightness": [20.0, 20.0, 20.0],
        "exptime": [29.2, 29.2, 29.2],
        "nexposure": [2, 2, 2],
    }
    obstable2 = OpSim(obsdata2)
    transmission2 = np.where((waves >= 5500) & (waves <= 6500), 1.0, 0.0)
    r_passband2 = Passband(np.column_stack((waves, transmission2)), "survey2", "r")
    passband_group2 = PassbandGroup([r_passband2])
    survey_info2 = SurveyInfo(obstable=obstable2, passbands=passband_group2)

    # Create an SED model that is a step function over part of the range.
    class _ToyRedSEDModel(SEDModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def compute_sed(self, times, wavelengths, graph_state, **kwargs):
            """Draw effect-free observations for this object.

            Parameters
            ----------
            times : numpy.ndarray
                A length T array of rest frame timestamps.
            wavelengths : numpy.ndarray, optional
                A length N array of wavelengths (in angstroms).
            graph_state : GraphState
                An object mapping graph parameters to their values.
            **kwargs : dict, optional
                Any additional keyword arguments.

            Returns
            -------
            flux_density : numpy.ndarray
                A length T x N matrix of SED values (in nJy).
            """
            results = np.full((len(times), len(wavelengths)), 0.0)
            red_mask = (wavelengths >= 6000.0) & (wavelengths <= 7000.0)
            results[:, red_mask] = 1000.0
            return results

    # Create a constant SED model with known brightnesses and RA, dec values that
    # match the (0.0, 10.0) pointing.
    model = _ToyRedSEDModel(t0=0.0, ra=0.0, dec=10.0, redshift=0.0)
    results = simulate_lightcurves(
        model,
        1,
        [survey_info1, survey_info2],
        obstable_save_cols=["zp", "custom_col"],
        save_full_filter_names=True,
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results) == 1
    assert results["nobs"][0] == 6

    # The first three observations should be brighter than the last three.
    lightcurve = results["lightcurve"][0]
    assert np.all(lightcurve["flux_perfect"][0:3] > 800.0)
    assert np.all(lightcurve["flux_perfect"][3:6] < 500.0)
    assert np.array_equal(lightcurve["filter"], ["r"] * 6)
    assert np.array_equal(
        lightcurve["full_filter_name"],
        ["survey1_r", "survey1_r", "survey1_r", "survey2_r", "survey2_r", "survey2_r"],
    )


def test_simulate_multiple_surveys_spectra(test_data_dir):
    """Test an end to end run of simulating a single light curve from multiple surveys
    where one of the instruments is a spectrograph."""
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
    survey_info1 = SurveyInfo(obstable=obstable1, passbands=passband_group1)

    # The second survey points at two locations on the sky and uses a spectrograph.
    obsdata2 = {
        "time": [0.5, 1.5, 2.5, 3.5],
        "ra": [0.0, 90.0, 0.0, 90.0],
        "dec": [10.0, -10.0, 10.0, -10.0],
        "filter": ["spectra", "spectra", "spectra", "spectra"],
        "zp": [0.05, 0.1, 0.2, 0.3],
        "seeing": [1.12, 1.12, 1.12, 1.12],
        "skybrightness": [20.0, 20.0, 20.0, 20.0],
        "exptime": [29.2, 29.2, 29.2, 29.2],
        "nexposure": [2, 2, 2, 2],
    }
    obstable2 = OpSim(obsdata2)
    spectrograph = Spectrograph.from_regular_grid(2000.0, 10000.0, 100.0, instrument="survey2")
    survey_info2 = SurveyInfo(
        obstable=obstable2,
        passbands=spectrograph,
        noise_model=ConstantSpectrographNoiseModel(1.0),
    )

    # The third survey points at a single location on the sky and uses a spectrograph.
    obsdata3 = {
        "time": [1.1],
        "ra": [0.0],
        "dec": [10.0],
    }
    obstable3 = SpectrographObsTable(obsdata3)
    spectrograph2 = Spectrograph.from_regular_grid(3000.0, 9000.0, 100.0, instrument="survey3")
    survey_info3 = SurveyInfo(
        obstable=obstable3,
        passbands=spectrograph2,
        noise_model=None,  # No noise model.
    )

    # Create a constant SED model with known brightnesses and RA, dec values that
    # match the (0.0, 10.0) pointing.
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=10.0, redshift=0.0, node_label="source")
    results = simulate_lightcurves(
        model,
        1,
        [survey_info1, survey_info2, survey_info3],
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results) == 1
    assert results["nobs"][0] == 5
    assert "lightcurve" in results
    assert "spectra" in results

    # Check that the light curve was simulated correctly, including saving the zeropoint information
    # from each ObsTable.
    lightcurve = results["lightcurve"][0]
    assert np.allclose(lightcurve["mjd"], np.array([0.0, 1.0]))
    assert np.array_equal(lightcurve["filter"], np.array(["g", "r"]))
    assert np.array_equal(lightcurve["survey_idx"], np.array([0, 0]))

    # Check that the spectra were simulated correctly. Waves and measured fluxes should have
    # an array per-nested entry. The measured_flux_perfect should all be around 100 (since we used
    # a constant SED model with brightness=100.0) and measured_flux_error should all be 1.0.
    spectra = results.iloc[0]["spectra"]
    assert np.allclose(spectra["mjd"], np.array([0.5, 2.5, 1.1]))
    assert np.shape(spectra.iloc[0]["waves"]) == (80,)
    assert np.shape(spectra.iloc[1]["waves"]) == (80,)
    assert np.shape(spectra.iloc[2]["waves"]) == (60,)
    assert np.shape(spectra.iloc[0]["measured_flux"]) == (80,)
    assert np.shape(spectra.iloc[1]["measured_flux"]) == (80,)
    assert np.shape(spectra.iloc[2]["measured_flux"]) == (60,)

    expected_spectro_0 = (
        fnu_to_flam(
            np.array([100.0] * 80),
            spectrograph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spectrograph.bin_widths
    )
    assert np.allclose(spectra.iloc[0]["measured_flux_perfect"], expected_spectro_0)

    expected_spectro_1 = (
        fnu_to_flam(
            np.array([100.0] * 80),
            spectrograph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spectrograph.bin_widths
    )
    assert np.allclose(spectra.iloc[1]["measured_flux_perfect"], expected_spectro_1)

    expected_spectro_2 = (
        fnu_to_flam(
            np.array([100.0] * 60),
            spectrograph2.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spectrograph2.bin_widths
    )
    assert np.allclose(spectra.iloc[2]["measured_flux_perfect"], expected_spectro_2)

    assert np.allclose(spectra.iloc[0]["measured_flux_error"], np.ones(80))
    assert np.allclose(spectra.iloc[1]["measured_flux_error"], np.ones(80))
    assert np.allclose(spectra.iloc[2]["measured_flux_error"], np.zeros(60))
    assert not np.allclose(spectra.iloc[0]["measured_flux"], spectra.iloc[0]["measured_flux_perfect"])
    assert not np.allclose(spectra.iloc[1]["measured_flux"], spectra.iloc[1]["measured_flux_perfect"])
    assert np.allclose(spectra.iloc[2]["measured_flux"], spectra.iloc[2]["measured_flux_perfect"])
    assert np.array_equal(spectra["instrument"], ["survey2", "survey2", "survey3"])

    # Check that we can save the full results to a file.
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "test_spectra_results.parquet"
        results.to_parquet(base_path)
        assert base_path.exists()


def test_simulate_drop_empty_rows(test_data_dir):
    """Test that we drop empty rows when configured to do so."""
    # Set up a small survey with observations at two different regions of the sky.
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
    }
    obstable1 = OpSim(obsdata1)
    passband_group1 = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r"],
    )
    survey_info = SurveyInfo(obstable=obstable1, passbands=passband_group1)

    # Create a constant SED model that is sometimes observed:
    # [seen, not seen, not seen, seen, seen, not seen]
    all_ra = GivenValueList([0.0, 0.0, 90.0, 180.0, 180.1, 270.0])
    all_dec = GivenValueList([10.0, -10.0, 0.0, -10.0, -9.9, -10.0])
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=all_ra, dec=all_dec, redshift=0.0)

    # Do the simulation with defaults. We should get 6 result rows.
    results = simulate_lightcurves(model, 6, [survey_info], progress_bar=False)
    assert len(results) == 6

    # Reset the lists and do the simulation with drop_empty_rows=True.
    all_ra.reset()
    all_dec.reset()
    results = simulate_lightcurves(model, 6, [survey_info], drop_empty_rows=True, progress_bar=False)
    assert len(results) == 3
    assert results["id"].tolist() == [0, 3, 4]


def test_compute_noise_free_lightcurves_single(test_data_dir):
    """Test computing noise-free light curves for a set of times and filters."""
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
    """Test computing noise-free light curves for a set of times and filters
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


def test_saturation_mags_initialization(test_data_dir):
    """Check initialization of saturation thresholds."""
    # A FakeObsTable should have no saturation thresholds by default.
    fake_obs = FakeObsTable(
        pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "ra": [0.0, 0.0],
                "dec": [0.0, 0.0],
                "filter": ["g", "r"],
            }
        ),
        bandflux_error={"g": 26.0, "r": 27.0},
        exptime=30.0,
        fwhm_px={"g": 2.5, "r": 3.1},
        nexposure=2,
        radius=100.0,
        read_noise=5.0,
        sky_bg_electrons={"g": 150.0, "r": 140.0},
        survey_name="none",
    )
    assert fake_obs._saturation_mags is None

    # Opsim saturation thresholds should be set by default.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")
    assert opsim_db._saturation_mags is not None
    assert isinstance(opsim_db._saturation_mags, dict)
    for filter in ["u", "g", "r", "i", "z", "y"]:
        assert filter in opsim_db._saturation_mags

    # For now, the ZTF table uses a single estimate for all filters.
    ztf_table_data = create_random_ztf_obs_data(100)
    ztf_obs_table = ZTFObsTable(table=ztf_table_data)
    assert ztf_obs_table._saturation_mags is not None
    assert isinstance(ztf_obs_table._saturation_mags, dict)
    for filter in ["g", "r", "i"]:
        assert filter in ztf_obs_table._saturation_mags


def test_simulate_with_saturation_mags_as_none(test_data_dir):
    """Test an end to end run of simulating a single light curve with no saturation thresholds."""
    # Set up obs table (with no saturation thresholds set)
    obs_table_values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(obs_table_values)
    bandflux_error = {"g": 26.0, "r": 27.0, "i": 28.0}

    ops_data = FakeObsTable(
        pdf,
        bandflux_error=bandflux_error,
        exptime=60.0,
        fwhm_px={"g": 2.5, "r": 3.1, "i": 1.9},
        nexposure=200,
        radius=100.0,
        read_noise=5.0,
        sky_bg_electrons={"g": 150.0, "r": 140.0, "i": 155.0},
        survey_name="none",
    )
    assert ops_data._saturation_mags is None

    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )
    survey_info = SurveyInfo(obstable=ops_data, passbands=passband_group)

    # Set up model
    source = ConstantSEDModel(
        brightness=50_000.0,
        t0=0.0,
        ra=GivenValueList(obs_table_values["ra"]),
        dec=GivenValueList(obs_table_values["dec"]),
        redshift=0.0,
        node_label="source",
    )

    # Simulate
    results = simulate_lightcurves(
        source,
        1,
        survey_info,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
    )
    assert len(results) == 1

    # Check that the "is_saturated" column is False for all observations.
    lightcurve = results["lightcurve"][0]
    assert "is_saturated" in lightcurve.columns
    assert not np.any(lightcurve["is_saturated"])


def test_simulate_with_default_saturation_mags_values(test_data_dir):
    """Test an end to end run of simulating a single light curve with default saturation thresholds."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the g and r filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r"],
    )
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    # Create a constant SED model with known RA and dec values that match the opsim.
    # Note the brightness is set very high to ensure saturation threshold is surpassed.
    source = ConstantSEDModel(
        brightness=(2.0e12),  # Sufficiently bright to ensure saturation
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].to_numpy()[0:5]),
        dec=GivenValueList(opsim_db["dec"].to_numpy()[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        1,
        survey_info,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
    )
    assert len(results) == 1

    # Check that every flux value is below the saturation threshold for its filter.
    lightcurve = results["lightcurve"][0]
    opsim_sat_thresholds_njy = {band: mag2flux(mag) for band, mag in opsim_db._saturation_mags.items()}

    for filter in ["g", "r"]:
        mask = lightcurve["filter"] == filter
        fluxes = lightcurve["flux"][mask]
        assert np.all(fluxes <= opsim_sat_thresholds_njy[filter])

    # Check the flux_error values are non-zero.
    flux_errors = lightcurve["fluxerr"]
    assert np.all(flux_errors > 0.0)

    # Check that the "is_saturated" column is True for all observations.
    assert "is_saturated" in lightcurve.columns
    assert np.all(lightcurve["is_saturated"])


def test_simulate_with_custom_saturation_mags(test_data_dir):
    """Test an end to end run of simulating a single light curve with custom saturation thresholds."""
    # Set up obs table
    obs_table_values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(obs_table_values)
    bandflux_error = {"g": 26.0, "r": 27.0, "i": 28.0}

    # Saturation thresholds are given in magnitudes.
    toy_sat_thresholds = {
        "g": 17.5,
        "r": 18.0,
        "i": 17.0,
        "z": 17.2,
    }

    ops_data = FakeObsTable(
        pdf,
        bandflux_error=bandflux_error,
        exptime=60.0,
        fwhm_px={"g": 2.5, "r": 3.1, "i": 1.9},
        nexposure=200,
        radius=100.0,
        read_noise=5.0,
        sky_bg_electrons={"g": 150.0, "r": 140.0, "i": 155.0},
        survey_name="none",
        saturation_mags=toy_sat_thresholds,
    )

    # Set up model
    source = ConstantSEDModel(
        brightness=5_000_000_000_000.0,  # Sufficiently bright to ensure saturation
        t0=0.0,
        ra=GivenValueList(obs_table_values["ra"]),
        dec=GivenValueList(obs_table_values["dec"]),
        redshift=0.0,
        node_label="source",
    )

    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )
    survey_info = SurveyInfo(obstable=ops_data, passbands=passband_group)

    # Simulate
    results = simulate_lightcurves(
        source,
        1,
        survey_info,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
    )
    assert len(results) == 1

    # Check that every flux value is below the saturation threshold for its filter.
    lightcurve = results["lightcurve"][0]
    toy_sat_thresholds_njy = {band: mag2flux(mag) for band, mag in toy_sat_thresholds.items()}
    for filter in ["g", "r", "i", "z"]:
        mask = lightcurve["filter"] == filter
        fluxes = lightcurve["flux"][mask]
        assert np.all(fluxes <= toy_sat_thresholds_njy[filter])

    # Check the flux_error values are non-zero.
    flux_errors = lightcurve["fluxerr"]
    fluxes = lightcurve["flux"]
    assert np.all(flux_errors > 0.0)

    # Check that the "is_saturated" column is True for all observations.
    assert "is_saturated" in lightcurve.columns
    assert np.all(lightcurve["is_saturated"])


class _DuplicatingConstantSEDModel(ConstantSEDModel):
    """A model with a constant SED over both wavelength and time that repeats entries.

    Parameters
    ----------
    brightness : parameter
        The inherent brightness.
    repeats : parameter
        The number of times to duplicate the observations.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, brightness, repeats, **kwargs):
        super().__init__(brightness, **kwargs)
        self.add_parameter(
            "number_of_duplicates",
            StateExpansionNode(repeats=repeats).repeats,
        )


def test_simulate_expanded_state(test_data_dir):
    """Test an end to end run with state expansion nodes."""
    # Load the OpSim data and passband data for the griz filters only.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )
    survey_info = SurveyInfo(obstable=opsim_db, passbands=passband_group)

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim. Duplicate the observations per row.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    given_repeats = [1, 2, 1, 3, 1]
    source = _DuplicatingConstantSEDModel(
        brightness=GivenValueList(given_brightness),
        repeats=GivenValueList(given_repeats),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].to_numpy()[0:5]),
        dec=GivenValueList(opsim_db["dec"].to_numpy()[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        5,
        survey_info,
        param_cols=["source.brightness"],
        progress_bar=False,  # Disable progress bar for testing
    )
    assert len(results) == 8
    assert np.all(results["nobs"].to_numpy() >= 1)
    assert np.allclose(results["z"].to_numpy(), 0.0)
    assert np.allclose(results["t0"].to_numpy(), 0.0)

    # Check that we correctly repeated values.
    assert np.allclose(results["ra"].to_numpy()[0], opsim_db["ra"].to_numpy()[0])
    assert np.allclose(results["ra"].to_numpy()[1:3], opsim_db["ra"].to_numpy()[1])
    assert np.allclose(results["ra"].to_numpy()[3], opsim_db["ra"].to_numpy()[2])
    assert np.allclose(results["ra"].to_numpy()[4:7], opsim_db["ra"].to_numpy()[3])
    assert np.allclose(results["ra"].to_numpy()[7], opsim_db["ra"].to_numpy()[4])
    assert np.allclose(results["dec"].to_numpy()[0], opsim_db["dec"].to_numpy()[0])
    assert np.allclose(results["dec"].to_numpy()[1:3], opsim_db["dec"].to_numpy()[1])
    assert np.allclose(results["dec"].to_numpy()[3], opsim_db["dec"].to_numpy()[2])
    assert np.allclose(results["dec"].to_numpy()[4:7], opsim_db["dec"].to_numpy()[3])
    assert np.allclose(results["dec"].to_numpy()[7], opsim_db["dec"].to_numpy()[4])
    assert np.allclose(results["source_brightness"].to_numpy()[0], given_brightness[0])
    assert np.allclose(results["source_brightness"].to_numpy()[1:3], given_brightness[1])
    assert np.allclose(results["source_brightness"].to_numpy()[3], given_brightness[2])
    assert np.allclose(results["source_brightness"].to_numpy()[4:7], given_brightness[3])
    assert np.allclose(results["source_brightness"].to_numpy()[7], given_brightness[4])


def test_simulate_save_nested_with_parameter_arrays():
    """Test that we can save non-scalar parameter values in the NestedFrame results
    and write them out to parquet."""
    num_samples = 10

    # Generate random array values uniformly between 0.01 and 100.0. Then
    # take the logarithm of those values.
    np_node = NumpyRandomFunc("uniform", low=0.01, high=100.0, seed=100, size=5)
    math_node = BasicMathNode("log(x)", x=np_node, node_label="math")
    sample_states = math_node.sample_parameters(num_samples=num_samples)

    # Create a nest results frame including the array values.
    index = list(range(num_samples))
    results_dict = {
        "id": index,
        "ra": np.full(num_samples, 15.0),
        "dec": np.full(num_samples, -10.0),
        "params": [state.to_dict() for state in sample_states],
    }
    nested_dict = {
        "mjd": np.full(num_samples * 2, 59000.0),
    }

    nested_index = [i % num_samples for i in range(2 * num_samples)]
    results = NestedFrame(data=results_dict, index=index)
    nested_frame = pd.DataFrame(data=nested_dict, index=nested_index)
    results = results.join_nested(nested_frame, "lightcurve")

    # Each entry in params["math.function_node_result"] contains a length 5 array.
    assert len(results["params"][0]["math.function_node_result"]) == 5

    # We can write the results to files.
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "test_array_df.parquet"
        results.to_parquet(base_path)
        assert base_path.exists()


class PicklingMapExecutor:
    """A fake executor whose ``map`` round-trips every item and result through pickle.

    It runs in this process, so it sees this process's preload slot, but like a process
    pool it never shares objects with the caller. ``sizes`` records the pickled size (in
    bytes) of every item sent.
    """

    def __init__(self):
        self.sizes = []

    def map(self, fn, items):
        """Apply ``fn`` to a pickled copy of each item and return pickled copies of the results."""
        results = []
        for item in items:
            payload = pickle.dumps(item)
            self.sizes.append(len(payload))
            results.append(pickle.loads(pickle.dumps(fn(pickle.loads(payload)))))
        return results


class InProcessMapExecutor:
    """A fake executor whose ``map`` calls ``fn`` on each item directly, without copying."""

    def map(self, fn, items):
        """Apply ``fn`` to each item in order and return the results."""
        return [fn(item) for item in items]


def _run(model, si, executor=None, **kw):
    """Run ``simulate_lightcurves`` with the settings shared by the preload tests."""
    kwargs = {
        "num_samples": 60,
        "batch_size": 10,
        "rng": np.random.default_rng(100),
        "obstable_save_cols": ["observationId", "zp_nJy"],
        "param_cols": ["source.brightness"],
        "progress_bar": False,
    }
    kwargs.update(kw)
    return simulate_lightcurves(model, survey_info=si, executor=executor, **kwargs)


def _stateful_model(si):
    """Return a model inside the survey footprint whose stateful GivenValueList refuses pickling."""
    return ConstantSEDModel(
        brightness=GivenValueList(np.linspace(100.0, 500.0, 100)),
        t0=0.0,
        ra=si.obstable["ra"][0],
        dec=si.obstable["dec"][0],
        redshift=0.0,
        node_label="source",
    )


def test_simulate_preloaded_matches_non_preloaded(preload_inputs, assert_identical_results):
    """A preloaded run gives identical results to a non-preloaded one, with much smaller batches."""
    model, si = preload_inputs
    base_ex = PicklingMapExecutor()
    with pytest.warns(WorkerPreloadWarning, match="not preloaded") as record:
        expected = _run(model, si, base_ex)
    assert record[0].filename == __file__  # The warning points at the simulate_lightcurves call.

    init = make_worker_initializer(model, si)
    init()
    ex = PicklingMapExecutor()
    with warnings.catch_warnings():
        warnings.simplefilter("error", WorkerPreloadWarning)
        result = _run(model, si, ex)
    assert_identical_results(expected, result)
    assert len(ex.sizes) == 6
    assert max(ex.sizes) * 10 < min(base_ex.sizes)


def test_simulate_preloaded_batch_frees_its_model(preload_inputs, monkeypatch):
    """Each preloaded batch frees its own copy of the model before it returns.

    The model's nodes reference each other in cycles, so only the cyclic garbage collector
    can free a copy. Automatic collection is disabled here, so any copy that is freed was
    freed by the batch itself.
    """
    model, si = preload_inputs
    loaded_models = []

    def recording_load(digest):
        loaded = load_preloaded_data(digest)
        if loaded is not None:
            loaded_models.append(weakref.ref(loaded[0]))
        return loaded

    # simulate.py imports load_preloaded_data by name, so patch it there.
    monkeypatch.setattr(simulate_module, "load_preloaded_data", recording_load)
    make_worker_initializer(model, si)()

    gc.disable()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", WorkerPreloadWarning)
            result = _run(model, si, PicklingMapExecutor())
        # Check before re-enabling automatic collection, which would free any leftover copies.
        alive = [ref() is not None for ref in loaded_models]
    finally:
        gc.enable()
    assert len(result) == 60
    assert alive == [False] * 6


def test_simulate_preloaded_thread_pool(preload_inputs, assert_identical_results):
    """A preloaded ThreadPoolExecutor gives identical results without falling back."""
    model, si = preload_inputs
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        expected = _run(model, si, PicklingMapExecutor())

    with ThreadPoolExecutor(4, initializer=make_worker_initializer(model, si)) as executor:
        with warnings.catch_warnings():
            warnings.simplefilter("error", WorkerPreloadWarning)
            result = _run(model, si, executor)
    assert_identical_results(expected, result)


def test_simulate_not_preloaded_skips_digest(preload_inputs, monkeypatch):
    """With nothing preloaded in this process, the run falls back without computing a digest."""
    # simulate.py imports compute_preload_digest by name, so patch it there.
    monkeypatch.setattr(
        simulate_module,
        "compute_preload_digest",
        lambda *args, **kwargs: pytest.fail("digest must not be computed"),
    )
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        assert len(_run(*preload_inputs, PicklingMapExecutor())) == 60


def test_simulate_preload_changed_model_falls_back(preload_inputs, assert_identical_results):
    """A model changed after preloading falls back, and the results use the changed model."""
    model, si = preload_inputs
    init = make_worker_initializer(model, si)
    init()
    model.set_parameter("redshift", 0.05)
    with pytest.warns(WorkerPreloadWarning, match="do not match"):
        result = _run(model, si, PicklingMapExecutor())
    assert np.all(result["z"].to_numpy() == 0.05)

    reset_preload_state()
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        expected = _run(model, si, PicklingMapExecutor())
    assert_identical_results(expected, result)


def test_simulate_preload_changed_obstable_falls_back(preload_inputs, assert_identical_results):
    """An ObsTable column changed in place after preloading falls back to the current data."""
    model, si = preload_inputs
    init = make_worker_initializer(model, si)
    init()
    # OpSim renames the raw "zp_nJy" column to the standard name "zp" on load.
    si.obstable._table.loc[0, "zp"] *= 2.0
    with pytest.warns(WorkerPreloadWarning, match="do not match"):
        result = _run(model, si, PicklingMapExecutor())
    # The changed observation is used, so stale data would change the results.
    assert (result["lightcurve"].nest.to_flat()["obs_idx"] == 0).any()

    reset_preload_state()
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        expected = _run(model, si, PicklingMapExecutor())
    assert_identical_results(expected, result)


def test_simulate_preload_miss_resends(preload_inputs, assert_identical_results):
    """Batches that reach workers without this preload are resent with the full data.

    InProcessMapExecutor does not copy the batches, so a light batch shares its rng with
    the full batch that is resent: identical results show that a miss leaves it untouched.
    """
    model, si = preload_inputs
    # An in-process run changes the model's internal state, so the baseline uses a copy.
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        expected = _run(copy.deepcopy(model), si, InProcessMapExecutor())

    # Case 1: this data is registered in this process, but the executor was never preloaded.
    make_worker_initializer(model, si)  # Not called, so this process's slot stays empty.
    with pytest.warns(WorkerPreloadWarning, match="6 of 6 batches reached workers") as record:
        result = _run(model, si, InProcessMapExecutor())
    assert record[0].filename == __file__  # The warning points at the simulate_lightcurves call.
    assert_identical_results(expected, result)

    # Case 2: the executor was preloaded with different data. The in-process resend above
    # changed the model's internal state (TableSampler._last_start_index), so register its
    # current digest again before filling the slot with another model's data.
    make_worker_initializer(model, si)
    other_model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0)
    make_worker_initializer(other_model, si)()
    with pytest.warns(WorkerPreloadWarning, match="6 of 6 batches reached workers"):
        result = _run(model, si, InProcessMapExecutor())
    assert_identical_results(expected, result)


def test_simulate_preload_unpicklable_falls_back(preload_inputs):
    """Data that refuses pickling falls back on a thread pool, and refuses preloading."""
    model, si = preload_inputs
    make_worker_initializer(model, si)  # Make the registry non-empty.
    stateful_model = _stateful_model(si)
    with ThreadPoolExecutor(2) as executor:
        with pytest.warns(WorkerPreloadWarning, match="could not be pickled"):
            result = _run(stateful_model, si, executor, num_samples=10, batch_size=5)
    assert len(result) == 10

    with pytest.raises(RuntimeError, match="cannot be pickled"):
        make_worker_initializer(stateful_model, si)


def test_simulate_num_jobs_preloads(preload_inputs, assert_identical_results):
    """num_jobs preloads its process pool, and data that refuses pickling still raises."""
    model, si = preload_inputs
    with ProcessPoolExecutor(2) as executor:
        with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
            expected = _run(model, si, executor)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WorkerPreloadWarning)
        result = _run(model, si, num_jobs=2)
    assert_identical_results(expected, result)

    with pytest.raises(RuntimeError, match="cannot be pickled"):
        _run(_stateful_model(si), si, num_jobs=2)


def test_simulate_preload_graph_state_replay(preload_inputs, assert_identical_results):
    """Replaying a GraphState through a preloaded executor matches the non-preloaded replay."""
    model, si = preload_inputs
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        first = _run(model, si, PicklingMapExecutor())
    state = GraphState.from_list(first["params"].to_numpy())

    # Replay with a different rng, so only the replayed state can reproduce the parameters.
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        expected = _run(model, si, PicklingMapExecutor(), graph_state=state, rng=np.random.default_rng(7))
    init = make_worker_initializer(model, si)
    init()
    with warnings.catch_warnings():
        warnings.simplefilter("error", WorkerPreloadWarning)
        result = _run(model, si, PicklingMapExecutor(), graph_state=state, rng=np.random.default_rng(7))
    assert_identical_results(expected, result)
    np.testing.assert_array_equal(
        result["source_brightness"].to_numpy(), first["source_brightness"].to_numpy()
    )


def test_simulate_preload_batch_error_propagates(preload_inputs):
    """An error raised inside a preloaded batch reaches the caller with the same type."""
    _, si = preload_inputs
    model = ConstantSEDModel(
        brightness=100.0,
        # Two columns, since a single-column TableSampler cannot sample several rows at once.
        t0=TableSampler({"t0": np.arange(15.0), "row": np.arange(15)}, in_order=True).t0,
        ra=si.obstable["ra"][0],
        dec=si.obstable["dec"][0],
        redshift=0.0,
        node_label="source",
    )
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        with pytest.raises(IndexError, match="ran out of entries"):
            _run(model, si, PicklingMapExecutor())

    init = make_worker_initializer(model, si)
    init()
    with warnings.catch_warnings():
        warnings.simplefilter("error", WorkerPreloadWarning)
        with pytest.raises(IndexError, match="ran out of entries"):
            _run(model, si, PicklingMapExecutor())


def test_simulate_preload_output_files(preload_inputs, tmp_path, assert_identical_results):
    """Preloaded runs that write files use the same _partN names and write the same data."""
    model, si = preload_inputs
    with pytest.warns(WorkerPreloadWarning, match="not preloaded"):
        expected = _run(model, si, PicklingMapExecutor(), output_file_path=tmp_path / "base" / "out.parquet")

    init = make_worker_initializer(model, si)
    init()
    with warnings.catch_warnings():
        warnings.simplefilter("error", WorkerPreloadWarning)
        result = _run(model, si, PicklingMapExecutor(), output_file_path=tmp_path / "pre" / "out.parquet")

    names = [f"out_part{idx}.parquet" for idx in range(6)]
    assert [Path(path).name for path in expected] == names
    assert [Path(path).name for path in result] == names
    for expected_path, result_path in zip(expected, result, strict=True):
        assert_identical_results(read_parquet(expected_path), read_parquet(result_path))


def test_simulate_preload_leaves_surveys_untouched(preload_inputs):
    """Serial and preloaded runs leave the survey data unchanged, including the preloaded
    copy that every batch in a worker shares."""
    model, si = preload_inputs

    def shared_digest(survey_info):
        return serialize_simulation_data(model, survey_info, keep_bytes=False).shared_digest

    before = shared_digest([si])
    _run(copy.deepcopy(model), si)  # A serial run, on a copy since it changes the model's state.
    assert shared_digest([si]) == before

    init = make_worker_initializer(model, si)
    init()
    _, preloaded_si = load_preloaded_data(init.digest)
    preloaded_before = shared_digest(preloaded_si)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WorkerPreloadWarning)
        _run(model, si, PicklingMapExecutor())
    assert shared_digest([si]) == before
    assert load_preloaded_data(init.digest)[1] is preloaded_si  # The batches used this copy.
    assert shared_digest(preloaded_si) == preloaded_before
