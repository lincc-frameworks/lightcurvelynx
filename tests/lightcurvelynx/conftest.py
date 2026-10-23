from pathlib import Path

import pytest

from lightcurvelynx import _LIGHTCURVELYNX_TEST_DATA_DIR


@pytest.fixture
def test_data_dir():
    """Return the base test data directory."""
    return Path(_LIGHTCURVELYNX_TEST_DATA_DIR)


@pytest.fixture
def grid_data_good_file(test_data_dir):
    """Return the file path for the good grid input file."""
    return test_data_dir / "grid_input_good.ecsv"


@pytest.fixture
def grid_data_bad_file(test_data_dir):
    """Return the file path for the bad grid input file."""
    return test_data_dir / "grid_input_bad.txt"


@pytest.fixture
def opsim_small(test_data_dir):
    """Return the file path for the small opsim db."""
    return test_data_dir / "opsim_small.db"


@pytest.fixture
def opsim_shorten(test_data_dir):
    """Return the file path for the shortened opsim db."""
    return test_data_dir / "opsim_shorten.db"


@pytest.fixture
def test_flow_filename(test_data_dir):
    """Return the file path for the test flow input file."""
    return test_data_dir / "test_flow.pkl"


@pytest.fixture
def test_conditional_flow_filename(test_data_dir):
    """Return the file path for the test flow input file."""
    return test_data_dir / "test_conditional_flow.pkl"


@pytest.fixture
def oversampled_observations(opsim_shorten):
    """Return an OpSim object with 0.05 day cadence spanning year 2027."""
    from lightcurvelynx.obstable.opsim import OpSim, oversample_opsim

    base_opsim = OpSim.from_db(opsim_shorten)
    return oversample_opsim(
        base_opsim,
        pointing=(0.0, 0.0),
        search_radius=180.0,
        delta_t=0.05,
        time_range=(61406.0, 61771.0),
        bands=None,
        strategy="darkest_sky",
    )


@pytest.fixture
def passbands_dir(test_data_dir):
    """Return the file path for passbands directory."""
    return test_data_dir / "passbands"


@pytest.fixture(autouse=True)
def _reset_worker_preload():
    """Clear the worker-preload registry and slot before and after every test.

    Autouse so preload state never leaks between tests, since the registry and slot
    are module-level state in ``lightcurvelynx.utils.worker_preload``.
    """
    from lightcurvelynx.utils.worker_preload import reset_preload_state

    reset_preload_state()
    yield
    reset_preload_state()


@pytest.fixture
def preload_inputs(test_data_dir):
    """Build a small model and SurveyInfo pair used by the worker-preload tests.

    Returns
    -------
    tuple of (ConstantSEDModel, SurveyInfo)
        A model with randomized brightness, t0, ra and dec, and a SurveyInfo built
        from the small OpSim test data with the default LSST passbands and noise model.
    """
    import numpy as np

    from lightcurvelynx.astro_utils.passbands import PassbandGroup
    from lightcurvelynx.math_nodes.given_sampler import TableSampler
    from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
    from lightcurvelynx.models.basic_models import ConstantSEDModel
    from lightcurvelynx.obstable.opsim import OpSim
    from lightcurvelynx.survey_info import SurveyInfo

    opsim = OpSim.from_db(test_data_dir / "opsim_small.db")
    passbands = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )
    survey_info = SurveyInfo(obstable=opsim, passbands=passbands)

    ra0 = opsim["ra"][0]
    dec0 = opsim["dec"][0]
    # The unused "row" column is needed: a single-column TableSampler fails when sampling
    # more than one sample at a time (its one output is saved as a length-1 list).
    table = {"t0": np.arange(500.0), "row": np.arange(500)}
    model = ConstantSEDModel(
        brightness=NumpyRandomFunc("uniform", low=100.0, high=500.0, seed=11),
        t0=TableSampler(table, in_order=True, node_label="t0_table").t0,
        ra=NumpyRandomFunc("uniform", low=ra0 - 0.5, high=ra0 + 0.5, seed=12),
        dec=NumpyRandomFunc("uniform", low=dec0 - 0.5, high=dec0 + 0.5, seed=13),
        redshift=0.0,
        node_label="source",
    )
    return model, survey_info


@pytest.fixture
def assert_identical_results():
    """Return a function that asserts two simulation result frames are identical.

    Returns
    -------
    callable
        A function ``compare(a, b)`` that raises an ``AssertionError`` if the two
        ``NestedFrame`` results differ, checking both the flat columns and the
        nested "lightcurve" column.
    """
    import pandas as pd

    def compare(a, b):
        # Compare the flattened light curves first: when they differ, comparing the whole
        # NestedFrame raises an opaque KeyError from pandas instead of an AssertionError.
        pd.testing.assert_frame_equal(
            a["lightcurve"].nest.to_flat(),
            b["lightcurve"].nest.to_flat(),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(a, b, check_exact=True)

    return compare
