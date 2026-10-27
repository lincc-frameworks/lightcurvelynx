"""Tests for worker preloading: serialization, the initializer and its slot, and refusal."""

import pickle

import pytest

from lightcurvelynx.math_nodes.ra_dec_sampler import ObsTableUniformRADECSampler
from lightcurvelynx.models.basic_models import ConstantSEDModel
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.survey_info import SurveyInfo
from lightcurvelynx.utils import worker_preload
from lightcurvelynx.utils.worker_preload import (
    compute_preload_digest,
    has_known_preload_digests,
    is_known_preload_digest,
    load_preloaded_data,
    make_worker_initializer,
    serialize_simulation_data,
)


def test_serialize_is_deterministic_and_modes_agree(preload_inputs):
    """Keeping or dropping the bytes must not change the digest, and re-running must agree."""
    model, si = preload_inputs
    kept = serialize_simulation_data(model, [si], keep_bytes=True)
    hashed = serialize_simulation_data(model, [si], keep_bytes=False)
    assert kept.digest == hashed.digest == compute_preload_digest(model, [si])
    assert hashed.data is None
    assert serialize_simulation_data(model, [si], keep_bytes=True).data == kept.data


def test_serialize_none_components(test_data_dir):
    """None-valued survey components and model parameters survive the round trip as None."""
    # SurveyInfo(obstable=opsim, survey_name="none") has noise_model None and passbands None.
    # Model attributes that are None stay None after the round trip, and the survey's
    # noise_model and passbands load as None.
    opsim = OpSim.from_db(test_data_dir / "opsim_small.db")
    si = SurveyInfo(obstable=opsim, survey_name="none")
    assert si.noise_model is None
    assert si.passbands is None

    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0)
    assert model.setters["distance"].value is None

    init = make_worker_initializer(model, si)
    init()
    loaded_model, loaded_si = load_preloaded_data(init.digest)

    assert loaded_model.setters["distance"].value is None
    assert loaded_si[0].noise_model is None
    assert loaded_si[0].passbands is None


def test_initializer_round_trip(preload_inputs):
    """``make_worker_initializer`` registers a digest, and the slot starts empty."""
    model, si = preload_inputs
    init = make_worker_initializer(model, si)  # a single SurveyInfo is accepted
    assert is_known_preload_digest(init.digest) and has_known_preload_digests()
    assert load_preloaded_data(init.digest) is None  # slot still empty
    init()
    m1, s1 = load_preloaded_data(init.digest)
    m2, s2 = load_preloaded_data(init.digest)
    assert m1 is m2 and m1 is not model  # one model per process, loaded once
    assert s1 is s2
    assert worker_preload._slot._data is None  # the pickled bytes are dropped once loaded
    assert load_preloaded_data("0" * 32) is None


def test_initializer_is_picklable(preload_inputs):
    """A pickled initializer still works, and the slot holds one digest per process."""
    model, si = preload_inputs
    init = pickle.loads(pickle.dumps(make_worker_initializer(model, si)))
    init()
    other = make_worker_initializer(
        ConstantSEDModel(brightness=1.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0), si
    )
    other()
    assert load_preloaded_data(init.digest) is None  # one preload per process
    assert load_preloaded_data(other.digest) is not None


def test_initializer_repr_has_no_payload(preload_inputs):
    """``repr`` shows the digest prefix and sizes, and stays short (no bytes dump)."""
    init = make_worker_initializer(*preload_inputs)
    assert init.digest[:8] in repr(init) and len(repr(init)) < 200


def test_make_worker_initializer_rejects_obstable(preload_inputs):
    """Passing a bare ObsTable instead of a SurveyInfo raises a clear TypeError naming its type."""
    model, si = preload_inputs
    with pytest.raises(TypeError, match="SurveyInfo") as excinfo:
        make_worker_initializer(model, si.obstable)
    assert "OpSim" in str(excinfo.value)


def test_make_worker_initializer_refuses_unpicklable_model(preload_inputs, make_stateful_model):
    """``make_worker_initializer`` raises for data that refuses pickling, with no fork pool needed.

    This pins the same refusal as ``test_fork_pool_refuses_unpicklable`` in
    ``test_simulate_parallel_backends.py``, without depending on the ``fork`` start method,
    which is not available on every platform.
    """
    _, si = preload_inputs
    stateful_model = make_stateful_model(si)
    with pytest.raises(RuntimeError, match="cannot be pickled"):
        make_worker_initializer(stateful_model, si)


def test_shared_objects_resolve_to_one_instance(preload_inputs):
    """Shared objects reachable from more than one place still resolve to one instance."""
    _, si = preload_inputs
    si2 = SurveyInfo(obstable=si.obstable, passbands=si.passbands, noise_model=si.noise_model)
    sampler = ObsTableUniformRADECSampler(si.obstable, radius=1.0, node_label="pos")
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=sampler.ra, dec=sampler.dec, redshift=0.0)

    init = make_worker_initializer(model, [si, si2])
    init()
    loaded_model, loaded_si = load_preloaded_data(init.digest)

    assert loaded_si[0].obstable is loaded_si[1].obstable
    loaded_sampler = loaded_model.setters["ra"].dependency
    assert loaded_sampler.data is loaded_si[0].obstable


def test_initializer_installs_at_most_once(preload_inputs):
    """An initializer releases its payload once installed, so calling it again never reinstalls."""
    model, si = preload_inputs
    init = make_worker_initializer(model, si)  # never pickled
    init()
    other = make_worker_initializer(
        ConstantSEDModel(brightness=1.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0), si
    )
    other()  # the slot now holds a different digest

    init()  # must not raise, and must not touch the slot
    assert load_preloaded_data(init.digest) is None
    assert load_preloaded_data(other.digest) is not None


def test_same_digest_install_keeps_loaded_slot(preload_inputs):
    """A second initializer with the digest the slot already holds leaves the loaded data in place.

    Backends may run an initializer more than once per worker (for example dask, when callbacks
    are registered again), and replacing the slot would unpickle everything again.
    """
    init = make_worker_initializer(*preload_inputs)
    first, second = (pickle.loads(pickle.dumps(init)) for _ in range(2))
    first()
    loaded_model, _ = load_preloaded_data(init.digest)
    second()
    assert load_preloaded_data(init.digest)[0] is loaded_model
