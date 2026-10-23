"""Tests for the two-part serialization and content digest used by worker preloading."""

import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

from lightcurvelynx.math_nodes.given_sampler import GivenValueList
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
    load_serialized_data,
    make_worker_initializer,
    reset_preload_state,
    serialize_simulation_data,
)


def test_serialize_is_deterministic_and_modes_agree(preload_inputs):
    """Keeping or dropping the bytes must not change the digest, and re-running must agree."""
    model, si = preload_inputs
    kept = serialize_simulation_data(model, [si], keep_bytes=True)
    hashed = serialize_simulation_data(model, [si], keep_bytes=False)
    assert kept.digest == hashed.digest == compute_preload_digest(model, [si])
    assert kept.shared_digest == hashed.shared_digest
    assert hashed.model_bytes is None and hashed.shared_bytes is None
    assert serialize_simulation_data(model, [si], keep_bytes=True).model_bytes == kept.model_bytes


def test_digest_changes_with_model_or_obstable(preload_inputs):
    """The digest must change if either the model or an ObsTable column changes."""
    model, si = preload_inputs
    before = compute_preload_digest(model, [si])
    model.set_parameter("redshift", 0.1)
    after_model = compute_preload_digest(model, [si])
    # NOTE: the raw sqlite column is "zp_nJy", but ObsTable's default OpSim colmap renames it
    # to the standard name "zp" on load, so the in-memory table column is "zp".
    si.obstable._table.loc[0, "zp"] *= 2.0
    after_table = compute_preload_digest(model, [si])
    assert len({before, after_model, after_table}) == 3


def test_obstable_inside_model_is_shared(preload_inputs, test_data_dir):
    """An ObsTable held inside the model (e.g. via a sampler) is shared, not duplicated."""
    # ObsTableUniformRADECSampler holds the ObsTable as .data. Its bytes go to the shared part.
    _, si = preload_inputs
    sampler = ObsTableUniformRADECSampler(si.obstable, radius=1.0, seed=3, node_label="pos")
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=sampler.ra, dec=sampler.dec, redshift=0.0)
    data = serialize_simulation_data(model, [si], keep_bytes=True)
    assert len(data.model_bytes) * 10 < len(data.shared_bytes)
    loaded_model, loaded_si, _ = load_serialized_data(data.model_bytes, data.shared_bytes)
    loaded_sampler = loaded_model.setters["ra"].dependency
    assert isinstance(loaded_sampler, ObsTableUniformRADECSampler)
    assert loaded_sampler.data is loaded_si[0].obstable


def test_extra_obstable_in_model_is_appended(preload_inputs, test_data_dir):
    """An ObsTable reachable from the model but not part of survey_info is still shared."""
    # An ObsTable not in survey_info (a second OpSim load) is also shared: the model bytes stay
    # small and the loaded sampler's .data is one of the shared objects but not loaded_si[0].obstable.
    _, si = preload_inputs
    extra_obstable = OpSim.from_db(test_data_dir / "opsim_small.db")
    sampler = ObsTableUniformRADECSampler(extra_obstable, radius=1.0, seed=3, node_label="pos")
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=sampler.ra, dec=sampler.dec, redshift=0.0)

    data = serialize_simulation_data(model, [si], keep_bytes=True)
    assert len(data.model_bytes) * 10 < len(data.shared_bytes)

    loaded_model, loaded_si, shared_objects = load_serialized_data(data.model_bytes, data.shared_bytes)
    loaded_sampler = loaded_model.setters["ra"].dependency
    assert isinstance(loaded_sampler, ObsTableUniformRADECSampler)
    assert loaded_sampler.data is not loaded_si[0].obstable
    assert any(loaded_sampler.data is obj for obj in shared_objects)


def test_serialize_none_components(test_data_dir):
    """None-valued survey components and model parameters survive the round trip as None."""
    # SurveyInfo(obstable=opsim, survey_name="none") has noise_model None and passbands None.
    # Model attributes that are None stay None after load_serialized_data, and the survey's
    # noise_model and passbands load as None.
    opsim = OpSim.from_db(test_data_dir / "opsim_small.db")
    si = SurveyInfo(obstable=opsim, survey_name="none")
    assert si.noise_model is None
    assert si.passbands is None

    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0)
    assert model.setters["distance"].value is None

    data = serialize_simulation_data(model, [si], keep_bytes=True)
    loaded_model, loaded_si, _ = load_serialized_data(data.model_bytes, data.shared_bytes)

    assert loaded_model.setters["distance"].value is None
    assert loaded_si[0].noise_model is None
    assert loaded_si[0].passbands is None


def test_serialize_refusing_object_raises(preload_inputs):
    """An object that refuses pickling must raise the same exception unchanged."""
    _, si = preload_inputs
    model = ConstantSEDModel(brightness=GivenValueList([1.0, 2.0]), t0=0.0, ra=0.0, dec=0.0, redshift=0.0)
    with pytest.raises(RuntimeError, match="cannot be pickled"):
        serialize_simulation_data(model, [si], keep_bytes=False)


def test_initializer_round_trip(preload_inputs):
    """``make_worker_initializer`` registers a digest, and the slot starts empty."""
    model, si = preload_inputs
    init = make_worker_initializer(model, si)  # a single SurveyInfo is accepted
    assert is_known_preload_digest(init.digest) and has_known_preload_digests()
    assert load_preloaded_data(init.digest) is None  # slot still empty
    init()
    m1, s1 = load_preloaded_data(init.digest)
    m2, s2 = load_preloaded_data(init.digest)
    assert m1 is not m2 and m1 is not model  # fresh model per call
    assert s1 is s2 and s1[0].obstable is s2[0].obstable  # shared data loaded once
    assert load_preloaded_data("0" * 32) is None


def test_initializer_is_picklable_and_idempotent(preload_inputs):
    """A pickled initializer still works, a second call is a no-op, and the slot holds one digest."""
    model, si = preload_inputs
    init = pickle.loads(pickle.dumps(make_worker_initializer(model, si)))
    init()
    init()  # second call is a no-op
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
    """Passing a bare ObsTable instead of a SurveyInfo raises a clear TypeError."""
    model, si = preload_inputs
    with pytest.raises(TypeError, match="SurveyInfo"):
        make_worker_initializer(model, si.obstable)


def test_shared_objects_resolve_to_one_instance(preload_inputs):
    """Shared objects reachable from more than one place still resolve to one instance."""
    _, si = preload_inputs
    si2 = SurveyInfo(obstable=si.obstable, passbands=si.passbands, noise_model=si.noise_model)
    sampler = ObsTableUniformRADECSampler(si.obstable, radius=1.0, seed=3, node_label="pos")
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=sampler.ra, dec=sampler.dec, redshift=0.0)

    init = make_worker_initializer(model, [si, si2])
    init()
    loaded_model, loaded_si = load_preloaded_data(init.digest)

    assert loaded_si[0].obstable is loaded_si[1].obstable
    loaded_sampler = loaded_model.setters["ra"].dependency
    assert loaded_sampler.data is loaded_si[0].obstable


def test_lazy_shared_load_is_thread_safe(preload_inputs):
    """Concurrent loads from many threads share one shared-data load, but not the model."""
    model, si = preload_inputs
    init = make_worker_initializer(model, si)
    init()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: load_preloaded_data(init.digest), range(8)))

    models = [result[0] for result in results]
    survey_lists = [result[1] for result in results]
    assert all(survey_list is survey_lists[0] for survey_list in survey_lists)
    assert len({id(m) for m in models}) == len(models)


def test_unpickled_initializer_releases_bytes_after_first_call(preload_inputs):
    """A pickled initializer drops its own bytes after being called and loaded once."""
    model, si = preload_inputs
    init = pickle.loads(pickle.dumps(make_worker_initializer(model, si)))
    init()
    load_preloaded_data(init.digest)
    assert init._model_bytes is None
    assert init._shared_bytes is None
    # The slot itself must also have dropped the shared bytes after the first load.
    assert worker_preload._slot._shared_bytes is None


def test_never_pickled_initializer_keeps_bytes_and_can_reinstall(preload_inputs):
    """An initializer that was never pickled keeps its bytes and can preload again."""
    model, si = preload_inputs
    init = make_worker_initializer(model, si)
    init()
    assert init._model_bytes is not None
    assert init._shared_bytes is not None

    reset_preload_state()
    init()  # must still work: this instance never released its bytes
    assert load_preloaded_data(init.digest) is not None


def test_emptied_unpickled_initializer_call_is_a_safe_noop(preload_inputs):
    """Calling an already-emptied, unpickled initializer never raises and never installs."""
    model, si = preload_inputs
    init = pickle.loads(pickle.dumps(make_worker_initializer(model, si)))
    init()
    load_preloaded_data(init.digest)  # releases init's own bytes
    assert init._model_bytes is None

    other = make_worker_initializer(
        ConstantSEDModel(brightness=1.0, t0=0.0, ra=0.0, dec=0.0, redshift=0.0), si
    )
    other()  # the slot now holds a different digest

    init()  # must not raise, and must not touch the slot
    assert load_preloaded_data(init.digest) is None
