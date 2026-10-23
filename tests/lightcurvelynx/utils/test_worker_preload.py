"""Tests for the two-part serialization and content digest used by worker preloading."""

import pytest

from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.ra_dec_sampler import ObsTableUniformRADECSampler
from lightcurvelynx.models.basic_models import ConstantSEDModel
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.survey_info import SurveyInfo
from lightcurvelynx.utils.worker_preload import (
    compute_preload_digest,
    load_serialized_data,
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
