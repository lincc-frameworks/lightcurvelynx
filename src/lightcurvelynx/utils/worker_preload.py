"""Two-part serialization and content digest for preloading simulation data on workers.

Running ``simulate_lightcurves`` in parallel pickles the model and the list of
``SurveyInfo`` objects into every batch sent to a worker. For a survey the size of the
full LSST OpSim, each ``ObsTable`` alone is close to a gigabyte, so that cost is paid
once per batch instead of once per worker. This module provides the building block that
lets the data be sent to each worker once: it serializes ``(model, survey_info)`` into
two parts and computes a content digest that changes whenever either part changes.

The two parts are:

* **The shared part**: every ``SurveyInfo`` together with its ``obstable``, ``passbands``
  and ``noise_model``, plus every ``ObsTable`` reachable from the model (for example one
  held by an ``ObsTableUniformRADECSampler``). All of these objects are read-only during a
  simulation, so they can be unpickled once per worker and reused by every batch.
* **The model part**: the model, pickled with a ``persistent_id`` hook that replaces any
  object already in the shared part (found by identity) with a small integer reference.
  Any ``ObsTable`` reachable from the model is treated the same way, even if it was not
  already part of a ``SurveyInfo``: this is what keeps the model part small when a sampler
  stores a whole ``ObsTable``, such as ``ObsTableUniformRADECSampler``.

Every ``ObsTable`` is shared (not just the ones already attached to a ``SurveyInfo``)
because an ``ObsTable`` is the one kind of object in a typical model that is both common
and large; sharing it is what makes preloading worthwhile.

Both parts are pickled into a writer that always updates a hash and only optionally keeps
the bytes, so the digest computed in hash-only mode (used for a cheap freshness check) is
identical to the digest computed when the bytes are kept (used to build the payload that is
actually sent to workers). ``cloudpickle`` is used instead of the standard library
``pickle`` when it is importable, so that classes defined in a notebook's ``__main__``
module keep working with backends such as loky, dask and ray. An object that refuses
pickling (for example a stateful ``GivenValueList``) raises the same exception it would
raise from a plain ``pickle.dumps`` call.
"""

import hashlib
import io
import pickle
from typing import NamedTuple

from lightcurvelynx.obstable.obs_table import ObsTable

try:
    import cloudpickle

    _BasePickler = cloudpickle.Pickler
except ImportError:  # pragma: no cover - exercised only when cloudpickle is not installed.
    _BasePickler = pickle.Pickler

# Pickle protocol used for both parts. Protocol 5 is available on all supported Python
# versions and is required for out-of-band buffers, which cloudpickle may use.
_PICKLE_PROTOCOL = 5

# Digest size (in bytes) for the blake2b hashes used throughout this module. 16 bytes
# (128 bits) is more than enough collision resistance for a content digest used only to
# detect staleness, and is fast to compute even over a full OpSim's worth of bytes.
_DIGEST_SIZE = 16


class SerializedData(NamedTuple):
    """The result of serializing a model and its survey data for worker preloading.

    Attributes
    ----------
    digest : str
        A hex digest computed over both the model part and the shared part. Changes
        whenever either part changes.
    shared_digest : str
        A hex digest computed over the shared part alone (the survey data and any
        ``ObsTable`` reachable from the model).
    model_bytes : bytes or None
        The pickled model part, with shared objects replaced by references. ``None``
        when the data was serialized with ``keep_bytes=False``.
    shared_bytes : bytes or None
        The pickled shared part: the survey info list and the list of shared objects.
        ``None`` when the data was serialized with ``keep_bytes=False``.
    """

    digest: str
    shared_digest: str
    model_bytes: bytes | None
    shared_bytes: bytes | None


class _HashingWriter:
    """A file-like object that hashes everything written to it and optionally keeps it.

    Used as the target of a ``Pickler`` so that the same code path computes a digest
    whether or not the pickled bytes are kept, which is what guarantees that
    ``keep_bytes=True`` and ``keep_bytes=False`` produce identical digests.
    """

    def __init__(self, *, keep_bytes):
        self._hasher = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        self._buffer = bytearray() if keep_bytes else None

    def write(self, data):
        """Hash (and optionally store) a chunk of bytes written by the pickler."""
        # Protocol 5 may write a pickle.PickleBuffer for objects that support the
        # buffer protocol (e.g. numpy arrays). It has no __len__, so normalize to
        # bytes first; hashlib and bytearray.extend both accept the result either way.
        if not isinstance(data, (bytes, bytearray)):
            data = bytes(data)
        self._hasher.update(data)
        if self._buffer is not None:
            self._buffer.extend(data)
        return len(data)

    def digest(self):
        """Return the raw digest bytes of everything written so far."""
        return self._hasher.digest()

    def hexdigest(self):
        """Return the hex digest of everything written so far."""
        return self._hasher.hexdigest()

    def getvalue(self):
        """Return the bytes written so far, or None if they were not kept."""
        return bytes(self._buffer) if self._buffer is not None else None


class _ModelPickler(_BasePickler):
    """Pickler for the model part that replaces shared objects with small references."""

    def __init__(self, file, id_to_index, shared_objects):
        super().__init__(file, protocol=_PICKLE_PROTOCOL)
        self._id_to_index = id_to_index
        self._shared_objects = shared_objects

    def persistent_id(self, obj):
        """Return an index into the shared objects for anything that should be shared."""
        obj_id = id(obj)
        index = self._id_to_index.get(obj_id)
        if index is not None:
            return index
        if isinstance(obj, ObsTable):
            index = len(self._shared_objects)
            self._shared_objects.append(obj)
            self._id_to_index[obj_id] = index
            return index
        return None


class _ModelUnpickler(pickle.Unpickler):
    """Unpickler for the model part that resolves references into the shared objects."""

    def __init__(self, file, shared_objects):
        super().__init__(file)
        self._shared_objects = shared_objects

    def persistent_load(self, pid):
        """Resolve a persistent id written by ``_ModelPickler`` back into an object."""
        return self._shared_objects[pid]


def _collect_shared_objects(survey_info):
    """Collect the survey components to share, preserving first-seen order.

    Parameters
    ----------
    survey_info : list of SurveyInfo
        The survey information to collect shared objects from.

    Returns
    -------
    shared_objects : list
        The SurveyInfo objects and their non-None obstable, passbands and noise_model,
        each appearing exactly once (by identity).
    id_to_index : dict
        A mapping from ``id(obj)`` to its index in ``shared_objects``.
    """
    shared_objects = []
    id_to_index = {}
    for info in survey_info:
        for obj in (info, info.obstable, info.passbands, info.noise_model):
            if obj is None:
                continue
            obj_id = id(obj)
            if obj_id in id_to_index:
                continue
            id_to_index[obj_id] = len(shared_objects)
            shared_objects.append(obj)
    return shared_objects, id_to_index


def serialize_simulation_data(model, survey_info, *, keep_bytes):
    """Serialize a model and its survey data into a model part and a shared part.

    Parameters
    ----------
    model : ParameterizedNode
        The model to serialize.
    survey_info : list of SurveyInfo
        The survey information to serialize alongside the model.
    keep_bytes : bool
        If True, the pickled bytes for both parts are kept in the returned
        ``SerializedData``. If False, only the digests are computed and both byte
        fields are ``None``; the bytes are never held in memory in this mode.

    Returns
    -------
    SerializedData
        The digests and (optionally) the pickled bytes for both parts.

    Raises
    ------
    Exception
        Propagates unchanged whatever pickling the model or the shared part raises,
        for example the ``RuntimeError`` raised by a stateful ``GivenValueList``.
    """
    shared_objects, id_to_index = _collect_shared_objects(survey_info)

    # Pickle the model part first: this is what may discover and append extra
    # ObsTable instances (e.g. one held by a sampler) to shared_objects.
    model_writer = _HashingWriter(keep_bytes=keep_bytes)
    _ModelPickler(model_writer, id_to_index, shared_objects).dump(model)

    # Now pickle the shared part, including any ObsTable instances appended above.
    shared_writer = _HashingWriter(keep_bytes=keep_bytes)
    _BasePickler(shared_writer, protocol=_PICKLE_PROTOCOL).dump((survey_info, shared_objects))

    digest = hashlib.blake2b(
        model_writer.digest() + shared_writer.digest(), digest_size=_DIGEST_SIZE
    ).hexdigest()

    return SerializedData(
        digest=digest,
        shared_digest=shared_writer.hexdigest(),
        model_bytes=model_writer.getvalue(),
        shared_bytes=shared_writer.getvalue(),
    )


def compute_preload_digest(model, survey_info):
    """Compute the content digest of a model and its survey data, without keeping bytes.

    Equivalent to ``serialize_simulation_data(model, survey_info, keep_bytes=False).digest``,
    but documents the intent of the cheap, hash-only check used to detect staleness.

    Parameters
    ----------
    model : ParameterizedNode
        The model to digest.
    survey_info : list of SurveyInfo
        The survey information to digest alongside the model.

    Returns
    -------
    str
        The hex digest covering both the model and the shared survey data.
    """
    return serialize_simulation_data(model, survey_info, keep_bytes=False).digest


def load_serialized_data(model_bytes, shared_bytes):
    """Reconstruct a model and survey data from bytes produced by ``serialize_simulation_data``.

    Parameters
    ----------
    model_bytes : bytes
        The pickled model part, as returned in ``SerializedData.model_bytes``.
    shared_bytes : bytes
        The pickled shared part, as returned in ``SerializedData.shared_bytes``.

    Returns
    -------
    model : ParameterizedNode
        A freshly unpickled model, with references into ``shared_objects`` resolved.
    survey_info : list of SurveyInfo
        A freshly unpickled list of SurveyInfo objects.
    shared_objects : list
        The shared objects (SurveyInfo components and any extra ObsTable instances)
        that ``model`` and ``survey_info`` reference.
    """
    survey_info, shared_objects = pickle.loads(shared_bytes)
    model = _ModelUnpickler(io.BytesIO(model_bytes), shared_objects).load()
    return model, survey_info, shared_objects
