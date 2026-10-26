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
actually sent to workers). ``make_worker_initializer`` uses ``cloudpickle`` instead of the
standard library ``pickle`` when it is importable, so that classes defined in a notebook's
``__main__`` module keep working with backends such as loky, dask and ray. The ``num_jobs``
path of ``simulate_lightcurves`` uses the standard library ``pickle``, as a plain
``ProcessPoolExecutor`` does, so it pickles exactly what it did before preloading existed
(for example, functions from ``__main__`` by reference). An object that refuses
pickling (for example a stateful ``GivenValueList``) raises the same exception it would
raise from a plain ``pickle.dumps`` call.
"""

import hashlib
import io
import os
import pickle
import threading
import warnings
from typing import NamedTuple

from lightcurvelynx.obstable.obs_table import ObsTable
from lightcurvelynx.survey_info import SurveyInfo

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
    model_bytes : bytes or None
        The pickled model part, with shared objects replaced by references. ``None``
        when the data was serialized with ``keep_bytes=False``.
    shared_bytes : bytes or None
        The pickled shared part: the survey info list and the list of shared objects.
        ``None`` when the data was serialized with ``keep_bytes=False``.
    """

    digest: str
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
        self._buffer = io.BytesIO() if keep_bytes else None

    def write(self, data):
        """Hash (and optionally store) a chunk of bytes written by the pickler, without copying it."""
        # Protocol 5 writes large buffers (e.g. numpy arrays) straight to the file as a
        # pickle.PickleBuffer, which has no len(). Its raw() view is the flat memory that pickle
        # writes, without a copy. hashlib and BytesIO accept it, as they do bytes and bytearray.
        if isinstance(data, pickle.PickleBuffer):
            data = data.raw()
        self._hasher.update(data)
        if self._buffer is not None:
            self._buffer.write(data)
        return len(data)

    def digest(self):
        """Return the raw digest bytes of everything written so far."""
        return self._hasher.digest()

    def getvalue(self):
        """Return the bytes written so far, or None if they were not kept."""
        return self._buffer.getvalue() if self._buffer is not None else None


class _ModelPicklerMixin:
    """Persistent-id hook for the model part, which replaces shared objects with small references.

    Mixed into a concrete pickler class (see ``_DEFAULT_PICKLERS`` and ``_STDLIB_PICKLERS``), so
    that the model part can be pickled with ``cloudpickle`` or with the standard library's
    ``pickle``.
    """

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


class _ModelPickler(_ModelPicklerMixin, _BasePickler):
    """Pickler for the model part, built on ``cloudpickle`` when it is importable."""


class _StdlibModelPickler(_ModelPicklerMixin, pickle.Pickler):
    """Pickler for the model part, built on the standard library's ``pickle``."""


# Pickler pairs, as (model-part pickler, shared-part pickler). The shared part always uses the
# plain pickler, without the persistent_id hook: with the hook, each shared object would be
# pickled as a reference to itself instead of its actual contents.
_DEFAULT_PICKLERS = (_ModelPickler, _BasePickler)
_STDLIB_PICKLERS = (_StdlibModelPickler, pickle.Pickler)


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


def serialize_simulation_data(model, survey_info, *, keep_bytes, picklers=_DEFAULT_PICKLERS):
    """Serialize a model and its survey data into a model part and a shared part.

    The digest depends on the picklers used, so a digest is only comparable with digests
    made with the same pickler pair.

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
    picklers : tuple of (type, type), optional
        The (model-part pickler, shared-part pickler) pair to serialize with. Defaults to
        ``_DEFAULT_PICKLERS`` (cloudpickle when importable).

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
    model_pickler, shared_pickler = picklers
    shared_objects, id_to_index = _collect_shared_objects(survey_info)

    # Pickle the model part first: this is what may discover and append extra
    # ObsTable instances (e.g. one held by a sampler) to shared_objects.
    model_writer = _HashingWriter(keep_bytes=keep_bytes)
    model_pickler(model_writer, id_to_index, shared_objects).dump(model)

    # Now pickle the shared part, including any ObsTable instances appended above.
    shared_writer = _HashingWriter(keep_bytes=keep_bytes)
    shared_pickler(shared_writer, protocol=_PICKLE_PROTOCOL).dump((survey_info, shared_objects))

    digest = hashlib.blake2b(
        model_writer.digest() + shared_writer.digest(), digest_size=_DIGEST_SIZE
    ).hexdigest()

    return SerializedData(
        digest=digest,
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


class WorkerInitializer:
    """A picklable, zero-argument callable that preloads simulation data on a worker.

    An instance holds the serialized payload produced by ``make_worker_initializer``.
    Calling it installs the payload into this process's module-level preload slot: the
    first call in a process does the installing, and later calls with the same digest
    are no-ops. Passing an instance as a ``ProcessPoolExecutor`` (or similar) initializer
    runs it once per worker process; calling it directly in each thread of a
    ``ThreadPoolExecutor`` is also safe, since a matching digest is a no-op.

    An instance that reaches a worker by being unpickled (the normal case for
    ``spawn``/``forkserver``/``loky``/``ray`` worker pools) releases its own copy of the
    bytes the first time it is called, since by then they have been handed to the
    process's slot: keeping them around would roughly double the memory cost of the
    largest preloaded object (typically an ``ObsTable``) for the worker's whole life. An
    instance that was never pickled -- held directly by a user, called from every thread
    of a ``ThreadPoolExecutor``, or inherited via ``fork`` -- keeps its bytes indefinitely,
    since in those cases it may legitimately need to install them again (for example
    after ``reset_preload_state`` in tests). Calling an instance that has already released
    its bytes is a safe no-op: it never raises and never touches the slot, so
    ``load_preloaded_data`` simply falls back to returning ``None`` for its digest.

    Parameters
    ----------
    digest : str
        The content digest identifying this payload, as computed by
        ``compute_preload_digest``.
    model_bytes : bytes
        The pickled model part, as returned in ``SerializedData.model_bytes``.
    shared_bytes : bytes
        The pickled shared part, as returned in ``SerializedData.shared_bytes``.
    """

    def __init__(self, digest, model_bytes, shared_bytes):
        self._digest = digest
        self._model_bytes = model_bytes
        self._shared_bytes = shared_bytes
        # True only for instances reconstructed by unpickling (see __setstate__), never
        # for one built directly by make_worker_initializer or inherited via fork.
        self._unpickled = False

    @property
    def digest(self):
        """str: The content digest of the preloaded payload."""
        return self._digest

    def __setstate__(self, state):
        """Restore state from a pickle and mark this instance as having been unpickled."""
        self.__dict__.update(state)
        self._unpickled = True

    def __call__(self):
        """Install this payload into the process's preload slot.

        Does nothing if the slot already holds this digest. Otherwise replaces the
        slot. This method only stores the pickled bytes; nothing is unpickled here. If
        this instance already released its bytes (see the class docstring), this is a
        safe no-op regardless of what the slot currently holds.
        """
        if self._model_bytes is None and self._shared_bytes is None:
            return
        _install_slot(self._digest, self._model_bytes, self._shared_bytes)
        if self._unpickled:
            self._model_bytes = None
            self._shared_bytes = None

    def __repr__(self):
        """Return a short repr with the digest prefix and payload sizes, never the bytes."""
        return (
            f"{type(self).__name__}(digest={self._digest[:8]}, "
            f"model={_format_size_mb(self._model_bytes)}, "
            f"shared={_format_size_mb(self._shared_bytes)})"
        )


def _format_size_mb(data):
    """Format bytes as a size in MB for ``repr``, or "released" once they are gone.

    Parameters
    ----------
    data : bytes or None
        The payload to measure, or None if it has already been released.

    Returns
    -------
    str
        A human-readable size, e.g. "1.23MB", or "released" if ``data`` is None.
    """
    return "released" if data is None else f"{len(data) / 1e6:.2f}MB"


def _normalize_survey_info(survey_info):
    """Normalize ``survey_info`` into a list of SurveyInfo, raising on anything else.

    Parameters
    ----------
    survey_info : SurveyInfo or list of SurveyInfo
        The survey information passed by the caller.

    Returns
    -------
    list of SurveyInfo
        ``survey_info`` as a list, if every element is a SurveyInfo.

    Raises
    ------
    TypeError
        If ``survey_info`` is neither a SurveyInfo nor a list (or tuple) of SurveyInfo.
    """
    if isinstance(survey_info, SurveyInfo):
        return [survey_info]
    if isinstance(survey_info, (list, tuple)):
        for info in survey_info:
            if not isinstance(info, SurveyInfo):
                raise TypeError(
                    "survey_info must be a SurveyInfo or a list of SurveyInfo instances, "
                    f"found {type(info).__name__} in the list; wrap bare ObsTable objects "
                    "in a SurveyInfo first."
                )
        return list(survey_info)
    raise TypeError(
        "survey_info must be a SurveyInfo or a list of SurveyInfo instances, "
        f"got {type(survey_info).__name__}; wrap bare ObsTable objects in a SurveyInfo first."
    )


# Process-level registry of digests created by make_worker_initializer in this process. Never
# pruned: it only ever holds a handful of small hex strings. A plain set's add/clear/membership
# check is a single atomic operation under the GIL, so this registry needs no lock of its own.
_known_digests = set()


def make_worker_initializer(model, survey_info):
    """Serialize a model and its survey data and return a picklable worker initializer.

    Uses cloudpickle (when importable) and registers the digest, so that later
    ``simulate_lightcurves`` calls in this process recognize the preload.

    Parameters
    ----------
    model : ParameterizedNode
        The model to preload on workers.
    survey_info : SurveyInfo or list of SurveyInfo
        The survey information to preload alongside the model.

    Returns
    -------
    WorkerInitializer
        A picklable, zero-argument callable that installs the serialized payload into
        a worker's preload slot when called.

    Raises
    ------
    TypeError
        If ``survey_info`` is neither a SurveyInfo nor a list of SurveyInfo.
    """
    survey_info = _normalize_survey_info(survey_info)
    initializer = _build_worker_initializer(model, survey_info, _DEFAULT_PICKLERS)
    _known_digests.add(initializer.digest)
    return initializer


def _build_worker_initializer(model, survey_info, picklers):
    """Serialize a model and its survey data with a chosen pickler pair and return an initializer.

    Parameters
    ----------
    model : ParameterizedNode
        The model to preload on workers.
    survey_info : list of SurveyInfo
        The survey information to preload alongside the model.
    picklers : tuple of (type, type)
        The (model-part pickler, shared-part pickler) pair to serialize with.

    Returns
    -------
    WorkerInitializer
        A picklable, zero-argument callable that installs the serialized payload into
        a worker's preload slot when called.
    """
    data = serialize_simulation_data(model, survey_info, keep_bytes=True, picklers=picklers)
    return WorkerInitializer(data.digest, data.model_bytes, data.shared_bytes)


def has_known_preload_digests():
    """Return whether any digest has been registered in this process.

    Returns
    -------
    bool
        True if ``make_worker_initializer`` has been called at least once in this
        process, False otherwise.
    """
    return bool(_known_digests)


def is_known_preload_digest(digest):
    """Return whether ``digest`` was registered by ``make_worker_initializer``.

    Parameters
    ----------
    digest : str
        The digest to check.

    Returns
    -------
    bool
        True if ``digest`` is in the process-level registry, False otherwise.
    """
    return digest in _known_digests


class _PreloadSlot:
    """A single process's preloaded payload, with the shared part loaded lazily once.

    Parameters
    ----------
    digest : str
        The content digest identifying this payload.
    model_bytes : bytes
        The pickled model part.
    shared_bytes : bytes
        The pickled shared part. Dropped after the first call to ``shared``.
    """

    def __init__(self, digest, model_bytes, shared_bytes):
        self.digest = digest
        self.model_bytes = model_bytes
        self._shared_bytes = shared_bytes
        self._shared_lock = threading.Lock()
        self._shared = None

    def shared(self):
        """Return the cached ``(survey_info, shared_objects)``, loading it once if needed.

        Returns
        -------
        tuple of (list of SurveyInfo, list)
            The survey info list and the shared objects it (and the model) reference.
        """
        if self._shared is None:
            with self._shared_lock:
                if self._shared is None:
                    self._shared = pickle.loads(self._shared_bytes)
                    self._shared_bytes = None
        return self._shared


# The current process's preload slot, replaced wholesale under _slot_lock. There is at
# most one preloaded payload per process: installing a new digest replaces the slot.
_slot_lock = threading.Lock()
_slot = None


def _install_slot(digest, model_bytes, shared_bytes):
    """Install a payload into the process's preload slot, unless it is already there.

    Parameters
    ----------
    digest : str
        The content digest identifying the payload.
    model_bytes : bytes
        The pickled model part.
    shared_bytes : bytes
        The pickled shared part.
    """
    global _slot
    with _slot_lock:
        if _slot is not None and _slot.digest == digest:
            return
        _slot = _PreloadSlot(digest, model_bytes, shared_bytes)


def load_preloaded_data(digest):
    """Return the preloaded model, survey info and model size for ``digest``, or None.

    Loading the shared part (the survey info and any shared ``ObsTable`` instances) is
    done at most once per slot, under the slot's lock; after that the shared bytes are
    dropped and the loaded objects are reused. The model is freshly unpickled on every
    call, so no two calls ever share a model object.

    Parameters
    ----------
    digest : str
        The digest of the payload to load, as returned by ``WorkerInitializer.digest``.

    Returns
    -------
    tuple of (ParameterizedNode, list of SurveyInfo, int) or None
        The freshly unpickled model, the (possibly shared, cached) survey info list and the
        size in bytes of the pickled model it was loaded from, or None if the process's
        preload slot is empty or holds a different digest.
    """
    slot = _slot
    if slot is None or slot.digest != digest:
        return None
    survey_info, shared_objects = slot.shared()
    model = _ModelUnpickler(io.BytesIO(slot.model_bytes), shared_objects).load()
    return model, survey_info, len(slot.model_bytes)


def reset_preload_state():
    """Clear the process-level registry and the preload slot.

    Intended for tests, so preload state from one test never leaks into another.
    """
    global _slot
    _known_digests.clear()
    with _slot_lock:
        _slot = None


def _reinit_locks_after_fork():
    """Give a forked child fresh, unlocked locks in place of the ones it inherited.

    ``fork`` copies each lock in the state it was in. If another thread of the parent held one
    at that moment (for example while loading the shared part), that thread does not exist in
    the child, so the child's copy would stay locked forever and its first batch would hang.
    Only the locks are replaced: the registry and the slot keep their contents.
    """
    global _slot_lock
    _slot_lock = threading.Lock()
    slot = _slot
    if slot is not None:
        slot._shared_lock = threading.Lock()


if hasattr(os, "register_at_fork"):  # Only platforms that can fork have it.
    os.register_at_fork(after_in_child=_reinit_locks_after_fork)


class WorkerPreloadWarning(UserWarning):
    """Warning emitted when ``simulate_lightcurves`` sends full batches to an executor.

    It is emitted whenever a parallel run cannot use data preloaded with
    ``make_worker_initializer``, so each batch carries a full copy of the model and survey
    data. It is a dedicated subclass so it can be filtered on its own.
    """


class PreloadMiss:
    """Marker returned by a preloaded batch that reached a worker without its data.

    The batch returns it before touching anything else, and the parent resends that batch
    with the full model and survey data. It has no state, so it is cheap to pickle.
    """


# The part of every fallback warning that does not depend on the case.
_FALLBACK_SUFFIX = (
    "Each of these batches carries a full copy of the model and survey data. To send this data to "
    "each worker once, pass initializer=make_worker_initializer(model, survey_info) to the executor "
    "(see the parallel_runs notebook). Batches sent this way that run as threads in one process "
    "share a single model object, which is not safe for models that keep per-call state "
    "(e.g. SncosmoWrapperModel)."
)

# The sentence that says which fallback case happened, keyed by case name.
_FALLBACK_CASES = {
    "not_preloaded": "The executor's workers were not preloaded with make_worker_initializer().",
    "unpicklable": (
        "The model or survey data could not be pickled, so the workers cannot have been preloaded "
        "with it{cause}."
    ),
    "changed": (
        "The model and survey data do not match any data preloaded with make_worker_initializer() "
        "in this process. They may have changed after preloading, including internal state changed "
        "by an earlier run; create the initializer after your last use of the model."
    ),
    "miss": (
        "{num_missed} of {num_batches} batches reached workers that were not preloaded with this "
        "model and survey data."
    ),
}


def warn_preload_fallback(case, *, num_missed=0, num_batches=0, cause=None):
    """Emit a ``WorkerPreloadWarning`` saying why batches are sent with full data.

    Parameters
    ----------
    case : str
        Which fallback happened: "not_preloaded", "unpicklable", "changed" or "miss".
    num_missed : int, optional
        The number of batches that missed. Only used for the "miss" case.
    num_batches : int, optional
        The total number of batches in the run. Only used for the "miss" case.
    cause : Exception, optional
        The exception that pickling raised. Only used for the "unpicklable" case, whose
        sentence then ends with its type and message.
    """
    cause_text = "" if cause is None else f" ({type(cause).__name__}: {cause})"
    sentence = _FALLBACK_CASES[case].format(num_missed=num_missed, num_batches=num_batches, cause=cause_text)
    # Both callers sit one level below simulate_lightcurves (simulate_lightcurves ->
    # _select_preload_digest or _simulate_lightcurves_parallel -> this function), so
    # stacklevel=4 points the warning at the user's simulate_lightcurves call.
    warnings.warn(f"{sentence} {_FALLBACK_SUFFIX}", WorkerPreloadWarning, stacklevel=4)
