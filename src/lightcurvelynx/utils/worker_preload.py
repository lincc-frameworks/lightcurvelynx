"""Preloading simulation data on worker processes.

Running ``simulate_lightcurves`` in parallel pickles the model and the list of
``SurveyInfo`` objects into every batch sent to a worker. For a survey the size of the
full LSST OpSim, each ``ObsTable`` alone is close to a gigabyte, so sending that data once
per worker instead of once per batch is worth a dedicated code path. This module provides
that path, in layers:

* **Serialization** (``serialize_simulation_data``, ``compute_preload_digest``): pickles
  ``(model, survey_info)`` into one payload and computes a content digest that changes
  whenever the model or the survey data changes.
* **The initializer** (``make_worker_initializer``, ``WorkerInitializer``): wraps the
  serialized payload in a picklable, zero-argument callable that a worker runs once to
  install it.
* **The registry and the slot** (``_known_digests``, ``_PreloadSlot``,
  ``load_preloaded_data``): the process-level set of digests that
  ``make_worker_initializer`` has produced, and the single process-level slot that holds
  whatever payload was last installed there.
* **The fallback warning and the miss marker** (``WorkerPreloadWarning``,
  ``warn_preload_fallback``, ``PreloadMiss``): cover the cases where a batch cannot use
  the preload and falls back to carrying its own copy of the data.

A worker unpickles the payload once, the first time it runs a preloaded batch, and every
later batch on that worker reuses the same model and survey data. LightCurveLynx supports
only one thread per worker, so these batches run one at a time. Because a single pickle
keeps shared objects shared, an ``ObsTable`` reachable from both the model (for example
one held by an ``ObsTableUniformRADECSampler``) and a ``SurveyInfo`` is loaded once.

The payload is pickled into a writer that always updates a hash and only optionally keeps
the bytes, so the digest computed in hash-only mode (used for a cheap freshness check) is
identical to the digest computed when the bytes are kept (used to build the payload that is
actually sent to workers). ``make_worker_initializer`` uses ``cloudpickle`` instead of the
standard library ``pickle`` when it is importable, so that classes defined in a notebook's
``__main__`` module keep working with backends such as loky, dask and ray. An object that
refuses pickling (for example a stateful ``GivenValueList``) raises the same exception it
would raise from a plain ``pickle.dumps`` call.
"""

import hashlib
import io
import pickle
import warnings
from typing import NamedTuple

from lightcurvelynx.survey_info import SurveyInfo

try:
    import cloudpickle

    _DEFAULT_PICKLER = cloudpickle.Pickler
except ImportError:  # pragma: no cover - exercised only when cloudpickle is not installed.
    _DEFAULT_PICKLER = pickle.Pickler

# Pickle protocol used for the payload. Protocol 5 lets _HashingWriter.write receive a large
# buffer (e.g. a numpy array) as a pickle.PickleBuffer instead of a copied bytes object; the
# stdlib default is still protocol 4 on Python 3.11-3.13. No buffer_callback is used, so these
# buffers are written inline rather than collected out-of-band.
_PICKLE_PROTOCOL = 5

# Digest size (in bytes) for the blake2b hash of the payload. 16 bytes (128 bits) is more
# than enough collision resistance for a content digest used only to detect staleness, and
# is fast to compute even over a full OpSim's worth of bytes.
_DIGEST_SIZE = 16


class SerializedData(NamedTuple):
    """The result of serializing a model and its survey data for worker preloading.

    Attributes
    ----------
    digest : str
        A hex digest of the pickled ``(model, survey_info)`` payload. Changes whenever
        the model or the survey data changes.
    data : bytes or None
        The pickled payload. ``None`` when the data was serialized with
        ``keep_bytes=False``.
    """

    digest: str
    data: bytes | None


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

    def hexdigest(self):
        """Return the hex digest of everything written so far."""
        return self._hasher.hexdigest()

    def getvalue(self):
        """Return the bytes written so far, or None if they were not kept."""
        return self._buffer.getvalue() if self._buffer is not None else None


def serialize_simulation_data(model, survey_info, *, keep_bytes, pickler=_DEFAULT_PICKLER):
    """Serialize a model and its survey data into one payload.

    The digest depends on the pickler used, so a digest is only comparable with digests
    made with the same pickler.

    Parameters
    ----------
    model : ParameterizedNode
        The model to serialize.
    survey_info : list of SurveyInfo
        The survey information to serialize alongside the model.
    keep_bytes : bool
        If True, the pickled bytes are kept in the returned ``SerializedData``. If False,
        only the digest is computed and ``data`` is ``None``; the bytes are never held in
        memory in this mode.
    pickler : type, optional
        The ``Pickler`` class to serialize with. Defaults to ``cloudpickle.Pickler`` when
        cloudpickle is importable, and to ``pickle.Pickler`` otherwise.

    Returns
    -------
    SerializedData
        The digest and (optionally) the pickled bytes.

    Raises
    ------
    Exception
        Propagates unchanged whatever pickling the model or the survey data raises,
        for example the ``RuntimeError`` raised by a stateful ``GivenValueList``.
    """
    writer = _HashingWriter(keep_bytes=keep_bytes)
    pickler(writer, protocol=_PICKLE_PROTOCOL).dump((model, survey_info))
    return SerializedData(digest=writer.hexdigest(), data=writer.getvalue())


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
        The hex digest covering both the model and the survey data.
    """
    return serialize_simulation_data(model, survey_info, keep_bytes=False).digest


class _PreloadSlot:
    """A single process's preloaded payload, unpickled the first time it is loaded.

    Parameters
    ----------
    digest : str
        The content digest identifying this payload.
    data : bytes
        The pickled ``(model, survey_info)`` payload. Dropped after the first call to ``load``.
    """

    def __init__(self, digest, data):
        self.digest = digest
        self._data = data
        self._loaded = None

    def load(self):
        """Return the cached ``(model, survey_info)``, unpickling it once if needed.

        Returns
        -------
        tuple of (ParameterizedNode, list of SurveyInfo)
            The model and the survey info list, the same objects on every call.
        """
        if self._loaded is None:
            self._loaded = pickle.loads(self._data)
            self._data = None
        return self._loaded


# The current process's preload slot. There is at most one preloaded payload per process:
# installing a new digest replaces the slot.
_slot = None


class WorkerInitializer:
    """A picklable, zero-argument callable that preloads simulation data on a worker.

    An instance holds the serialized payload produced by ``make_worker_initializer``.
    Calling it installs the payload into this process's module-level preload slot, unless
    the slot already holds this digest, and then releases the instance's own copy of the
    bytes: keeping them would roughly double the memory cost of the largest preloaded object
    (typically an ``ObsTable``) for the worker's whole life. Passing an instance as a
    ``ProcessPoolExecutor`` (or similar) initializer runs it once per worker process, where
    each worker has its own copy of the instance. Calling an instance that has already
    released its bytes is a safe no-op: it never raises and never touches the slot,
    whatever the slot now holds.

    Parameters
    ----------
    digest : str
        The content digest identifying this payload, as returned in
        ``SerializedData.digest``.
    data : bytes
        The pickled payload, as returned in ``SerializedData.data``.
    """

    def __init__(self, digest, data):
        self._digest = digest
        self._data = data

    @property
    def digest(self):
        """str: The content digest of the preloaded payload."""
        return self._digest

    def __call__(self):
        """Install this payload into the process's preload slot, then release it.

        Does nothing if this instance already released its bytes. Otherwise replaces the
        slot, unless it already holds this digest. This method only stores the pickled
        bytes; nothing is unpickled here.
        """
        global _slot
        if self._data is None:
            return
        if _slot is None or _slot.digest != self._digest:
            _slot = _PreloadSlot(self._digest, self._data)
        self._data = None

    def __repr__(self):
        """Return a short repr with the digest prefix and payload size, never the bytes."""
        size = "released" if self._data is None else f"{len(self._data) / 1e6:.2f}MB"
        return f"{type(self).__name__}(digest={self._digest[:8]}, data={size})"


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
        If ``survey_info`` is neither a SurveyInfo nor a list of SurveyInfo.
    """
    if isinstance(survey_info, SurveyInfo):
        return [survey_info]
    if isinstance(survey_info, list) and all(isinstance(info, SurveyInfo) for info in survey_info):
        return list(survey_info)
    if isinstance(survey_info, list):
        bad_type = next(type(info).__name__ for info in survey_info if not isinstance(info, SurveyInfo))
    else:
        bad_type = type(survey_info).__name__
    raise TypeError(
        "survey_info must be a SurveyInfo or a list of SurveyInfo instances; "
        f"wrap bare ObsTable objects in a SurveyInfo first. Got {bad_type}."
    )


# Process-level registry of digests created by make_worker_initializer in this process. Never
# pruned: it only ever holds a handful of small hex strings.
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
    initializer = _build_worker_initializer(model, survey_info, _DEFAULT_PICKLER)
    _known_digests.add(initializer.digest)
    return initializer


def _build_worker_initializer(model, survey_info, pickler):
    """Serialize a model and its survey data with a chosen pickler and return an initializer.

    Parameters
    ----------
    model : ParameterizedNode
        The model to preload on workers.
    survey_info : list of SurveyInfo
        The survey information to preload alongside the model.
    pickler : type
        The ``Pickler`` class to serialize with.

    Returns
    -------
    WorkerInitializer
        A picklable, zero-argument callable that installs the serialized payload into
        a worker's preload slot when called.
    """
    data = serialize_simulation_data(model, survey_info, keep_bytes=True, pickler=pickler)
    return WorkerInitializer(data.digest, data.data)


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


def load_preloaded_data(digest):
    """Return the preloaded model and survey info for ``digest``, or None.

    The payload is unpickled the first time it is loaded, after which its bytes are dropped
    and every call returns the same model and survey info objects.

    Parameters
    ----------
    digest : str
        The digest of the payload to load, as returned by ``WorkerInitializer.digest``.

    Returns
    -------
    tuple of (ParameterizedNode, list of SurveyInfo) or None
        This process's preloaded model and survey info list, or None if the process's
        preload slot is empty or holds a different digest.
    """
    if _slot is None or _slot.digest != digest:
        return None
    return _slot.load()


def reset_preload_state():
    """Clear the process-level registry and the preload slot.

    Intended for tests, so preload state from one test never leaks into another.
    """
    global _slot
    _known_digests.clear()
    _slot = None


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
    "(see the parallel_runs notebook)."
)


def warn_preload_fallback(sentence):
    """Emit a ``WorkerPreloadWarning`` saying why batches are sent with full data.

    Parameters
    ----------
    sentence : str
        The sentence describing which fallback happened. The caller builds this,
        including any case-specific detail (for example the miss count or the
        exception that pickling raised).
    """
    # Both callers sit one level below simulate_lightcurves (simulate_lightcurves ->
    # _select_preload_digest or _simulate_lightcurves_parallel -> this function), so
    # stacklevel=4 points the warning at the user's simulate_lightcurves call.
    warnings.warn(f"{sentence} {_FALLBACK_SUFFIX}", WorkerPreloadWarning, stacklevel=4)
