"""A sampling node that uses caustics to build a strong-lens mesh for a caustics lens model,
draw a source inside its strong-lensing region, raytrace it to images, and evaluate the
per-image observables that ``ResolvedStrongLensModel`` consumes.

The node runs on caustics' active backend, torch or jax, which caustics picks once per process, at
import, from the ``CASKADE_BACKEND`` environment variable (default ``torch``).
"""

import copy
import os
import pickle
import sys
import threading
import uuid
import warnings
from itertools import pairwise

import numpy as np

from lightcurvelynx.base_models import FunctionNode

# Substrings of the caustics warnings that mark a failed build: the mesh did not close or the
# μ-region is unbounded (lens_mesh.py, region.py), refinement was depth-limited (lattice.py), or
# a hole curve hit its sample cap (holes.py). Some are emitted with stacklevel=3, which hides the
# emitting module, so all are recognized by text.
_FAILURE_MESSAGES = ("extension(s), at fov=", "is depth-limited", "before every chord fell below")
# Serializes the caustics work of every CausticsStrongLensSystem across threads: every call fills the
# parameters of a node's working lens, which its threads share; under torch, torch.func.jacfwd, used by
# caustics' builds, raytraces and Jacobians, keeps process-global forward-AD state; and the warning
# capture around each build (catch_warnings) mutates process-global state.
_CAUSTICS_LOCK = threading.Lock()
# This process's working lenses and their jitted observables, keyed by (node token, backend, device):
# every copy of one node (e.g. one unpickled per executor batch) shares one, so caustics' compiled
# programs, cached per lens object, are reused. Process-local, never pickled, kept for the life of
# the process, and used only under _CAUSTICS_LOCK.
_WORKING_LENSES = {}


def _import_caustics():
    """Import torch and caustics (with its adaptive mesh API), lazily.

    Returns
    -------
    torch : module
        The ``torch`` module.
    caustics : module
        The ``caustics`` module.

    Raises
    ------
    ImportError
        If torch or caustics (with the adaptive mesh API) are not installed.
    """
    try:
        import caustics
        import caustics.lenses.func.adaptive  # noqa: F401
        import torch
    except ImportError as err:
        raise ImportError(
            "CausticsStrongLensSystem requires torch and caustics (with the adaptive mesh API), and "
            "jax for caustics' jax backend. Install caustics from its feat/adaptive_refactor branch."
        ) from err
    return torch, caustics


def _jax_device(device):
    """Return the jax device named by a torch-style device string.

    Parameters
    ----------
    device : str
        ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or ``"gpu"``.

    Returns
    -------
    jax.Device
        The jax CPU device, or the first (or N-th) jax GPU device.
    """
    import jax

    kind, _, index = device.partition(":")
    return jax.devices("gpu" if kind == "cuda" else kind)[int(index or 0)]


def _lens_param_index(lens):
    """Map every non-pointer parameter of a caustics lens to its ``"<module.name>.<child key>"`` key.

    Parameters
    ----------
    lens : caustics.lenses.base.Lens
        A caustics lens (e.g. a ``SinglePlane``).

    Returns
    -------
    index : dict
        A dictionary mapping each key to its ``caskade.Param``.

    Raises
    ------
    ValueError
        If two different parameters share the same key, e.g. two modules with the same
        ``name=``.
    """
    from caskade import Module, Param

    index = {}
    for node in lens.topological_ordering():
        if isinstance(node, Module):
            for key, child in node.children.items():
                if isinstance(child, Param) and not child.pointer:
                    full_key = f"{node.name}.{key}"
                    existing = index.get(full_key)
                    if existing is not None and existing is not child:
                        raise ValueError(
                            f"Two different parameters are both named '{full_key}'. Give the "
                            "modules that own them distinct `name=` values."
                        )
                    index.setdefault(full_key, child)
    return index


def _inside_caustics(curves, points):
    """Test which source-plane points lie inside the caustics or pseudo-caustics.

    A point is inside when its winding number about at least one curve's source polyline is
    nonzero. This is the rule ``caustic_area`` uses for the area of the multiply imaged region.
    Each winding-number call holds a few (B, curve points) temporaries, so B bounds the memory.

    Parameters
    ----------
    curves : caustics.lenses.func.adaptive.CriticalCurvesAndCaustics
        The curves of a closed lens mesh.
    points : array
        A (B, 2) float64 array of caustics' active backend: source-plane positions in
        arcseconds, on the device of ``curves.source``.

    Returns
    -------
    inside : numpy.ndarray
        A (B,) bool array, True where a point is inside.
    """
    from caustics.lenses.func.adaptive.geometry import winding_number
    from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh

    # winding_number runs on caustics' mesh backend (torch, also under jax), so it gets copies there.
    source, offsets, points = to_mesh((curves.source, curves.offsets, points))
    inside = mesh_backend.zeros(
        (points.shape[0],), dtype=mesh_backend.bool, device=mesh_backend.device(points)
    )
    for first, stop in pairwise(offsets.tolist()):
        inside |= winding_number(source[first:stop], points) != 0
    return mesh_backend.to_numpy(inside)


def _reemit(w):
    """Re-emit a recorded warning unchanged.

    A recorded warning keeps only its filename, so ``warn_explicit`` is also given the name of the
    loaded module that owns the file, as the original ``warn`` call was; module-scoped filters then
    still apply. When no loaded module owns it, ``module`` is left out: CPython drops a warning
    whose module is given as None.
    """
    modules = list(sys.modules.values())
    name = next((m.__name__ for m in modules if getattr(m, "__file__", None) == w.filename), None)
    module = {} if name is None else {"module": name}
    warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source, **module)


def _describe_system(system_index, values):
    """Return ``"system <index> (<name>=<value>, ...)"``, naming a system in error messages."""
    inputs = ", ".join(f"{name}={value}" for name, value in values.items())
    return f"system {system_index} ({inputs})"


class CausticsStrongLensSystem(FunctionNode):
    """A sampling node that produces strong-lens image arrays with caustics.

    For each sampled lens system this node uses `caustics
    <https://github.com/Ciela-Institute/caustics>`_ to build a closed lens mesh, draw a
    source uniformly inside the region of the source plane where it is strongly lensed,
    raytrace it to its images, and evaluate the per-image observables. Its outputs plug
    directly into ``ResolvedStrongLensModel`` (see the example below); nothing in
    lightcurvelynx consumes convergence or shear yet, but they are evaluated now in
    preparation for future microlensing work.

    Every offset (``source_x``/``source_y``, ``image_x``/``image_y``) is
    caustics' ``(x, y)``: east and north in arcseconds from the lens origin, the same
    convention ``ResolvedStrongLensModel`` expects.

    By default the source is drawn uniformly inside the union of the caustics and 
    pseudo-caustics (every point there is multiply imaged). If ``mu_min`` is given, it is 
    instead drawn uniformly inside the source-plane region where ``mu_min <= mu_total <= mu_max`` 
    (singly imaged sources are then legitimate). ``area`` is the area (in in ``arcsec^2``) 
    of whichever region was sampled from (``caustic_area(curves)``, or ``region.area`` in μ-region mode), 
    so a population can be weighted by lensing cross-section. A limitation: the region contours 
    that caustics traces can misestimate the area of ``mu_min <= mu_total <= mu_max`` near 
    the lensing caustics, by about 1% at ``mu_min = 10`` and roughly 5-30% for ``mu_min`` 
    from about 30to 300 on an SIE + external shear test lens. The sampled sources still 
    follow the true ``mu_total``, since each candidate is accepted by caustics' 
    ``total_magnification``; caustic mode is unaffected.

    Candidate sources are drawn uniformly in the axis-aligned bounding box of the sampled
    region's contour, ``candidate_batch_size`` at a time, and the first one inside (in draw
    order) is accepted; ``n_draws`` counts the candidates drawn up to and including it, and
    ``bb_area`` is the box's area. At most ``max_candidates`` are drawn. For a system whose
    ``estimate_mc_area`` is true, all ``max_candidates`` are drawn and tested, and
    ``mc_area = bb_area * accepted / max_candidates`` estimates the sampled region's area,
    with standard error ``sqrt(mc_area * (bb_area - mc_area) / max_candidates)``, as a
    cross-check of ``area`` (above all in μ-region mode). It sees only the part of the region
    inside the box, so it cannot catch a traced contour that misses part of the region beyond
    the contour's bounding box. The source drawn and ``n_draws`` do not depend on
    ``estimate_mc_area``. Like ``fov``, ``estimate_mc_area`` is a node parameter, so it can
    be a constant or a link to another node's output, e.g. a ``FunctionNode`` that returns
    whether a sampled lens parameter exceeds a threshold, and it is evaluated per system.

    **Outputs** (one entry per system; ``I = max_images``):

    ===================== ========== ========= ==============================================
    Output                One system S systems  Meaning
    ===================== ========== ========= ==============================================
    source_x, source_y    scalar     (S,)       Source offset (east, north) from the lens
                                                 origin, in arcsec.
    num_images            int        (S,)       Number of active leading image entries.
    image_x, image_y      (I,)       (S, I)     Image offsets (east, north), in arcsec;
                                                 NaN padding.
    macro_magnifications  (I,)       (S, I)     ``|μ|``, dimensionless; 0 padding.
    time_delays           (I,)       (S, I)     Observer-frame days; earliest image is 0;
                                                 NaN padding.
    convergences          (I,)       (S, I)     κ, dimensionless; NaN padding.
    shear_1, shear_2      (I,)       (S, I)     γ1 = (ψ_xx - ψ_yy) / 2, γ2 = ψ_xy, with ψ the
                                                 lensing potential; dimensionless; NaN padding.
    area                  scalar     (S,)       The sampled region's area, in ``arcsec^2``.
    bb_area               scalar     (S,)       The area of the box the candidate sources are
                                                 drawn in, in ``arcsec^2``.
    mc_area               scalar     (S,)       Monte Carlo estimate of ``area``, in
                                                 ``arcsec^2``; NaN unless ``estimate_mc_area``.
    n_draws               int        (S,)       Candidate sources drawn up to and including the
                                                 accepted one.
    ===================== ========== ========= ==============================================

    Within a system the images are sorted by arrival time (a stable sort), so
    ``image_id`` in ``ResolvedStrongLensModel`` matches this order; an image's parity is
    the sign of ``(1 - κ)**2 - γ1**2 - γ2**2``.

    Construction-time input errors and a per-system ``z_l >= z_s`` raise ``ValueError``. 
    Every other per-system failure raises ``RuntimeError`` naming the system 
    (its global index and parameter values): the mesh failing to close, depth-limited mesh 
    refinement, an unbounded μ-region, a hole curve hitting its sample cap, no 
    strong-lensing region or an empty μ-region, a source outside the image of the
    fov boundary (too few images in the fov), too few images for the sampling mode, or
    more than ``max_images`` images. Other warnings from the mesh build (e.g. a
    fov-widening notice) are re-emitted unchanged.

    The node runs, in float64, on caustics' active backend: torch or jax.
    Caustics picks it once per process, at import, from the ``CASKADE_BACKEND``
    environment variable (default ``"torch"``), so set it in the environment before
    caustics is imported, e.g. ``CASKADE_BACKEND=jax python run.py``; worker processes
    then inherit it. The node records the backend at construction, and sampling in a
    process that runs the other one raises ``ValueError``. Under jax, the mesh build, the
    raytrace and the observables run as compiled programs; caustics' mesh bookkeeping
    still runs on torch, on the CPU.

    The first ``compute`` of a node in a process deep-copies its stored
    lens to ``device`` and wraps this working lens's ``magnification``, ``time_delay``,
    ``convergence`` and ``shear`` with ``caustics.jit_forward``. Both are kept in a
    process-local registry, keyed by a token made when the node is constructed, for the
    life of the process. Every copy of the node, such as the one an executor unpickles
    for each batch, carries the token and reuses them, and with them jax's compiled
    programs. So under any executor each worker process compiles once per constructed
    node, device and padded input shape, not once per batch. Two separately constructed
    nodes compile separately, even with identical lenses. Parameter values are passed on
    every call, so none carry over between systems. The observables see the images padded
    to ``max_images`` (by repeating the first image), so their input shape never changes.
    Neither the working lens nor the wrappers are pickled.

    All of a system's caustics work (mesh build, source sampling, raytracing,
    observables) runs under one module-level lock, because every call fills the parameters
    of the shared working lens, and under torch ``torch.func.jacfwd``, which caustics uses
    internally, keeps process-global forward-AD state and is not thread-safe. Under a
    ``ThreadPoolExecutor`` this node's work is therefore serialized across threads:
    correct, but no faster than running serially. Prefer a process-based executor
    (``num_jobs`` under torch, ``ProcessPoolExecutor``, Dask, loky, Ray) for speed.

    Pickling this node (e.g. for ``num_jobs``) requires ``cloudpickle``, and
    unpickling it in a worker does too, because the pickled state holds the stored lens
    serialized by ``cloudpickle.dumps``; install ``cloudpickle`` on every worker.

    If ``torch_num_threads`` is an int, it is applied with ``torch.set_num_threads`` 
    at the start of sampling, a process-wide side effect, under either backend. If it is 
    left as ``None`` (the default), sampling leaves torch's thread count alone in the 
    process that built the node, but uses 1 thread in any other process (a worker process), 
    which avoids a fork deadlock after multi-threaded torch use in the parent and avoids 
    oversubscription under many workers; pass an explicit int for a cluster worker that 
    owns many cores. ``device`` defaults to ``"cpu"``. The node moves its working lens 
    to ``device`` only inside sampling, never in the process that builds it. Under torch, 
    caustics' mesh bookkeeping follows ``device`` through a ``torch.device`` context. 
    ``import caustics`` itself probes CUDA, though, so a forked worker cannot use CUDA 
    ("Cannot re-initialize CUDA in forked subprocess"); ``num_jobs`` and a default 
    ``ProcessPoolExecutor`` fork on Linux with Python 3.13 or earlier. For ``device="cuda"`` 
    with a process-based executor, pass ``simulate_lightcurves`` an executor with a spawn 
    or forkserver context, e.g. ``ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))``, 
    or use Dask, loky or Ray, or set ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` before importing 
    torch. Under jax, ``"cpu"`` is jax's CPU device and ``"cuda"``, ``"cuda:N"`` or ``"gpu"`` 
    its first (or N-th) GPU, made jax's default device during sampling. Forked workers are 
    unsafe under jax on any device, since jax is multithreaded: use an executor with a spawn or
    forkserver context, as above, or Dask, loky or Ray, and not ``num_jobs`` where it
    forks. GPU speedup is unmeasured, since caustics handles one system per call.

    Example wiring this node into ``ResolvedStrongLensModel``:

    .. code-block:: python

        system = CausticsStrongLensSystem(
            lens, lens_parameters=lens_parameters, z_s=source.redshift,
            fov=5.0, init_res=32, min_img_sep=1e-2, max_images=5,
        )
        lensed = ResolvedStrongLensModel(source, **system.resolved_inputs())

    Parameters
    ----------
    lens : caustics.lenses.base.Lens
        A caustics lens (e.g. ``SinglePlane([SIE, ExternalShear])``). It is never modified;
        the node works on its own copy.
    lens_parameters : dict
        Maps a ``"<module.name>.<param>"`` key to a node setter (a constant or a link to
        another node's output). Each key becomes a node parameter named after it with ``.``
        replaced by ``_`` (e.g. ``"sie.q"`` becomes ``system.sie_q``). A lens parameter that
        already has a value on ``lens`` need not be mapped; a mapped key overrides such a
        value. ``"<lens.name>.z_s"`` may not be given here; it comes from ``z_s``.
    z_s : parameter
        The source redshift, usually the source model's ``redshift``.
    fov : parameter
        The side length (arcsec) of the starting square mesh domain.
    init_res : parameter
        The number of level-0 mesh cells per axis.
    min_img_sep : parameter
        The minimum image separation (arcsec) the mesh must resolve.
    max_images : int
        The fixed padded width of the per-image outputs. Must be at least 1.
    mu_min : float, optional
        The lower bound of the total-magnification sampling region. If given, the source is
        drawn uniformly where ``mu_min <= mu_total <= mu_max`` instead of inside the caustics.
    mu_max : float, optional
        The upper bound of the total-magnification sampling region. Required when ``mu_min``
        is given, and must be greater than ``mu_min``.
    max_candidates : int, optional
        The most candidate sources drawn per system before sampling gives up, and the number
        drawn when ``estimate_mc_area`` is set. Must be at least 1. Default: 16384.
    candidate_batch_size : int, optional
        The number of candidate sources drawn, and tested in one call, at a time. In caustic
        mode each call holds a few (candidate_batch_size, caustic polyline points) float64
        temporaries, so this bounds the memory. Must be at least 1. Default: 1024.
    estimate_mc_area : parameter, optional
        Whether a system draws and tests all ``max_candidates`` candidates and reports
        ``mc_area``; where false, sampling stops at the batch holding the first accepted
        candidate and ``mc_area`` is NaN. A bool, or a link to another node's output, evaluated
        per system. Default: False.
    mesh_kwargs : dict, optional
        Extra keyword arguments passed to ``lens.build_closed_lens_mesh``. May not set
        ``fov``, ``init_res``, ``min_img_sep``, ``mu_min``, ``mu_max``, ``device`` or
        ``params``, which the node sets itself.
    raytrace_kwargs : dict, optional
        Extra keyword arguments passed to ``lens.forward_raytrace``. May not set ``mesh`` or
        ``params``, which the node sets itself.
    device : str, optional
        The device for caustics calls: a torch device under torch, and ``"cpu"``, ``"cuda"``,
        ``"cuda:N"`` or ``"gpu"`` under jax. Default: "cpu".
    torch_num_threads : int, optional
        If given, ``torch.set_num_threads`` is called with this value at the start of
        ``compute``, under either backend. Default: None, which uses 1 thread in a worker
        process (any process other than the one that built the node) and leaves torch's
        thread count alone in the building process.
    seed : int, optional
        The seed for this node's random number generator. Default: None (OS entropy).
    node_label : str, optional
        The label for this node.
    """

    OUTPUTS = (
        "source_x",
        "source_y",
        "num_images",
        "image_x",
        "image_y",
        "macro_magnifications",
        "time_delays",
        "convergences",
        "shear_1",
        "shear_2",
        "area",
        "bb_area",
        "mc_area",
        "n_draws",
    )

    def __init__(
        self,
        lens,
        *,
        lens_parameters,
        z_s,
        fov,
        init_res,
        min_img_sep,
        max_images,
        mu_min=None,
        mu_max=None,
        max_candidates=16384,
        candidate_batch_size=1024,
        estimate_mc_area=False,
        mesh_kwargs=None,
        raytrace_kwargs=None,
        device="cpu",
        torch_num_threads=None,
        seed=None,
        node_label=None,
    ):
        _, caustics = _import_caustics()

        if max_images < 1:
            raise ValueError(f"max_images must be at least 1, got {max_images}.")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be at least 1, got {max_candidates}.")
        if candidate_batch_size < 1:
            raise ValueError(f"candidate_batch_size must be at least 1, got {candidate_batch_size}.")
        if mu_max is not None and mu_min is None:
            raise ValueError("mu_min must be given whenever mu_max is given.")
        if mu_min is not None and mu_max is not None and mu_max <= mu_min:
            raise ValueError(f"mu_max ({mu_max}) must be greater than mu_min ({mu_min}).")

        mesh_kwargs = {} if mesh_kwargs is None else dict(mesh_kwargs)
        raytrace_kwargs = {} if raytrace_kwargs is None else dict(raytrace_kwargs)
        reserved_mesh_keys = {"fov", "init_res", "min_img_sep", "mu_min", "mu_max", "device", "params"}
        bad_mesh_keys = sorted(reserved_mesh_keys & mesh_kwargs.keys())
        if bad_mesh_keys:
            raise ValueError(f"mesh_kwargs may not set {bad_mesh_keys}; the node sets these itself.")
        reserved_raytrace_keys = {"mesh", "params"}
        bad_raytrace_keys = sorted(reserved_raytrace_keys & raytrace_kwargs.keys())
        if bad_raytrace_keys:
            raise ValueError(f"raytrace_kwargs may not set {bad_raytrace_keys}; the node sets these itself.")

        index = _lens_param_index(lens)
        z_s_key = f"{lens.name}.z_s"
        if z_s_key in lens_parameters:
            raise ValueError(f"lens_parameters may not include '{z_s_key}'; z_s comes from the z_s argument.")

        unknown_keys = sorted(set(lens_parameters) - set(index))
        if unknown_keys:
            raise ValueError(
                f"Unknown lens parameter key(s) {unknown_keys}. Available keys: {sorted(index)}."
            )

        unset_keys = sorted(
            key
            for key, p in index.items()
            if p.value is None and key not in lens_parameters and key != z_s_key
        )
        if unset_keys:
            raise ValueError(
                f"Lens parameter(s) {unset_keys} are unset and not mapped in lens_parameters. "
                f"Available keys: {sorted(index)}."
            )

        reserved_names = {"z_s", "fov", "init_res", "min_img_sep", "estimate_mc_area"} | set(self.OUTPUTS)
        node_inputs, key_of = {}, {}
        for key, setter in lens_parameters.items():
            name = key.replace(".", "_")
            if name in reserved_names:
                raise ValueError(
                    f"Mapped key '{key}' collides with the reserved node parameter name '{name}'."
                )
            if name in key_of:
                raise ValueError(
                    f"Mapped keys '{key_of[name]}' and '{key}' both become the node parameter name '{name}'."
                )
            key_of[name] = key
            node_inputs[name] = setter
        node_inputs.update(
            z_s=z_s, fov=fov, init_res=init_res, min_img_sep=min_img_sep, estimate_mc_area=estimate_mc_area
        )

        z_l_key = f"{lens.name}.z_l"
        if z_l_key not in index:
            z_l_name, z_l_value = None, None
        elif z_l_key in lens_parameters:
            z_l_name, z_l_value = z_l_key.replace(".", "_"), None
        else:
            z_l_name, z_l_value = None, float(index[z_l_key].value)

        super().__init__(self._non_func, outputs=list(self.OUTPUTS), node_label=node_label, **node_inputs)

        mapped_keys = set(lens_parameters) | {z_s_key}
        stored_lens = copy.deepcopy(lens)
        names = {}
        for key, p in _lens_param_index(stored_lens).items():
            if key in mapped_keys:
                p.to_dynamic()
            else:
                p.to_static()
            names[id(p)] = "z_s" if key == z_s_key else key.replace(".", "_")
        # caskade's Module.to casts in place, so the parameters keep their identities.
        self._lens = stored_lens.to(dtype=caustics.backend.float64)
        self._param_order = [names[id(p)] for p in self._lens.dynamic_params]

        self._z_l_name = z_l_name
        self._z_l_value = z_l_value
        self._max_images = int(max_images)
        self._mu_min = mu_min
        self._mu_max = mu_max
        self._max_candidates = int(max_candidates)
        self._candidate_batch_size = int(candidate_batch_size)
        self._mesh_kwargs = mesh_kwargs
        self._raytrace_kwargs = raytrace_kwargs
        self._device = device
        self._backend = caustics.backend.backend
        self._torch_num_threads = torch_num_threads
        self._owner_pid = os.getpid()
        # Copies of this node, pickled with it, share this token and so one working lens per process.
        self._lens_key = uuid.uuid4().hex
        self._working = None
        self._observables = None
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        self._rng = np.random.default_rng(seed)

    def resolved_inputs(self):
        """Return this node's outputs mapped to ``ResolvedStrongLensModel``'s input names.

        Returns
        -------
        dict
            Keyword arguments ready to pass to ``ResolvedStrongLensModel``.
        """
        return {
            "source_x": self.source_x,
            "source_y": self.source_y,
            "image_x": self.image_x,
            "image_y": self.image_y,
            "macro_magnifications": self.macro_magnifications,
            "time_delays": self.time_delays,
            "num_images": self.num_images,
        }

    def __getstate__(self):
        """Return the pickling state, with the stored lens serialized by cloudpickle.

        A caustics ``SinglePlane`` does not pickle with the standard library ``pickle``. The
        working lens and its jitted observables are left out: an unpickled copy takes them from
        its process's registry, or makes them there, on its first ``compute``.

        Returns
        -------
        state : dict
            This node's ``__dict__``, with ``_lens`` replaced by its ``cloudpickle.dumps``
            bytes, and the working lens and its jitted observables by None.

        Raises
        ------
        ImportError
            If cloudpickle is not installed.
        """
        try:
            import cloudpickle
        except ImportError as err:
            raise ImportError(
                "Pickling CausticsStrongLensSystem (e.g. for num_jobs) requires cloudpickle: "
                "pip install cloudpickle"
            ) from err
        state = self.__dict__.copy()
        state["_lens"] = cloudpickle.dumps(self._lens)
        state["_working"] = state["_observables"] = None
        return state

    def __setstate__(self, state):
        """Restore the pickling state, loading the stored lens from its cloudpickle bytes.

        The call itself is ``pickle.loads``, but the bytes were produced by
        ``cloudpickle.dumps`` and reference it, so cloudpickle must also be importable
        wherever this runs (e.g. on every worker of a process-based executor).

        Parameters
        ----------
        state : dict
            A state as returned by ``__getstate__``: this node's ``__dict__``, with
            ``_lens`` as ``cloudpickle.dumps`` bytes.
        """
        self.__dict__.update(state)
        if isinstance(self._lens, bytes):
            self._lens = pickle.loads(self._lens)

    def _working_lens(self, device):
        """Return this process's working copy of the stored lens, making it on first use.

        The working lens and its observables, wrapped by ``caustics.jit_forward``, are made once per
        process for every copy of this node and kept in ``_WORKING_LENSES``; this copy holds them in
        ``_working`` and ``_observables``. ``compute`` calls this under the module lock. The stored
        lens is never modified.

        Parameters
        ----------
        device : str or jax.Device
            ``device`` in the form caustics' active backend takes.

        Returns
        -------
        lens : caustics.lenses.base.Lens
            The working lens, on ``device``.
        """
        if self._working is None:
            key = (self._lens_key, self._backend, str(device))
            if key not in _WORKING_LENSES:
                from caustics import jit_forward

                lens = copy.deepcopy(self._lens).to(device=device)
                observables = {
                    name: jit_forward(getattr(lens, name))
                    for name in ("magnification", "time_delay", "convergence", "shear")
                }
                _WORKING_LENSES[key] = lens, observables
            self._working, self._observables = _WORKING_LENSES[key]
        return self._working

    def _set_torch_threads(self, torch):
        """Apply ``torch_num_threads``, or 1 thread in a process other than the one that built the node.

        In a forked worker, the first parallel torch op can deadlock if the parent already ran
        multi-threaded torch work; one thread avoids it.

        Parameters
        ----------
        torch : module
            The ``torch`` module.
        """
        num_threads = self._torch_num_threads
        if num_threads is None:
            if os.getpid() == self._owner_pid:
                return
            num_threads = 1
        if torch.get_num_threads() != num_threads:
            torch.set_num_threads(num_threads)

    def _build(self, lens, values, params, device, system_index):
        """Build one system's closed lens mesh, raising on caustics' failure warnings.

        The build runs with its warnings recorded (``compute`` holds the module lock around it).
        Failure warnings are raised as one error; every other warning is re-emitted unchanged.

        Parameters
        ----------
        lens : caustics.lenses.base.Lens
            The working lens, on ``device``.
        values : dict
            This system's node input values, keyed by node parameter name.
        params : list of array
            The values of the lens's dynamic parameters, in ``lens.dynamic_params`` order.
        device : str or jax.Device
            ``device`` in the form caustics' active backend takes.
        system_index : int
            The global index of the system, used in error messages.

        Returns
        -------
        mesh : caustics.lenses.func.adaptive.LensMesh
            The closed lens mesh.
        curves : caustics.lenses.func.adaptive.CriticalCurvesAndCaustics
            Its critical curves, caustics and pseudo-caustics.
        region : caustics.lenses.func.adaptive.MagnificationRegion or None
            The μ-region, or None when ``mu_min`` is not given.

        Raises
        ------
        RuntimeError
            If caustics warns that the mesh did not close, the μ-region is unbounded,
            refinement was depth-limited, or a hole curve hit its sample cap.
        """
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            built = lens.build_closed_lens_mesh(
                float(values["fov"]),
                int(values["init_res"]),
                float(values["min_img_sep"]),
                mu_min=self._mu_min,
                mu_max=self._mu_max,
                device=device,
                params=params,
                **self._mesh_kwargs,
            )
        failures = []
        for w in records:
            message = str(w.message)
            if any(text in message for text in _FAILURE_MESSAGES):
                failures.append(message)
            else:
                _reemit(w)
        if failures:
            raise RuntimeError(
                f"{_describe_system(system_index, values)}: caustics could not build a valid lens "
                "mesh; adjust fov, init_res or min_img_sep (or max_depth / max_iters in mesh_kwargs). "
                f"Caustics warned: {' | '.join(failures)}"
            )
        return built

    def _sample_source(self, mesh, curves, region, rng, estimate_mc_area):
        """Draw a source uniformly inside the caustics, or inside the μ-region, by rejection sampling.

        Candidates are drawn uniformly in the bounding box of the caustics' source polylines (or,
        in μ-region mode, of ``region.source``), in batches of ``candidate_batch_size``, each tested
        in one call, up to ``max_candidates`` in all; the first one inside (in draw order) is
        accepted. Sampling stops at the batch holding it, unless ``estimate_mc_area`` is true: then
        all ``max_candidates`` are drawn and tested, and the fraction inside estimates the area.

        Parameters
        ----------
        mesh : caustics.lenses.func.adaptive.LensMesh
            The closed lens mesh.
        curves : caustics.lenses.func.adaptive.CriticalCurvesAndCaustics
            The mesh's critical curves, caustics and pseudo-caustics.
        region : caustics.lenses.func.adaptive.MagnificationRegion or None
            The μ-region, or None when ``mu_min`` is not given.
        rng : numpy.random.Generator
            This system's random number generator.
        estimate_mc_area : bool
            This system's ``estimate_mc_area``: whether to draw and test all ``max_candidates``
            candidates and estimate the sampled region's area from them.

        Returns
        -------
        source : dict or None
            ``source_x`` and ``source_y``, the accepted source offset (east, north) from the lens
            origin in arcseconds; ``n_draws``, the number of candidates drawn up to and including
            it; ``bb_area``, the bounding box's area in arcsec^2; and ``mc_area``, the Monte Carlo
            estimate of the sampled region's area in arcsec^2 (NaN unless ``estimate_mc_area``).
            None if none of the ``max_candidates`` candidates is accepted.
        """
        from caustics import backend

        if region is not None:
            from caustics.lenses.func.adaptive import total_magnification

            source = region.source
            upper = np.inf if self._mu_max is None else self._mu_max

            def accept(points):
                mu_tot = backend.to_numpy(total_magnification(points[:, 0], points[:, 1], mesh))
                return (mu_tot >= self._mu_min) & (mu_tot <= upper)
        else:
            source = curves.source

            def accept(points):
                return _inside_caustics(curves, points)

        corners = backend.to_numpy(source)
        low, high = corners.min(axis=0), corners.max(axis=0)
        drawn, num_inside = None, 0
        for start in range(0, self._max_candidates, self._candidate_batch_size):
            size = min(self._candidate_batch_size, self._max_candidates - start)
            candidates = rng.uniform(low, high, size=(size, 2))
            points = backend.as_array(candidates, dtype=backend.float64, device=backend.device(source))
            inside = accept(points)
            num_inside += int(inside.sum())
            if drawn is None and inside.any():
                first = int(np.argmax(inside))
                drawn = {
                    "source_x": float(candidates[first, 0]),
                    "source_y": float(candidates[first, 1]),
                    "n_draws": start + first + 1,
                }
                if not estimate_mc_area:
                    break
        if drawn is None:
            return None
        drawn["bb_area"] = float(np.prod(high - low))
        drawn["mc_area"] = (
            drawn["bb_area"] * num_inside / self._max_candidates if estimate_mc_area else np.nan
        )
        return drawn

    def _solve_system(self, lens, values, rng, device, system_index):
        """Build one system's mesh, draw its source, and solve and evaluate its images.

        Parameters
        ----------
        lens : caustics.lenses.base.Lens
            The working lens, on ``device``.
        values : dict
            This system's node input values, keyed by node parameter name.
        rng : numpy.random.Generator
            This system's random number generator.
        device : str or jax.Device
            ``device`` in the form caustics' active backend takes.
        system_index : int
            The global index of the system, used in error messages.

        Returns
        -------
        row : dict
            One entry per name in ``OUTPUTS``: scalars for ``source_x``, ``source_y``,
            ``num_images``, ``area``, ``bb_area``, ``mc_area`` and ``n_draws``, and (max_images,)
            float64 arrays, sorted by arrival time and padded, for the per-image outputs.

        Raises
        ------
        ValueError
            If ``z_l >= z_s``.
        RuntimeError
            If the mesh build fails (see ``_build``); the sampled region (the strong-lensing
            region, or the μ-region) has an area that is not finite or not positive, or none of
            the candidate sources drawn for it is accepted; the source lies outside the image of
            the fov boundary; or the source has more than ``max_images`` images, or fewer than
            the sampling region guarantees (2 in caustic mode, 1 in μ-region mode).
        """
        from caustics import backend
        from caustics.lenses.func.adaptive import caustic_area
        from caustics.lenses.func.adaptive.lens_mesh import inside_fov_image
        from caustics.lenses.func.adaptive.mesh_backend import to_mesh

        z_l = values[self._z_l_name] if self._z_l_name is not None else self._z_l_value
        if z_l is not None and z_l >= values["z_s"]:
            raise ValueError(
                f"{_describe_system(system_index, values)}: z_l ({z_l}) must be less than "
                f"z_s ({values['z_s']})."
            )

        params = [
            backend.as_array(values[name], dtype=backend.float64, device=device) for name in self._param_order
        ]
        mesh, curves, region = self._build(lens, values, params, device, system_index)
        if region is not None:
            area = float(region.area)
            sampled = f"mu-region ({region.mu_min} <= mu_tot <= {region.mu_max})"
            empty = "NaN when it is unbounded, 0 when mu_tot never reaches mu_min"
        else:
            area = float(caustic_area(curves))
            sampled = "strong-lensing region"
            empty = "NaN when the fov cuts a critical curve, 0 when the lens forms none"
        if not np.isfinite(area) or area <= 0.0:
            raise RuntimeError(
                f"{_describe_system(system_index, values)} has no {sampled}: its area is {area} "
                f"arcsec^2 ({empty})."
            )
        source = self._sample_source(mesh, curves, region, rng, bool(values["estimate_mc_area"]))
        if source is None:
            raise RuntimeError(
                f"{_describe_system(system_index, values)}: none of {self._max_candidates} candidate "
                f"sources drawn in the bounding box of its {sampled} lies inside it."
            )
        source_x, source_y = source["source_x"], source["source_y"]

        # forward_raytrace raises for a source outside the image of the fov boundary, where some
        # of its images lie outside the fov; check first so the error names the system.
        beta = backend.as_array([[source_x, source_y]], dtype=backend.float64, device=device)
        # inside_fov_image runs on caustics' mesh backend (torch, also under jax), so it gets copies there.
        if not bool(inside_fov_image(*to_mesh((mesh, beta)))[0]):
            raise RuntimeError(
                f"{_describe_system(system_index, values)}: too few images in the fov: the source lies "
                "outside the image of the fov boundary, so some of its images lie outside the fov; the "
                "starting fov may lie inside the critical curves, so try a larger fov."
            )
        x, y, counts = lens.forward_raytrace(
            beta[:, 0], beta[:, 1], mesh=mesh, params=params, **self._raytrace_kwargs
        )
        num_images = int(counts[0])
        if num_images > self._max_images:
            raise RuntimeError(
                f"{_describe_system(system_index, values)}: the source has {num_images} images, "
                f"more than max_images={self._max_images}; raise max_images."
            )
        min_images = 1 if self._mu_min is not None else 2
        if num_images < min_images:
            raise RuntimeError(
                f"{_describe_system(system_index, values)}: too few images ({num_images}, expected "
                f"at least {min_images}); the starting fov may lie inside the critical curves, so "
                "try a larger fov."
            )
        x, y = backend.to_numpy(x), backend.to_numpy(y)
        # Pad to max_images by repeating the first image, so the observables always see one input
        # shape (one compiled program each under jax) and the padding is finite; sliced off below.
        pad = np.arange(self._max_images)
        pad[num_images:] = 0
        x_pad, y_pad = (backend.as_array(v[pad], dtype=backend.float64, device=device) for v in (x, y))
        jitted = self._observables
        shear_1, shear_2 = jitted["shear"](x_pad, y_pad, params=params)
        per_image = {
            "macro_magnifications": jitted["magnification"](x_pad, y_pad, params=params),
            "time_delays": jitted["time_delay"](x_pad, y_pad, params=params),
            "convergences": jitted["convergence"](x_pad, y_pad, params=params),
            "shear_1": shear_1,
            "shear_2": shear_2,
        }
        per_image = {"image_x": x, "image_y": y} | {
            name: backend.to_numpy(value)[:num_images] for name, value in per_image.items()
        }
        order = np.argsort(per_image["time_delays"], kind="stable")
        per_image["time_delays"] = per_image["time_delays"] - per_image["time_delays"][order[0]]

        row = {"num_images": num_images, "area": area} | source
        for name, value in per_image.items():
            row[name] = np.full(self._max_images, 0.0 if name == "macro_magnifications" else np.nan)
            row[name][:num_images] = value[order]
        return row

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Build each system's mesh, draw its source, and save its images and observables.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator. It advances by
            exactly one draw per system.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        list
            The values of the outputs, in ``OUTPUTS`` order.

        Raises
        ------
        ValueError
            If caustics runs a backend other than the one the node was built under, or a
            system has ``z_l >= z_s``.
        RuntimeError
            If a system fails otherwise (see the failure policy in the class docstring).
        """
        torch, caustics = _import_caustics()

        if caustics.backend.backend != self._backend:
            raise ValueError(
                f"This CausticsStrongLensSystem was built under caustics' '{self._backend}' backend, but "
                f"caustics runs its '{caustics.backend.backend}' backend in this process. Set "
                f"CASKADE_BACKEND={self._backend} in the environment, before caustics is imported, so "
                "that worker processes inherit it."
            )
        self._set_torch_threads(torch)
        args = self._build_inputs(graph_state, **kwargs)
        rng = rng_info if rng_info is not None else self._rng
        num_samples = graph_state.num_samples
        sub_seeds = rng.integers(0, 2**63, size=num_samples, dtype=np.uint64)

        if self._backend == "jax":
            import jax

            # Caustics' mesh bookkeeping stays on torch's CPU under jax.
            device = _jax_device(self._device)
            default_device = jax.default_device(device)
        else:
            # torch.device also moves caustics' mesh bookkeeping, which runs on torch's default device
            # (geometry.build_device()), onto device.
            device = self._device
            default_device = torch.device(device)
        sample_offset = graph_state.sample_offset or 0
        rows = []
        with _CAUSTICS_LOCK, default_device:
            lens = self._working_lens(device)
            for i, sub_seed in enumerate(sub_seeds):
                values = {name: value if num_samples == 1 else value[i] for name, value in args.items()}
                rng_i = np.random.default_rng(sub_seed)
                rows.append(self._solve_system(lens, values, rng_i, device, sample_offset + i))

        results = [np.array([row[name] for row in rows]) for name in self.OUTPUTS]
        if num_samples == 1:
            results = [value[0] for value in results]
        self._save_results(results, graph_state)
        return results
