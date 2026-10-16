"""A sampling node that uses caustics to build a strong-lens mesh for a caustics lens model,
draw a source inside its strong-lensing region, raytrace it to images, and evaluate the
per-image observables that ``ResolvedStrongLensModel`` consumes.
"""

import copy
import os
import threading
import warnings
from itertools import pairwise

import numpy as np

from lightcurvelynx.base_models import FunctionNode

# Points per winding-number call. Each call holds a few (points, curve points, 2) float64
# temporaries, so this bounds the memory; it also sets how many candidates are tested at once.
_WINDING_CHUNK = 128
# Candidate sources per rng draw, and the number of draws before the rejection sampler gives up.
_CANDIDATE_BATCH = 1024
_MAX_CANDIDATE_BATCHES = 100
# Substrings of the caustics warnings that mark a failed build: the mesh did not close or the
# μ-region is unbounded (lens_mesh.py, region.py), refinement was depth-limited (lattice.py), or
# a hole curve hit its sample cap (holes.py). Some are emitted with stacklevel=3, which hides the
# emitting module, so all are recognized by text.
_FAILURE_MESSAGES = ("extension(s), at fov=", "is depth-limited", "before every chord fell below")
# Serializes the warning capture around mesh builds: catch_warnings mutates process-global state.
_BUILD_LOCK = threading.Lock()


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
            "CausticsStrongLensSystem requires torch and caustics (with the adaptive mesh API). "
            "Install caustics from its feat/adaptive_refactor branch."
        ) from err
    return torch, caustics


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

    Parameters
    ----------
    curves : caustics.lenses.func.adaptive.CriticalCurvesAndCaustics
        The curves of a closed lens mesh.
    points : torch.Tensor
        A (B, 2) float64 tensor of source-plane positions in arcseconds, on the device of
        ``curves.source``.

    Returns
    -------
    inside : torch.Tensor
        A (B,) bool tensor, True where a point is inside.
    """
    import torch
    from caustics.lenses.func.adaptive.geometry import winding_number

    bounds = curves.offsets.tolist()
    inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for start in range(0, points.shape[0], _WINDING_CHUNK):
        chunk = slice(start, start + _WINDING_CHUNK)
        for first, stop in pairwise(bounds):
            inside[chunk] |= winding_number(curves.source[first:stop], points[chunk]) != 0
    return inside


def _describe_system(system_index, values):
    """Return ``"system <index> (<name>=<value>, ...)"``, naming a system in error messages."""
    inputs = ", ".join(f"{name}={value}" for name, value in values.items())
    return f"system {system_index} ({inputs})"


class CausticsStrongLensSystem(FunctionNode):
    """A sampling node that produces strong-lens image arrays with caustics.

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
    mesh_kwargs : dict, optional
        Extra keyword arguments passed to ``lens.build_closed_lens_mesh``. May not set
        ``fov``, ``init_res``, ``min_img_sep``, ``mu_min``, ``mu_max``, ``device`` or
        ``params``, which the node sets itself.
    raytrace_kwargs : dict, optional
        Extra keyword arguments passed to ``lens.forward_raytrace``. May not set ``mesh`` or
        ``params``, which the node sets itself.
    device : str, optional
        The torch device used for caustics calls. Default: "cpu".
    torch_num_threads : int, optional
        If given, ``torch.set_num_threads`` is called with this value at the start of
        ``compute``.
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
        mesh_kwargs=None,
        raytrace_kwargs=None,
        device="cpu",
        torch_num_threads=None,
        seed=None,
        node_label=None,
    ):
        torch, _ = _import_caustics()

        if max_images < 1:
            raise ValueError(f"max_images must be at least 1, got {max_images}.")
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

        reserved_names = {"z_s", "fov", "init_res", "min_img_sep"} | set(self.OUTPUTS)
        node_inputs = {}
        for key, setter in lens_parameters.items():
            name = key.replace(".", "_")
            if name in reserved_names:
                raise ValueError(
                    f"Mapped key '{key}' collides with the reserved node parameter name '{name}'."
                )
            node_inputs[name] = setter
        node_inputs.update(z_s=z_s, fov=fov, init_res=init_res, min_img_sep=min_img_sep)

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
        stored_index = _lens_param_index(stored_lens)
        for key, p in stored_index.items():
            if key in mapped_keys:
                p.to_dynamic()
            else:
                p.to_static()
        self._lens = stored_lens.to(dtype=torch.float64)

        stored_index = _lens_param_index(self._lens)
        param_order = []
        for p in self._lens.dynamic_params:
            key = next(k for k, v in stored_index.items() if v is p)
            param_order.append("z_s" if key == z_s_key else key.replace(".", "_"))
        self._param_order = param_order

        self._z_l_name = z_l_name
        self._z_l_value = z_l_value
        self._max_images = int(max_images)
        self._mu_min = mu_min
        self._mu_max = mu_max
        self._mesh_kwargs = mesh_kwargs
        self._raytrace_kwargs = raytrace_kwargs
        self._device = device
        self._torch_num_threads = torch_num_threads
        self._owner_pid = os.getpid()
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

    def _build(self, lens, values, params, system_index):
        """Build one system's closed lens mesh, raising on caustics' failure warnings.

        The build runs under a module lock with its warnings recorded. Failure warnings are
        raised as one error; every other warning is re-emitted unchanged.

        Parameters
        ----------
        lens : caustics.lenses.base.Lens
            This call's working copy of the stored lens, on ``device``.
        values : dict
            This system's node input values, keyed by node parameter name.
        params : list of torch.Tensor
            The values of the lens's dynamic parameters, in ``lens.dynamic_params`` order.
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
        with _BUILD_LOCK, warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            built = lens.build_closed_lens_mesh(
                float(values["fov"]),
                int(values["init_res"]),
                float(values["min_img_sep"]),
                mu_min=self._mu_min,
                mu_max=self._mu_max,
                device=self._device,
                params=params,
                **self._mesh_kwargs,
            )
        failures = []
        for w in records:
            message = str(w.message)
            if any(text in message for text in _FAILURE_MESSAGES):
                failures.append(message)
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)
        if failures:
            raise RuntimeError(
                f"{_describe_system(system_index, values)}: caustics could not build a valid lens "
                "mesh; adjust fov, init_res or min_img_sep (or max_depth / max_iters in mesh_kwargs). "
                f"Caustics warned: {' | '.join(failures)}"
            )
        return built

    def _sample_source(self, mesh, curves, region, rng):
        """Draw a source uniformly inside the caustics by rejection sampling.

        Candidates are drawn uniformly in the bounding box of the caustics' source polylines,
        in batches of 1024, and the first one inside (in draw order) is accepted.

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

        Returns
        -------
        source_x, source_y : float
            The source offset (east, north) from the lens origin in arcseconds.

        Raises
        ------
        RuntimeError
            If no candidate is accepted within 100 batches.
        """
        import torch

        source = curves.source
        low = source.min(dim=0).values.cpu().numpy()
        high = source.max(dim=0).values.cpu().numpy()
        for _ in range(_MAX_CANDIDATE_BATCHES):
            candidates = rng.uniform(low, high, size=(_CANDIDATE_BATCH, 2))
            # Test the batch a chunk at a time, stopping at the first chunk that accepts one.
            for start in range(0, _CANDIDATE_BATCH, _WINDING_CHUNK):
                chunk = candidates[start : start + _WINDING_CHUNK]
                inside = _inside_caustics(curves, torch.as_tensor(chunk, device=source.device)).cpu().numpy()
                if inside.any():
                    source_x, source_y = chunk[np.argmax(inside)]
                    return float(source_x), float(source_y)
        raise RuntimeError(
            f"no strong-lensing region found: none of {_MAX_CANDIDATE_BATCHES * _CANDIDATE_BATCH} "
            "candidate sources drawn in the bounding box of the caustics lies inside them."
        )

    def _solve_system(self, lens, values, rng, system_index):
        """Build one system's mesh, draw its source, and solve and evaluate its images.

        Parameters
        ----------
        lens : caustics.lenses.base.Lens
            This call's working copy of the stored lens, on ``device``.
        values : dict
            This system's node input values, keyed by node parameter name.
        rng : numpy.random.Generator
            This system's random number generator.
        system_index : int
            The global index of the system, used in error messages.

        Returns
        -------
        row : dict
            One entry per name in ``OUTPUTS``: scalars for ``source_x``, ``source_y``,
            ``num_images`` and ``area``, and (max_images,) float64 arrays, sorted by arrival
            time and padded, for the per-image outputs.

        Raises
        ------
        ValueError
            If ``z_l >= z_s``.
        RuntimeError
            If the mesh build fails (see ``_build``); the system has no strong-lensing region
            (its caustic area is not finite or not positive, or no candidate source is
            accepted); the source lies outside the image of the fov boundary; or the source has
            more than ``max_images`` images, or fewer than the sampling region guarantees (2 in
            caustic mode, 1 in μ-region mode).
        """
        import torch
        from caustics.lenses.func.adaptive import caustic_area
        from caustics.lenses.func.adaptive.lens_mesh import inside_fov_image

        z_l = values[self._z_l_name] if self._z_l_name is not None else self._z_l_value
        if z_l is not None and z_l >= values["z_s"]:
            raise ValueError(
                f"{_describe_system(system_index, values)}: z_l ({z_l}) must be less than "
                f"z_s ({values['z_s']})."
            )

        params = [
            torch.as_tensor(values[name], dtype=torch.float64, device=self._device)
            for name in self._param_order
        ]
        mesh, curves, region = self._build(lens, values, params, system_index)
        area = float(caustic_area(curves))
        if not np.isfinite(area) or area <= 0.0:
            raise RuntimeError(
                f"{_describe_system(system_index, values)} has no strong-lensing region: its caustic "
                f"area is {area} arcsec^2 (NaN when the fov cuts a critical curve, 0 when the lens "
                "forms none)."
            )
        try:
            source_x, source_y = self._sample_source(mesh, curves, region, rng)
        except RuntimeError as err:
            raise RuntimeError(f"{_describe_system(system_index, values)}: {err}") from err

        # forward_raytrace raises for a source outside the image of the fov boundary, where some
        # of its images lie outside the fov; check first so the error names the system.
        beta = torch.tensor([[source_x, source_y]], dtype=torch.float64, device=mesh.vertices_source.device)
        if not bool(inside_fov_image(mesh, beta)[0]):
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
        shear_1, shear_2 = lens.shear(x, y, params=params)
        per_image = {
            "image_x": x,
            "image_y": y,
            "macro_magnifications": lens.magnification(x, y, params=params),
            "time_delays": lens.time_delay(x, y, params=params),
            "convergences": lens.convergence(x, y, params=params),
            "shear_1": shear_1,
            "shear_2": shear_2,
        }
        per_image = {name: value.detach().cpu().numpy() for name, value in per_image.items()}
        order = np.argsort(per_image["time_delays"], kind="stable")
        per_image["time_delays"] = per_image["time_delays"] - per_image["time_delays"][order[0]]

        row = {"source_x": source_x, "source_y": source_y, "num_images": num_images, "area": area}
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
        """
        import torch

        args = self._build_inputs(graph_state, **kwargs)
        rng = rng_info if rng_info is not None else self._rng
        num_samples = graph_state.num_samples
        sub_seeds = rng.integers(0, 2**63, size=num_samples, dtype=np.uint64)

        lens = copy.deepcopy(self._lens).to(device=self._device)
        sample_offset = graph_state.sample_offset or 0
        rows = []
        with torch.device(self._device):
            for i, sub_seed in enumerate(sub_seeds):
                values = {name: value if num_samples == 1 else value[i] for name, value in args.items()}
                rows.append(
                    self._solve_system(lens, values, np.random.default_rng(sub_seed), sample_offset + i)
                )

        results = [np.array([row[name] for row in rows]) for name in self.OUTPUTS]
        if num_samples == 1:
            results = [value[0] for value in results]
        self._save_results(results, graph_state)
        return results
