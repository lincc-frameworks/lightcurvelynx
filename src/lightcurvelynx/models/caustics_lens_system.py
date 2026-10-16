"""A sampling node that uses caustics to build a strong-lens mesh for a caustics lens model,
draw a source inside its strong-lensing region, raytrace it to images, and evaluate the
per-image observables that ``ResolvedStrongLensModel`` consumes.
"""

import copy
import os

import numpy as np

from lightcurvelynx.base_models import FunctionNode


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

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Placeholder; the mesh build, sampling and raytracing are implemented in a later task."""
        raise NotImplementedError("CausticsStrongLensSystem.compute is implemented in a later task.")
