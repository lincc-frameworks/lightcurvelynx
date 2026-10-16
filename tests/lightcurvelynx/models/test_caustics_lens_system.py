import json
import pickle  # noqa: F401 (used starting Task 8's pickling tests)
import warnings
from concurrent.futures import ThreadPoolExecutor  # noqa: F401 (used starting Task 8's parallel-safety tests)
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

torch = pytest.importorskip("torch")
caustics = pytest.importorskip("caustics")
adaptive = pytest.importorskip("caustics.lenses.func.adaptive")

from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode  # noqa: E402,I001
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc  # noqa: E402
from lightcurvelynx.models.caustics_lens_system import (  # noqa: E402
    CausticsStrongLensSystem,
    _inside_caustics,
    _lens_param_index,
)
# Task 5 adds the LinearTimeModel, ResolvedStrongLensModel and simulate_lightcurves
# imports here, at the top.

Z_L, Z_S = 0.5, 1.5
VALUES = {"q": 0.7, "phi": 0.3, "Rein": 1.0, "gamma_1": 0.05, "gamma_2": -0.02}
LENS_PARAMETERS = {
    "lens.z_l": Z_L,
    "sie.q": VALUES["q"],
    "sie.phi": VALUES["phi"],
    "sie.Rein": VALUES["Rein"],
    "shear.gamma_1": VALUES["gamma_1"],
    "shear.gamma_2": VALUES["gamma_2"],
}
MESH = {"fov": 5.0, "init_res": 32, "min_img_sep": 1e-2}


def _make_lens(sie_name="sie", q=None, phi=None, Rein=None, gamma_1=None, gamma_2=None, z_l=None, z_s=None):
    """Return an SIE + external shear SinglePlane lens; unset values stay unset."""
    cosmo = caustics.FlatLambdaCDM(name="cosmo")
    sie = caustics.SIE(cosmology=cosmo, name=sie_name, x0=0.0, y0=0.0, s=0.0, q=q, phi=phi, Rein=Rein)
    shear = caustics.ExternalShear(
        cosmology=cosmo, name="shear", x0=0.0, y0=0.0, gamma_1=gamma_1, gamma_2=gamma_2
    )
    return caustics.SinglePlane(cosmology=cosmo, lenses=[sie, shear], name="lens", z_l=z_l, z_s=z_s)


def _reference_lens():
    """Return the test lens with every value set (static), in float64, for direct caustics calls.

    The values are passed as float64 tensors: caskade would store python floats in float32, and
    casting the lens to float64 afterwards would keep that rounding (e.g. q = 0.699999988).
    """
    values = VALUES | {"z_l": Z_L, "z_s": Z_S}
    return _make_lens(**{k: torch.tensor(v, dtype=torch.float64) for k, v in values.items()}).to(
        dtype=torch.float64
    )


def _snapshot(lens):
    """Return each parameter's (state, value, dtype), to check that a lens is unchanged."""
    return {
        key: (p.node_type, None if p.value is None else p.value.clone(), p.dtype)
        for key, p in _lens_param_index(lens).items()
    }


def _assert_same_snapshot(before, after):
    """Assert that two lens snapshots are identical."""
    assert before.keys() == after.keys()
    for key, (state, value, dtype) in before.items():
        assert after[key][0] == state and after[key][2] == dtype
        assert (value is None and after[key][1] is None) or torch.equal(value, after[key][1])


def _make_system(lens=None, **overrides):
    """Return a CausticsStrongLensSystem over the test lens; keyword overrides replace defaults."""
    kwargs = (
        {
            "lens_parameters": LENS_PARAMETERS,
            "z_s": Z_S,
            "max_images": 5,
            "seed": 11,
            "node_label": "system",
        }
        | MESH
        | overrides
    )
    return CausticsStrongLensSystem(_make_lens() if lens is None else lens, **kwargs)


def test_parameters_and_outputs_registered():
    """Test that mapped keys become node parameters and all outputs are registered."""
    system = _make_system()
    for name in (
        "lens_z_l",
        "sie_q",
        "sie_phi",
        "sie_Rein",
        "shear_gamma_1",
        "shear_gamma_2",
        "z_s",
        "fov",
        "init_res",
        "min_img_sep",
    ):
        assert hasattr(system, name)
    assert tuple(system.outputs) == CausticsStrongLensSystem.OUTPUTS
    assert set(system.resolved_inputs()) == {
        "source_x",
        "source_y",
        "image_x",
        "image_y",
        "macro_magnifications",
        "time_delays",
        "num_images",
    }


def test_user_lens_is_not_modified():
    """Test that construction leaves the user's lens untouched."""
    lens = _make_lens()
    before = _snapshot(lens)
    _make_system(lens)
    _assert_same_snapshot(before, _snapshot(lens))


def test_private_lens_copy_is_float64_with_mapped_dynamic():
    """Test that the stored copy has exactly the mapped parameters and z_s dynamic, in float64."""
    system = _make_system()
    assert sorted(system._param_order) == sorted(
        ["lens_z_l", "sie_q", "sie_phi", "sie_Rein", "shear_gamma_1", "shear_gamma_2", "z_s"]
    )
    assert all(
        p.static
        for key, p in _lens_param_index(system._lens).items()
        if key.replace(".", "_") not in system._param_order and key != "lens.z_s"
    )
    assert _lens_param_index(system._lens)["sie.x0"].value.dtype == torch.float64


def test_static_lens_values_need_no_mapping():
    """Test that parameters set on the caustics object stay constants and need no mapping."""
    system = _make_system(
        _make_lens(q=0.7, phi=0.3),
        lens_parameters={k: v for k, v in LENS_PARAMETERS.items() if k not in ("sie.q", "sie.phi")},
    )
    assert "sie_q" not in system._param_order


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"lens_parameters": LENS_PARAMETERS | {"sie.nope": 1.0}}, "sie.nope"),
        ({"lens_parameters": {k: v for k, v in LENS_PARAMETERS.items() if k != "sie.q"}}, "sie.q"),
        ({"lens_parameters": LENS_PARAMETERS | {"lens.z_s": 1.0}}, "z_s"),
        ({"max_images": 0}, "max_images"),
        ({"mu_max": 50.0}, "mu_min"),
        ({"mu_min": 10.0, "mu_max": 5.0}, "mu_max"),
        ({"mesh_kwargs": {"fov": 3.0}}, "fov"),
        ({"raytrace_kwargs": {"mesh": None}}, "mesh"),
    ],
)
def test_construction_errors(overrides, match):
    """Test the construction-time checks."""
    with pytest.raises(ValueError, match=match):
        _make_system(**overrides)


def test_name_collision_raises():
    """Test that a mapped key colliding with a node parameter raises (module 'z' + param 's' -> 'z_s')."""
    params = {k.replace("sie.", "z."): v for k, v in LENS_PARAMETERS.items()} | {"z.s": 0.0}
    with pytest.raises(ValueError, match="z_s"):
        _make_system(_make_lens(sie_name="z"), lens_parameters=params)


def test_key_errors_list_available_keys():
    """Test that unknown-key errors list the available module.param keys."""
    with pytest.raises(ValueError, match=r"sie\.Rein"):
        _make_system(lens_parameters=LENS_PARAMETERS | {"SIE.q": 0.5})


def test_user_lens_unchanged_by_sampling():
    """Test that sampling leaves the user's lens untouched."""
    lens = _make_lens()
    before = _snapshot(lens)
    _make_system(lens).sample_parameters(num_samples=1)
    _assert_same_snapshot(before, _snapshot(lens))


def _mesh_and_curves():
    """Return a mesh and curves of the reference lens at the test settings."""
    mesh, curves, _ = _reference_lens().build_closed_lens_mesh(
        MESH["fov"], MESH["init_res"], MESH["min_img_sep"]
    )
    return mesh, curves


def test_inside_caustics_matches_caustic_area_and_is_multiply_imaged():
    """Test that the acceptance test agrees with caustic_area and only accepts multiply imaged sources."""
    mesh, curves = _mesh_and_curves()
    lo, hi = curves.source.min(dim=0).values, curves.source.max(dim=0).values
    points = lo + (hi - lo) * torch.rand(
        20000, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    inside = _inside_caustics(curves, points)
    box_area = float(torch.prod(hi - lo))
    assert inside.double().mean().item() * box_area == pytest.approx(adaptive.caustic_area(curves), rel=0.03)
    accepted = points[inside][:2000]
    _, _, counts = _reference_lens().forward_raytrace(accepted[:, 0], accepted[:, 1], mesh=mesh)
    assert counts.min() >= 2
    assert (counts == 2).any() and (counts >= 3).any()  # pseudo-caustic-only and inside-caustic parts


def test_single_system_shapes_and_padding():
    """Test one-system outputs: scalars, (I,) arrays, padding and sorted, zero-based delays."""
    state = _make_system().sample_parameters(num_samples=1)
    out = state["system"]
    n = int(out["num_images"])
    assert np.ndim(out["source_x"]) == 0 and np.ndim(out["area"]) == 0
    for name in (
        "image_x",
        "image_y",
        "macro_magnifications",
        "time_delays",
        "convergences",
        "shear_1",
        "shear_2",
    ):
        assert np.shape(out[name]) == (5,)
    assert 2 <= n <= 5
    assert np.all(np.isnan(out["image_x"][n:])) and np.all(out["macro_magnifications"][n:] == 0.0)
    assert out["time_delays"][0] == 0.0 and np.all(np.diff(out["time_delays"][:n]) >= 0.0)


def test_multi_system_shapes():
    """Test that S systems give (S,) and (S, I) outputs."""
    out = _make_system().sample_parameters(num_samples=3)["system"]
    assert out["source_x"].shape == (3,) and out["num_images"].shape == (3,) and out["area"].shape == (3,)
    assert out["image_x"].shape == (3, 5) and out["shear_2"].shape == (3, 5)


def test_images_solve_lens_equation_and_observables_match_caustics():
    """Test that images map back to the source and observables equal direct caustics calls."""
    out = _make_system().sample_parameters(num_samples=2)["system"]
    ref = _reference_lens()
    for i in range(2):
        n = int(out["num_images"][i])
        x = torch.as_tensor(out["image_x"][i, :n])
        y = torch.as_tensor(out["image_y"][i, :n])
        bx, by = ref.raytrace(x, y)
        assert_allclose(bx.numpy(), out["source_x"][i], atol=1e-5)
        assert_allclose(by.numpy(), out["source_y"][i], atol=1e-5)
        assert_allclose(out["macro_magnifications"][i, :n], ref.magnification(x, y).numpy(), rtol=1e-8)
        delays = ref.time_delay(x, y).numpy()
        assert_allclose(out["time_delays"][i, :n], delays - delays.min(), atol=1e-8)
        assert_allclose(out["convergences"][i, :n], ref.convergence(x, y).numpy(), rtol=1e-8)
        g1, g2 = ref.shear(x, y)
        assert_allclose(out["shear_1"][i, :n], g1.numpy(), atol=1e-10)
        assert_allclose(out["shear_2"][i, :n], g2.numpy(), atol=1e-10)
    _, curves = _mesh_and_curves()
    assert_allclose(out["area"], adaptive.caustic_area(curves), rtol=1e-10)


def test_same_seed_reproduces_and_rng_advances_by_num_samples():
    """Test reproducibility and that the shared rng advances by exactly num_samples draws."""
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    first = _make_system().sample_parameters(num_samples=2, rng_info=rng_a)["system"]
    rng_b.integers(0, 2**63, size=2, dtype=np.uint64)
    assert rng_a.random() == rng_b.random()
    second = _make_system().sample_parameters(num_samples=2, rng_info=np.random.default_rng(7))["system"]
    for name in CausticsStrongLensSystem.OUTPUTS:
        assert_array_equal(first[name], second[name])


def test_linked_fov_varies_per_system():
    """Test that fov linked to a sampled Einstein radius is used per system."""
    rein = NumpyRandomFunc("uniform", low=0.8, high=1.2, seed=1, node_label="rein")
    system = _make_system(
        lens_parameters=LENS_PARAMETERS | {"sie.Rein": rein}, fov=BasicMathNode("5 * r", r=rein)
    )
    out = system.sample_parameters(num_samples=2)["system"]
    assert_allclose(out["fov"], 5.0 * out["sie_Rein"])
    assert out["fov"][0] != out["fov"][1]


def test_integer_settings_given_as_floats():
    """Test that init_res given as a float-valued node output is cast to int."""
    out = _make_system(init_res=BasicMathNode("16.0 * 2")).sample_parameters(num_samples=1)["system"]
    assert int(out["num_images"]) >= 2


def test_mapped_parameter_link_samples():
    """Test that a node linked to a mapped lens parameter sees its sampled value."""
    system = _make_system()
    double_q = BasicMathNode("2 * q", q=system.sie_q, node_label="double_q")
    state = double_q.sample_parameters(num_samples=1)
    assert state["double_q"]["function_node_result"] == pytest.approx(2 * VALUES["q"])


def test_no_strong_lensing_region_raises():
    """Test that a lens with no critical curves (external shear alone) raises, naming the system."""
    cosmo = caustics.FlatLambdaCDM(name="cosmo")
    shear = caustics.ExternalShear(cosmology=cosmo, name="shear", x0=0.0, y0=0.0)
    lens = caustics.SinglePlane(cosmology=cosmo, lenses=[shear], name="lens")
    params = {k: v for k, v in LENS_PARAMETERS.items() if not k.startswith("sie.")}
    with pytest.raises(RuntimeError, match="system 0 .*no strong-lensing region"):
        _make_system(lens, lens_parameters=params).sample_parameters(num_samples=1)


def test_rejection_cap_raises():
    """Test that sampling gives up when no candidate lands inside the caustics."""
    # A polyline that doubles back on itself has a non-empty bounding box but winds about nothing.
    source = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    curves = SimpleNamespace(source=source, offsets=torch.tensor([0, 2]))
    with pytest.raises(RuntimeError, match="no strong-lensing region"):
        _make_system()._sample_source(None, curves, None, np.random.default_rng(0))


@pytest.mark.parametrize(
    "overrides, error, match",
    [
        ({"lens_parameters": LENS_PARAMETERS | {"lens.z_l": 2.0}}, ValueError, "z_l"),
        ({"max_images": 1}, RuntimeError, "max_images"),
        ({"fov": 2.0, "mesh_kwargs": {"max_iters": 0}}, RuntimeError, r"extension\(s\)"),
        # A starting fov inside the critical curves, where the mesh holds only the pseudo-caustic.
        # A source drawn inside it either lies outside the image of the fov boundary, where
        # caustics' forward_raytrace would raise (the seed-11 source at fov=0.5), or inside it
        # with zero images in the fov (the seed-11 source at fov=0.1). Both say "too few images".
        ({"fov": 0.5}, RuntimeError, "too few images in the fov: the source lies outside"),
        ({"fov": 0.1}, RuntimeError, r"too few images \(0, expected at least 2\)"),
        ({"mesh_kwargs": {"max_depth": 2}}, RuntimeError, "depth-limited"),
    ],
)
def test_per_system_failures_raise(overrides, error, match):
    """Test that bad systems raise instead of reaching the output."""
    with pytest.raises(error, match=match):
        _make_system(**overrides).sample_parameters(num_samples=1)


def test_error_names_global_system_index():
    """Test that errors name the system by its global index (sample_offset + local index)."""
    system = _make_system(lens_parameters=LENS_PARAMETERS | {"lens.z_l": 2.0})
    with pytest.raises(ValueError, match="system 7"):
        system.sample_parameters(num_samples=1, sample_offset=7)


def test_informational_caustics_warnings_are_reemitted():
    """Test that the fov-widening notice passes through as a warning, not an error."""
    # The hole around a center has radius min_img_sep / 2 = 0.005, so a center at x = 2.498
    # reaches past the fov edge at 2.5 and caustics widens the fov (a center at 2.45 does not).
    system = _make_system(mesh_kwargs={"centers": [[2.498, 0.0]]})
    with pytest.warns(UserWarning, match="reach outside fov"):
        system.sample_parameters(num_samples=1)


def test_build_escalates_failure_warnings_and_reemits_others():
    """Test that _build raises on caustics' capped-hole-curve warning and re-emits the rest unchanged."""

    def build_closed_lens_mesh(*args, **kwargs):
        warnings.warn_explicit("a dependency notice", DeprecationWarning, "dependency.py", 3)
        # The text of caustics' holes.py warning, at its sample cap of 1 << 16.
        warnings.warn(
            "the hole curve around (0, 0) stopped at 65536 samples (cap 65536) before every chord "
            "fell below min_img_sep=0.01; it is exact at its samples but coarser than min_img_sep "
            "between them"
        )

    system = _make_system()
    lens = SimpleNamespace(build_closed_lens_mesh=build_closed_lens_mesh)
    values = {"fov": 5.0, "init_res": 32, "min_img_sep": 0.01}
    with pytest.warns(DeprecationWarning, match="a dependency notice") as record:
        with pytest.raises(RuntimeError, match="system 3 .*before every chord fell below"):
            system._build(lens, values, [], 3)
    # Only the dependency notice comes back, at its recorded location; the failure is not re-emitted.
    assert [(w.filename, w.lineno) for w in record] == [("dependency.py", 3)]


@pytest.mark.parametrize("ignore_json, expected", [(False, 1), (True, 0)])
def test_reemitted_warnings_keep_module_scoped_filters(ignore_json, expected):
    """Test that a filter scoped to the emitting module still applies to a re-emitted warning."""

    def build_closed_lens_mesh(*args, **kwargs):
        warnings.warn_explicit("a json notice", UserWarning, json.__file__, 1)

    system = _make_system()
    lens = SimpleNamespace(build_closed_lens_mesh=build_closed_lens_mesh)
    values = {"fov": 5.0, "init_res": 32, "min_img_sep": 0.01}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        if ignore_json:
            warnings.filterwarnings("ignore", category=UserWarning, module="json")
        system._build(lens, values, [], 0)
    assert len(record) == expected
