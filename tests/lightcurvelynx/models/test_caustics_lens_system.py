import json
import multiprocessing
import os
import pickle
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

torch = pytest.importorskip("torch")
caustics = pytest.importorskip("caustics")
adaptive = pytest.importorskip("caustics.lenses.func.adaptive")

import lightcurvelynx  # noqa: E402,I001
from caskade import Module, Param  # noqa: E402
from caustics import backend  # noqa: E402
from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup  # noqa: E402
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode  # noqa: E402
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc  # noqa: E402
from lightcurvelynx.models.basic_models import LinearTimeModel  # noqa: E402
from lightcurvelynx.models.caustics_lens_system import (  # noqa: E402
    CausticsStrongLensSystem,
    _inside_caustics,
    _jax_device,
    _lens_param_index,
)
from lightcurvelynx.models.strong_lens_model import ResolvedStrongLensModel  # noqa: E402
from lightcurvelynx.noise_models.base_noise_models import GivenNoiseModel  # noqa: E402
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable  # noqa: E402
from lightcurvelynx.simulate import simulate_lightcurves  # noqa: E402
from lightcurvelynx.survey_info import SurveyInfo  # noqa: E402

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

    The values are passed as float64 arrays of caustics' active backend: under torch caskade would
    store python floats in float32, and casting the lens to float64 afterwards would keep that
    rounding (e.g. q = 0.699999988).
    """
    values = VALUES | {"z_l": Z_L, "z_s": Z_S}
    return _make_lens(**{k: backend.as_array(v, dtype=backend.float64) for k, v in values.items()}).to(
        dtype=backend.float64
    )


def _snapshot(lens):
    """Return each parameter's (state, value, dtype), to check that a lens is unchanged."""
    return {
        key: (p.node_type, None if p.value is None else np.array(backend.to_numpy(p.value)), p.dtype)
        for key, p in _lens_param_index(lens).items()
    }


def _assert_same_snapshot(before, after):
    """Assert that two lens snapshots are identical."""
    assert before.keys() == after.keys()
    for key, (state, value, dtype) in before.items():
        assert after[key][0] == state and after[key][2] == dtype
        assert (value is None and after[key][1] is None) or np.array_equal(value, after[key][1])


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
    assert _lens_param_index(system._lens)["sie.x0"].value.dtype == backend.float64


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


def test_mapped_keys_with_one_node_name_raise():
    """Test that two mapped keys becoming the same node parameter name ('a.b_c', 'a_b.c' -> 'a_b_c') raise."""
    lens, inner = Module("a"), Module("a_b")
    lens.b_c, inner.c, lens.inner = Param("b_c", 1.0), Param("c", 1.0), inner
    with pytest.raises(ValueError, match=r"'a\.b_c' and 'a_b\.c'.*'a_b_c'"):
        _make_system(lens, lens_parameters={"a.b_c": 1.0, "a_b.c": 2.0})


def test_key_errors_list_available_keys():
    """Test that unknown-key errors list the available module.param keys."""
    with pytest.raises(ValueError, match=r"sie\.Rein"):
        _make_system(lens_parameters=LENS_PARAMETERS | {"SIE.q": 0.5})


def test_compute_under_another_backend_raises():
    """Test that compute raises when caustics' active backend is not the one the node was built under."""
    system = _make_system()
    system._backend = "jax" if backend.backend == "torch" else "torch"
    with pytest.raises(
        ValueError, match=f"built under caustics' '{system._backend}' backend.*CASKADE_BACKEND"
    ):
        system.sample_parameters(num_samples=1)


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
    source = backend.to_numpy(curves.source)
    lo, hi = source.min(axis=0), source.max(axis=0)
    points = lo + (hi - lo) * np.random.default_rng(0).random((20000, 2))
    inside = _inside_caustics(curves, backend.as_array(points))
    box_area = float(np.prod(hi - lo))
    assert inside.mean() * box_area == pytest.approx(adaptive.caustic_area(curves), rel=0.03)
    accepted = backend.as_array(points[inside][:2000])
    _, _, counts = _reference_lens().forward_raytrace(accepted[:, 0], accepted[:, 1], mesh=mesh)
    counts = backend.to_numpy(counts)
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
    """Test that images map back to the source, and image counts and observables match caustics."""
    out = _make_system().sample_parameters(num_samples=2)["system"]
    ref = _reference_lens()
    for i in range(2):
        n = int(out["num_images"][i])
        x = backend.as_array(out["image_x"][i, :n])
        y = backend.as_array(out["image_y"][i, :n])
        bx, by = (backend.to_numpy(b) for b in ref.raytrace(x, y))
        assert_allclose(bx, out["source_x"][i], atol=1e-5)
        assert_allclose(by, out["source_y"][i], atol=1e-5)
        mu = backend.to_numpy(ref.magnification(x, y))
        assert_allclose(out["macro_magnifications"][i, :n], mu, rtol=1e-8)
        delays = backend.to_numpy(ref.time_delay(x, y))
        assert_allclose(out["time_delays"][i, :n], delays - delays.min(), atol=1e-8)
        assert_allclose(out["convergences"][i, :n], backend.to_numpy(ref.convergence(x, y)), rtol=1e-8)
        g1, g2 = (backend.to_numpy(g) for g in ref.shear(x, y))
        assert_allclose(out["shear_1"][i, :n], g1, atol=1e-10)
        assert_allclose(out["shear_2"][i, :n], g2, atol=1e-10)
    mesh, curves = _mesh_and_curves()
    sx, sy = backend.as_array(out["source_x"]), backend.as_array(out["source_y"])
    _, _, counts = ref.forward_raytrace(sx, sy, mesh=mesh)
    assert_array_equal(out["num_images"], backend.to_numpy(counts))
    assert_allclose(out["area"], adaptive.caustic_area(curves), rtol=1e-10)


def test_mu_region_area_and_membership():
    """Test that mu-region mode samples inside mu_min <= mu_tot <= mu_max and returns region.area."""
    out = _make_system(mu_min=10.0, mu_max=50.0).sample_parameters(num_samples=2)["system"]
    mesh, _, region = _reference_lens().build_closed_lens_mesh(
        MESH["fov"], MESH["init_res"], MESH["min_img_sep"], mu_min=10.0, mu_max=50.0
    )
    assert_allclose(out["area"], region.area, rtol=1e-10)
    mu_tot = backend.to_numpy(
        adaptive.total_magnification(
            backend.as_array(out["source_x"]), backend.as_array(out["source_y"]), mesh
        )
    )
    assert np.all((mu_tot >= 10.0) & (mu_tot <= 50.0))
    assert np.all(out["num_images"] >= 1)


@pytest.mark.parametrize(
    "mu_min, mu_max",
    [
        (10.0, None),
        (10.0, 50.0),
        pytest.param(
            100.0,
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="caustics' region contours misestimate the area near caustics; see "
                "CausticsStrongLensSystem docstring",
            ),
        ),
    ],
)
def test_mu_region_acceptance_matches_region_area(mu_min, mu_max):
    """Test that the node's mu-region acceptance rule agrees with region.area (statistical)."""
    mesh, _, region = _reference_lens().build_closed_lens_mesh(
        MESH["fov"], MESH["init_res"], MESH["min_img_sep"], mu_min=mu_min, mu_max=mu_max
    )
    source = backend.to_numpy(region.source)
    lo, hi = source.min(axis=0), source.max(axis=0)
    # At mu_min=100 the region fills ~1% of its bounding box, so this many points are needed to keep
    # the statistical error (~1.5% there, ~0.25% at mu_min=10) small against the 3% tolerance.
    points = backend.as_array(lo + (hi - lo) * np.random.default_rng(1).random((400_000, 2)))
    mu_tot = backend.to_numpy(adaptive.total_magnification(points[:, 0], points[:, 1], mesh))
    upper = np.inf if mu_max is None else mu_max
    fraction = ((mu_tot >= mu_min) & (mu_tot <= upper)).mean()  # NaN compares False
    assert fraction * float(np.prod(hi - lo)) == pytest.approx(region.area, rel=0.03)


def test_unbounded_mu_region_raises():
    """Test that mu_min at/below the far-field total magnification raises (no fov bounds the region)."""
    # External shear gamma ~= 0.054 gives far-field mu_tot ~= 1 / |1 - gamma^2| ~= 1.003, so no fov
    # ever bounds the region; max_iters=0 skips the (slow, and here pointless) fov-growing retries
    # and reaches caustics' "still reaches the edge" warning on the first attempt.
    with pytest.raises(RuntimeError, match=r"system 0.*extension\(s\)"):
        _make_system(mu_min=1.0, mesh_kwargs={"max_iters": 0}).sample_parameters(num_samples=1)


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
    """Test that fov and Rein linked to a sampled Einstein radius are used per system."""
    rein = NumpyRandomFunc("uniform", low=0.8, high=1.2, seed=1, node_label="rein")
    system = _make_system(
        lens_parameters=LENS_PARAMETERS | {"sie.Rein": rein}, fov=BasicMathNode("5 * r", r=rein)
    )
    out = system.sample_parameters(num_samples=2)["system"]
    assert_allclose(out["fov"], 5.0 * out["sie_Rein"])
    assert out["fov"][0] != out["fov"][1]
    assert out["area"][0] != out["area"][1]  # Each system's mesh is built from its own inputs.


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


@pytest.mark.parametrize(
    "mu_min, match",
    [(None, "no strong-lensing region"), (10.0, r"no mu-region \(10.0 <= mu_tot <= inf\)")],
)
def test_no_sampling_region_raises(mu_min, match):
    """Test that a lens with no critical curves (external shear alone) raises in both modes."""
    # Its mu_tot stays near the far-field 1.003 everywhere, so the mu-region is empty (area 0).
    cosmo = caustics.FlatLambdaCDM(name="cosmo")
    shear = caustics.ExternalShear(cosmology=cosmo, name="shear", x0=0.0, y0=0.0)
    lens = caustics.SinglePlane(cosmology=cosmo, lenses=[shear], name="lens")
    params = {k: v for k, v in LENS_PARAMETERS.items() if not k.startswith("sie.")}
    with pytest.raises(RuntimeError, match=f"system 0 .*{match}"):
        _make_system(lens, lens_parameters=params, mu_min=mu_min).sample_parameters(num_samples=1)


def test_rejection_cap_returns_none():
    """Test that sampling gives up, returning None, when no candidate lands inside the caustics."""
    # A polyline that doubles back on itself has a non-empty bounding box but winds about nothing.
    source = backend.as_array([[0.0, 0.0], [1.0, 1.0]], dtype=backend.float64)
    curves = SimpleNamespace(source=source, offsets=backend.as_array([0, 2]))
    assert _make_system()._sample_source(None, curves, None, np.random.default_rng(0)) is None


@pytest.mark.parametrize(
    "mu_min, match",
    [(None, "strong-lensing region"), (10.0, r"mu-region \(10.0 <= mu_tot <= inf\)")],
)
def test_rejection_cap_error_names_system_and_region(monkeypatch, mu_min, match):
    """Test that hitting the rejection cap raises, naming the system and the mode's sampling region."""
    system = _make_system(mu_min=mu_min)
    monkeypatch.setattr(system, "_sample_source", lambda *args: None)
    with pytest.raises(RuntimeError, match=f"system 0 .*candidate sources.*{match}"):
        system.sample_parameters(num_samples=1)


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
            system._build(lens, values, [], "cpu", 3)
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
        system._build(lens, values, [], "cpu", 0)
    assert len(record) == expected


def _toy_passbands():
    """Return a two-filter toy passband group in Angstroms."""
    return PassbandGroup(
        [
            Passband(np.array([[4000.0, 0.5], [5000.0, 0.75], [6000.0, 0.25]]), "toy", "g"),
            Passband(np.array([[5000.0, 0.25], [6000.0, 0.5], [7000.0, 0.75]]), "toy", "r"),
        ]
    )


def _survey_info():
    """Return a fake survey with 18 g/r observations pointed at (10, -30) deg."""
    values = {
        "time": np.linspace(90.0, 260.0, 18),
        "ra": np.full(18, 10.0),
        "dec": np.full(18, -30.0),
        "filter": np.tile(["g", "r"], 9),
    }
    obstable = FakeObsTable(
        values,
        bandflux_error={"g": 1.0, "r": 1.0},
        radius=0.5,
        fwhm_px=2.0,
        sky_bg_electrons=100.0,
    )
    return SurveyInfo(obstable=obstable, passbands=_toy_passbands(), noise_model=GivenNoiseModel())


class _InProcessExecutor:
    """Executor that runs batches sequentially in this process."""

    def map(self, fn, iterable):
        return [fn(item) for item in iterable]


def _make_resolved():
    """Return a ResolvedStrongLensModel whose images come from the caustics test system."""
    source = LinearTimeModel(
        linear_base=10.0, linear_scale=2.0, t0=100.0, ra=10.0, dec=-30.0, redshift=Z_S, node_label="source"
    )
    system = _make_system(z_s=source.redshift)
    return ResolvedStrongLensModel(source, **system.resolved_inputs(), node_label="lensed")


def _simulate(**kwargs):
    """Run a two-system simulation with param_cols for the image layout."""
    return simulate_lightcurves(
        _make_resolved(),
        2,
        _survey_info(),
        param_cols=["lensed.system_id", "lensed.image_id"],
        rng=np.random.default_rng(4),
        progress_bar=False,
        **kwargs,
    )


def test_simulate_one_row_per_image_with_delayed_t0():
    """Test that simulate_lightcurves gives one row per caustics image with per-image t0."""
    results = _simulate()
    assert set(results["lensed_system_id"]) == {0, 1}
    for system_id in range(2):
        rows = results[results["lensed_system_id"] == system_id]
        assert 2 <= len(rows) <= 5
        assert_array_equal(rows["lensed_image_id"], np.arange(len(rows)))
        assert rows["t0"].iloc[0] == pytest.approx(100.0)  # Earliest image keeps the source t0.
        assert np.all(np.diff(rows["t0"].to_numpy()) >= 0.0)  # Arrival order.
        assert np.all(rows["nobs"].to_numpy() > 0)  # Every image falls in the pointing.


def test_pickle_round_trip_samples_identically():
    """Test that a stdlib pickle round trip (via cloudpickle for the lens) samples identically."""
    pytest.importorskip("cloudpickle")
    system = _make_system()
    clone = pickle.loads(pickle.dumps(system))
    a = system.sample_parameters(num_samples=2, rng_info=np.random.default_rng(3))["system"]
    b = clone.sample_parameters(num_samples=2, rng_info=np.random.default_rng(3))["system"]
    for name in CausticsStrongLensSystem.OUTPUTS:
        assert_array_equal(a[name], b[name])


def test_working_lens_is_reused_and_not_pickled():
    """Test that compute makes one working lens and its observables, reuses them, and pickling drops them."""
    pytest.importorskip("cloudpickle")
    system = _make_system()
    assert system._working is None and system._observables is None  # Nothing is copied at construction.
    system.sample_parameters(num_samples=1)
    working, observables = system._working, system._observables
    system.sample_parameters(num_samples=1)
    assert working is not None and working is not system._lens
    assert system._working is working and system._observables is observables
    assert all(f.module is working for f in observables.values())
    clone = pickle.loads(pickle.dumps(system))
    assert clone._working is None and clone._observables is None


@pytest.mark.skipif(backend.backend != "jax", reason="only jax compiles")
def test_second_compute_compiles_no_new_observable_programs(monkeypatch):
    """Test that a second compute, at new lens parameter values, reuses the observables' compiled programs."""
    import jax
    from caustics.jit import JitForward

    compiles = []

    def count_compiles(event, duration, **kwargs):
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(event)

    # Compiles during each observable call, as (method name, count).
    observable_compiles = []
    jit_call = JitForward.__call__

    def counting_call(self, *args, params=None):
        before = len(compiles)
        out = jit_call(self, *args, params=params)
        if self.method.__name__ in ("magnification", "time_delay", "convergence", "shear"):
            observable_compiles.append((self.method.__name__, len(compiles) - before))
        return out

    monkeypatch.setattr(JitForward, "__call__", counting_call)
    rein = NumpyRandomFunc("uniform", low=0.8, high=1.2, seed=1, node_label="rein")
    # Seed 25 draws a 2-image system, then a 4-image one.
    system = _make_system(lens_parameters=LENS_PARAMETERS | {"sie.Rein": rein}, seed=25)
    jax.monitoring.register_event_duration_secs_listener(count_compiles)
    try:
        first = system.sample_parameters(num_samples=1)["system"]
        num_first = len(observable_compiles)
        second = system.sample_parameters(num_samples=1)["system"]
    finally:
        jax.monitoring.unregister_event_duration_listener(count_compiles)
    assert second["sie_Rein"] != first["sie_Rein"]
    assert second["num_images"] != first["num_images"]  # Padding keeps one input shape.
    first_calls, second_calls = observable_compiles[:num_first], observable_compiles[num_first:]
    assert len(first_calls) == 4 and all(count > 0 for _, count in first_calls)
    assert len(second_calls) == 4 and all(count == 0 for _, count in second_calls)


@pytest.fixture
def restore_torch_threads():
    """Restore torch's thread count after a test changes it."""
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


def test_worker_process_defaults_to_one_thread(restore_torch_threads):
    """Test that a node running outside its construction process uses one torch thread by default."""
    system = _make_system()
    system._owner_pid = -1  # Pretend we are in a worker process.
    system.sample_parameters(num_samples=1)
    assert torch.get_num_threads() == 1


def test_explicit_thread_count_is_applied(restore_torch_threads):
    """Test that torch_num_threads is applied in compute."""
    _make_system(torch_num_threads=2).sample_parameters(num_samples=1)
    assert torch.get_num_threads() == 2


def test_stored_lens_unchanged_by_sampling():
    """Test that sampling leaves the stored lens copy unmoved, float64, with the same dynamic set."""
    system = _make_system()
    dynamic_before = [p.name for p in system._lens.dynamic_params]
    x0 = _lens_param_index(system._lens)["sie.x0"]
    x0_before = x0.value
    first = system.sample_parameters(num_samples=1, rng_info=np.random.default_rng(2))["system"]
    second = system.sample_parameters(num_samples=1, rng_info=np.random.default_rng(2))["system"]
    assert [p.name for p in system._lens.dynamic_params] == dynamic_before
    assert x0.value is x0_before and x0_before.dtype == backend.float64  # Not moved or cast.
    assert_array_equal(first["image_x"], second["image_x"])


def test_thread_executor_matches_in_process_batches():
    """Test that a ThreadPoolExecutor run matches in-process batches."""
    serial = _simulate(executor=_InProcessExecutor(), batch_size=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        threaded = _simulate(executor=pool, batch_size=1)
    assert_array_equal(threaded["lensed_system_id"], serial["lensed_system_id"])
    for col in ("ra", "dec", "t0"):
        assert_allclose(threaded[col], serial[col], rtol=1e-12)


def test_process_executor_matches_in_process_batches():
    """Test that a process-based run matches in-process batches within tolerance.

    Under torch it is num_jobs=2 (fork, 1 torch thread per worker); under jax, where forked workers
    are unsafe, a spawn ProcessPoolExecutor.
    """
    pytest.importorskip("cloudpickle")
    serial = _simulate(executor=_InProcessExecutor(), batch_size=1)
    if backend.backend == "jax":
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
            parallel = _simulate(executor=pool, batch_size=1)
    else:
        parallel = _simulate(num_jobs=2)
    assert_array_equal(parallel["lensed_system_id"], serial["lensed_system_id"])
    for col in ("ra", "dec", "t0"):
        assert_allclose(parallel[col], serial[col], rtol=1e-9)


def _has_gpu():
    """Return whether caustics' active backend has a GPU."""
    if backend.backend == "jax":
        import jax

        return jax.default_backend() == "gpu"
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_gpu(), reason="caustics' active backend has no GPU")
def test_cuda_matches_cpu():
    """Test that running on CUDA gives the same outputs as CPU for the same seed."""
    cpu = _make_system().sample_parameters(num_samples=2, rng_info=np.random.default_rng(5))["system"]
    gpu = _make_system(device="cuda").sample_parameters(num_samples=2, rng_info=np.random.default_rng(5))
    gpu = gpu["system"]
    assert_array_equal(gpu["num_images"], cpu["num_images"])
    for name in ("source_x", "source_y", "image_x", "macro_magnifications", "time_delays", "area"):
        assert_allclose(gpu[name], cpu[name], rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize(
    "device, expected",
    [("cpu", ("cpu", 0)), ("cuda", ("gpu", 0)), ("cuda:1", ("gpu", 1)), ("gpu", ("gpu", 0))],
)
def test_jax_device_maps_torch_style_names(monkeypatch, device, expected):
    """Test that a torch-style device name maps to the jax device of its platform and index."""
    jax = pytest.importorskip("jax")
    monkeypatch.setattr(jax, "devices", lambda platform: [(platform, 0), (platform, 1)])
    assert _jax_device(device) == expected


# The fixed case of test_jax_backend_matches_torch, run in a fresh process: caustics picks its backend
# once, at import, from CASKADE_BACKEND.
_CASE = f"""
import sys

import caustics
import numpy as np

from lightcurvelynx.models.caustics_lens_system import CausticsStrongLensSystem

cosmo = caustics.FlatLambdaCDM(name="cosmo")
sie = caustics.SIE(cosmology=cosmo, name="sie", x0=0.0, y0=0.0, s=0.0)
shear = caustics.ExternalShear(cosmology=cosmo, name="shear", x0=0.0, y0=0.0)
lens = caustics.SinglePlane(cosmology=cosmo, lenses=[sie, shear], name="lens")
system = CausticsStrongLensSystem(
    lens, lens_parameters={LENS_PARAMETERS!r}, z_s={Z_S!r}, max_images=5, node_label="system", **{MESH!r}
)
out = system.sample_parameters(num_samples=2, rng_info=np.random.default_rng(5))["system"]
np.savez(sys.argv[1], **{{name: out[name] for name in system.OUTPUTS}})
"""


def _run_case(caustics_backend, path):
    """Run ``_CASE`` in a new process under the given caustics backend and return its outputs."""
    env = os.environ | {
        "CASKADE_BACKEND": caustics_backend,
        "PYTHONPATH": str(Path(lightcurvelynx.__file__).parents[1]),
    }
    result = subprocess.run(
        [sys.executable, "-c", _CASE, str(path)], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    with np.load(path) as outputs:
        return dict(outputs)


def test_jax_backend_matches_torch(tmp_path):
    """Test that a fixed two-system case gives the same outputs under caustics' jax and torch backends."""
    pytest.importorskip("jax")
    jax_out = _run_case("jax", tmp_path / "jax.npz")
    torch_out = _run_case("torch", tmp_path / "torch.npz")
    assert_array_equal(jax_out["num_images"], torch_out["num_images"])
    for name in CausticsStrongLensSystem.OUTPUTS:
        assert_allclose(jax_out[name], torch_out[name], rtol=1e-6, equal_nan=True, err_msg=name)
