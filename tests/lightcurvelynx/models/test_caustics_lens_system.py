import pickle  # noqa: F401 (used starting Task 8's pickling tests)
import warnings  # noqa: F401 (used starting Task 3's warning-escalation tests)
from concurrent.futures import ThreadPoolExecutor  # noqa: F401 (used starting Task 8's parallel-safety tests)

import numpy as np  # noqa: F401 (used starting Task 2)
import pytest
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401 (used starting Task 2)

torch = pytest.importorskip("torch")
caustics = pytest.importorskip("caustics")
adaptive = pytest.importorskip("caustics.lenses.func.adaptive")

from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode  # noqa: E402,F401,I001 (used starting Task 2)
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc  # noqa: E402,F401 (used starting Task 2)
from lightcurvelynx.models.caustics_lens_system import (  # noqa: E402
    CausticsStrongLensSystem,
    _lens_param_index,
)
# Task 2 adds `_inside_caustics` to this import. Task 5 adds the LinearTimeModel,
# ResolvedStrongLensModel and simulate_lightcurves imports here, at the top.

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
    """Return the test lens with every value set (static), in float64, for direct caustics calls."""
    return _make_lens(**VALUES, z_l=Z_L, z_s=Z_S).to(dtype=torch.float64)


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
