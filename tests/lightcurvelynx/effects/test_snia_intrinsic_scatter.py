import numpy as np
import pytest
from lightcurvelynx.effects.snia_intrinsic_scatter import (
    _C11_COV_KNOTS,
    _C11_KNOT_WAVELENGTHS,
    SNIaIntrinsicScatter,
)


def test_invalid_model() -> None:
    """Test that we raise an error if we specify an invalid model."""
    coh_scatter = SNIaIntrinsicScatter(modelpars={"modelname": "INVALID_MODEL", "sigma": 0.1})
    with pytest.raises(ValueError):
        coh_scatter.apply(
            np.full((5, 3), 100.0),
            wavelengths=np.array([4000.0, 5000.0, 6000.0]),
            modelpars={"modelname": "INVALID_MODEL", "sigma": 0.1},
        )


def test_coh_scatter() -> None:
    """Test that we can apply COH intrinsic scatter."""
    coh_scatter = SNIaIntrinsicScatter(modelpars={"modelname": "COH", "sigma": 0.1})

    # COH is coherent: same magnitude shift at all wavelengths and not dependent on time.
    flux = np.full((5, 3), 100.0)
    wavelengths = np.array([4000.0, 5000.0, 6000.0])
    flux_new = coh_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "COH", "sigma": 0.1})
    assert len(np.unique(flux_new)) == 1
    assert np.sum(np.abs(-2.5 * np.log10(flux_new / 100.0)) <= 0.5) >= 4

    # We can override sigma via modelpars kwarg; larger sigma → larger scatter.
    flux_new = coh_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "COH", "sigma": 1.0})
    assert len(np.unique(flux_new)) == 1
    assert np.sum(np.abs(-2.5 * np.log10(flux_new / 100.0)) <= 5.0) >= 4


def test_g10_scatter() -> None:
    """Test that we can apply G10 intrinsic scatter."""
    g10_scatter = SNIaIntrinsicScatter(modelpars={"modelname": "G10"})

    assert g10_scatter.modelpars["modelname"] == "G10"

    # Scatter is drawn once per apply() call and broadcast across all epochs,
    # so T=5 epochs all receive the same wavelength-dependent shift → 3 unique values (one per wavelength).
    flux = np.full((5, 3), 100.0)
    wavelengths = np.array([3000.0, 5000.0, 8000.0])
    flux_new = g10_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "G10"})
    assert len(np.unique(flux_new)) == 3
    # Scatter varies with wavelength (chromatic), so values differ across the 3 wavelengths.
    assert not np.allclose(flux_new[0], flux_new[0, 0])


def test_c11_scatter() -> None:
    """Test that we can apply C11 intrinsic scatter."""
    c11_scatter = SNIaIntrinsicScatter(modelpars={"modelname": "C11"})

    # Scatter is drawn once per apply() call and broadcast across all epochs,
    # so T=5 epochs all receive the same wavelength-dependent shift → 3 unique values (one per wavelength).
    flux = np.full((5, 3), 100.0)
    wavelengths = np.array([4000.0, 5000.0, 6000.0])
    flux_new = c11_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "C11"})
    assert len(np.unique(flux_new)) == 3
    # Scatter varies with wavelength (chromatic), so values differ across the 3 wavelengths.
    assert not np.allclose(flux_new[0], flux_new[0, 0])


def test_invalid_interp_method() -> None:
    """Test that an invalid interp_method raises a ValueError at construction."""
    with pytest.raises(ValueError):
        SNIaIntrinsicScatter(modelpars={"modelname": "G10"}, interp_method="spooky")


def test_output_shape_and_positive() -> None:
    """Test that output shape matches input and all flux values are positive for all models."""
    flux = np.full((3, 7), 100.0)
    wavelengths = np.linspace(3500, 8000, 7)
    for model in ["COH", "G10", "C11"]:
        result = SNIaIntrinsicScatter(modelpars={"modelname": model}).apply(flux, wavelengths=wavelengths)
        assert result.shape == (3, 7), f"{model}: unexpected shape {result.shape}"
        assert np.all(result > 0), f"{model}: non-positive flux values"


def test_seed_reproducibility() -> None:
    """Test that the same snia_scatter_seed produces identical output and different seeds differ."""
    flux = np.full((1, 5), 100.0)
    wavelengths = np.linspace(3500, 8000, 5)
    for model in ["G10", "C11"]:
        eff = SNIaIntrinsicScatter(modelpars={"modelname": model})
        out1 = eff.apply(flux, wavelengths=wavelengths, snia_scatter_seed=42)
        out2 = eff.apply(flux, wavelengths=wavelengths, snia_scatter_seed=42)
        out3 = eff.apply(flux, wavelengths=wavelengths, snia_scatter_seed=99)
        assert np.allclose(out1, out2), f"{model}: same seed gave different results"
        assert not np.allclose(out1, out3), f"{model}: different seeds gave identical results"


def test_interp_methods_agree_at_nodes() -> None:
    """All interpolation methods must be exact at node wavelengths and differ between them."""
    # Use two adjacent C11 knot wavelengths plus a midpoint between them.
    wavelengths = np.array([4390.0, 4900.0, 5490.0])
    flux = np.full((1, 3), 100.0)

    results = {}
    for method in ["sine", "linear", "pchip", "cubic"]:
        eff = SNIaIntrinsicScatter(modelpars={"modelname": "G10", "coh_sigma": 0.0}, interp_method=method)
        results[method] = eff.apply(flux, wavelengths=wavelengths, snia_scatter_seed=7)

    # At exact node wavelengths (columns 0 and 2) all methods must agree.
    for method in ["linear", "pchip", "cubic"]:
        assert np.allclose(
            results["sine"][:, 0], results[method][:, 0]
        ), f"sine and {method} disagree at left node"
        assert np.allclose(
            results["sine"][:, 2], results[method][:, 2]
        ), f"sine and {method} disagree at right node"

    # Between the nodes the shapes differ — sine and linear should not match.
    assert not np.allclose(results["sine"][:, 1], results["linear"][:, 1])


def test_sine_interp_properties() -> None:
    """Test _sine_interp: exact at nodes, clamped outside range, bounded between nodes."""
    eff = SNIaIntrinsicScatter(modelpars={"modelname": "COH"})
    node_waves = np.array([3000.0, 5000.0, 7000.0])
    node_values = np.array([1.0, 3.0, 2.0])

    # Exact at node wavelengths.
    assert np.allclose(eff._sine_interp(node_waves, node_values, node_waves), node_values)

    # Clamped to first node value below the range.
    result_below = eff._sine_interp(node_waves, node_values, np.array([1000.0, 2000.0]))
    assert np.allclose(result_below, node_values[0])

    # Clamped to last node value above the range.
    result_above = eff._sine_interp(node_waves, node_values, np.array([8000.0, 9000.0]))
    assert np.allclose(result_above, node_values[-1])

    # Between nodes [3000, 5000] (values 1→3): result must lie strictly between them.
    result_mid = eff._sine_interp(node_waves, node_values, np.array([4000.0]))
    assert node_values[0] < result_mid[0] < node_values[1]


def test_coh_sigma_zero() -> None:
    """G10 and C11 with coh_sigma=0 still produce non-uniform chromatic scatter."""
    flux = np.full((1, 5), 100.0)
    wavelengths = np.linspace(3500, 8000, 5)
    for model in ["G10", "C11"]:
        eff = SNIaIntrinsicScatter(modelpars={"modelname": model, "coh_sigma": 0.0})
        result = eff.apply(flux, wavelengths=wavelengths, snia_scatter_seed=42)
        # Scatter should vary across wavelengths (chromatic); first and last values should differ.
        assert not np.isclose(
            result[0, 0], result[0, -1]
        ), f"{model}: scatter appears uniform with coh_sigma=0"


def test_g10_coh_sigma_override() -> None:
    """Different coh_sigma values produce different scatter even with the same seed."""
    flux = np.full((1, 5), 100.0)
    wavelengths = np.linspace(3500, 8000, 5)

    eff_small = SNIaIntrinsicScatter(modelpars={"modelname": "G10", "coh_sigma": 0.01})
    eff_large = SNIaIntrinsicScatter(modelpars={"modelname": "G10", "coh_sigma": 1.0})
    out_small = eff_small.apply(flux, wavelengths=wavelengths, snia_scatter_seed=42)
    out_large = eff_large.apply(flux, wavelengths=wavelengths, snia_scatter_seed=42)
    assert not np.allclose(out_small, out_large)


def test_c11_cov_knots_positive_semidefinite() -> None:
    """_C11_COV_KNOTS must be positive semi-definite (all eigenvalues >= 0)."""
    eigenvalues = np.linalg.eigvalsh(_C11_COV_KNOTS)
    assert np.all(eigenvalues >= -1e-8), f"Negative eigenvalues found: {eigenvalues}"


def test_c11_knot_wavelengths_sorted() -> None:
    """_C11_KNOT_WAVELENGTHS must be strictly ascending (required by interpolation)."""
    assert np.all(np.diff(_C11_KNOT_WAVELENGTHS) > 0)
