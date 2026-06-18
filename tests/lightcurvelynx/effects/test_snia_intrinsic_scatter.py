import numpy as np
import pytest
from lightcurvelynx.effects.snia_intrinsic_scatter import SNIaIntrinsicScatter


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

    # COH is coherent: same magnitude shift at all wavelengths per epoch → 5 unique values (one per epoch).
    flux = np.full((5, 3), 100.0)
    wavelengths = np.array([4000.0, 5000.0, 6000.0])
    flux_new = coh_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "COH", "sigma": 0.1})
    assert len(np.unique(flux_new)) == 5
    assert np.sum(np.abs(-2.5 * np.log10(flux_new / 100.0)) <= 0.5) >= 4

    # We can override sigma via modelpars kwarg; larger sigma → larger scatter.
    flux_new = coh_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "COH", "sigma": 1.0})
    assert len(np.unique(flux_new)) == 5
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
