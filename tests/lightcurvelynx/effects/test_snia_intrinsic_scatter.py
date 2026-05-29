import pytest
import numpy as np
from lightcurvelynx.effects.snia_intrinsic_scatter import SNIaIntrinsicScatter

def test_invalid_model() -> None:
    """Test that we raise an error if we specify an invalid model."""
    coh_scatter = SNIaIntrinsicScatter(modelpars={"modelname": "COH", "sigma": 0.1})
    with pytest.raises(ValueError):
        coh_scatter.apply(np.full((5, 3), 100.0), wavelengths=np.array([4000.0, 5000.0, 6000.0]), modelpars={"modelname": "INVALID_MODEL", "sigma": 0.1})

def test_coh_scatter() -> None:
    """Test that we can apply COH intrinsic scatter."""
    coh_scatter = SNIaIntrinsicScatter(modelpars={"modelname": "COH", "sigma": 0.1})

    # We can apply the scatter.
    flux = np.full((5, 3), 100.0)
    wavelengths = np.array([4000.0, 5000.0, 6000.0])
    flux_new = coh_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "COH", "sigma": 0.1})
    assert len(np.unique(flux_new)) == 15
    assert np.sum(np.abs(flux_new - 100.0) <= 0.5) >= 14

    # We can override the default value using the parameters.
    flux_new = coh_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "COH", "sigma": 1.0})
    assert len(np.unique(flux_new)) == 15
    assert np.sum(np.abs(flux_new - 100.0) <= 5.0) >= 14

def test_g10_scatter() -> None:
    """Test that we can apply G10 intrinsic scatter."""
    g10_scatter = SNIaIntrinsicScatter(modelpars={"modelname": "G10"})

    # We can apply the scatter.
    flux = np.full((5, 3), 100.0)
    wavelengths = np.array([4000.0, 5000.0, 6000.0])
    flux_new = g10_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "G10"})
    assert len(np.unique(flux_new)) == 15
    assert np.sum(np.abs(flux_new - 100.0) <= 0.5) >= 14

def test_c11_scatter() -> None:
    """Test that we can apply C11 intrinsic scatter."""
    c11_scatter = SNIaIntrinsicScatter(modelpars={"modelname": "C11"})

    # We can apply the scatter.
    flux = np.full((5, 3), 100.0)
    wavelengths = np.array([4000.0, 5000.0, 6000.0])
    flux_new = c11_scatter.apply(flux, wavelengths=wavelengths, modelpars={"modelname": "C11"})
    assert len(np.unique(flux_new)) == 15
    assert np.sum(np.abs(flux_new - 100.0) <= 0.5) >= 14