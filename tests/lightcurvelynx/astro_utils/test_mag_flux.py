"""Test magnitude-flux conversion utilities."""

import numpy as np
from lightcurvelynx.astro_utils.mag_flux import flux2mag, mag2flux


def test_flux2mag():
    """Test that mag2flux is correct."""
    flux = np.array([3631e9, 1e9, 3631])
    desired_mag = np.array([0, 8.9, 22.5])
    np.testing.assert_allclose(flux2mag(flux), desired_mag, atol=1e-3)


def test_mag2flux():
    """Test that flux2mag is correct."""
    mag = np.array([0, 8.9, 8.9 + 2.5 * 9])
    desired_flux = np.array([3631e9, 1e9, 1])
    np.testing.assert_allclose(mag2flux(mag), desired_flux, rtol=1e-3)


def test_mag2flux2mag():
    """Test that mag2flux inverts flux2mag."""
    rng = np.random.default_rng(42)
    mag = rng.uniform(-10, 30, 1024)
    flux = mag2flux(mag)
    mag2 = flux2mag(flux)
    np.testing.assert_allclose(mag, mag2, rtol=1e-10)


def test_flux2mag2flux():
    """Test that flux2mag inverts mag2flux."""
    rng = np.random.default_rng(43)
    flux = rng.uniform(1e-3, 1e3, 1024)
    mag = flux2mag(flux)
    flux2 = mag2flux(mag)
    np.testing.assert_allclose(flux, flux2, rtol=1e-10)
