"""Tests for the Milky Way stellar density models."""

import numpy as np
import pytest
from astropy.coordinates import ICRS
from lightcurvelynx.astro_utils.milky_way_density import (
    MilkyWayDensityBase,
    MilkyWayDensityJuric2008,
)


class TestMilkyWayDensityJuric2008:
    """Tests for MilkyWayDensityJuric2008."""

    def test_instantiation_default(self):
        """Test that the model can be instantiated with default parameters."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        assert model.n_grid == 64
        assert model.thin_disk_weight == 1.0
        assert model.thick_disk_weight == 1.0
        assert model.halo_weight == 1.0

    def test_instantiation_custom_weights(self):
        """Test that the model can be instantiated with custom component weights."""
        model = MilkyWayDensityJuric2008(
            n_grid=32,
            thin_disk_weight=2.0,
            thick_disk_weight=0.5,
            halo_weight=0.1,
        )
        assert model.thin_disk_weight == 2.0
        assert model.thick_disk_weight == 0.5
        assert model.halo_weight == 0.1

    def test_dens_at_sun(self):
        """Test that the total density at the Sun's position is positive."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        d = model.dens(model.sun_rho_kpc, 0.0, model.sun_z_kpc)
        assert d > 0.0

    def test_thin_disk_dens_scalar(self):
        """Test the thin disk density at a scalar position."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        d = model.thin_disk_dens(model.sun_rho_kpc, 0.0, 0.0)
        assert d > 0.0

    def test_thick_disk_dens_scalar(self):
        """Test the thick disk density at a scalar position."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        d = model.thick_disk_dens(model.sun_rho_kpc, 0.0, 0.0)
        assert d > 0.0

    def test_halo_dens_scalar(self):
        """Test the halo density at a scalar position."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        d = model.halo_dens(model.sun_rho_kpc, 0.0, 0.0)
        assert d > 0.0

    def test_dens_array(self):
        """Test that dens accepts and returns arrays."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        rho = np.array([1.0, 5.0, 10.0])
        z = np.array([0.0, 0.5, -1.0])
        d = model.dens(rho, 0.0, z)
        assert d.shape == (3,)
        assert np.all(d > 0.0)

    def test_dens_decreases_with_z(self):
        """Test that the density decreases as |z| increases (mid-plane should be denser)."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        rho = model.sun_rho_kpc
        d_mid = model.dens(rho, 0.0, 0.0)
        d_above = model.dens(rho, 0.0, 1.0)
        d_far = model.dens(rho, 0.0, 5.0)
        assert d_mid > d_above > d_far

    def test_cumulative_distribution_built(self):
        """Test that the precomputed cumulative distribution is valid."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        assert model._dens_flat_cum[-1] == pytest.approx(1.0)
        assert np.all(np.diff(model._dens_flat_cum) >= 0.0)

    def test_sample_galactic_cylindrical_shape(self):
        """Test that sampling returns arrays of the correct size."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        n = 50
        rho, phi, z = model.sample_galactic_cylindrical(n_samples=n, rng=np.random.default_rng(0))
        assert rho.shape == (n,)
        assert phi.shape == (n,)
        assert z.shape == (n,)

    def test_sample_galactic_cylindrical_ranges(self):
        """Test that sampled cylindrical coordinates are within the grid bounds."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        n = 200
        rho, phi, z = model.sample_galactic_cylindrical(n_samples=n, rng=np.random.default_rng(1))
        assert np.all(rho >= model.rho_min_kpc)
        assert np.all(rho <= model.rho_max_kpc)
        assert np.all(phi >= 0.0)
        assert np.all(phi <= 2.0 * np.pi)
        assert np.all(z >= model.z_min_kpc)
        assert np.all(z <= model.z_max_kpc)

    def test_sample_galactic_cylindrical_reproducible(self):
        """Test that sampling is reproducible with the same seed."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        rho1, phi1, z1 = model.sample_galactic_cylindrical(n_samples=20, rng=np.random.default_rng(42))
        rho2, phi2, z2 = model.sample_galactic_cylindrical(n_samples=20, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(rho1, rho2)
        np.testing.assert_array_equal(phi1, phi2)
        np.testing.assert_array_equal(z1, z2)

    def test_sample_icrs_returns_icrs(self):
        """Test that sample_icrs returns an astropy ICRS coordinate object."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        coords = model.sample_icrs(n_samples=5, rng=np.random.default_rng(7))
        assert isinstance(coords, ICRS)
        assert len(coords) == 5

    def test_sample_icrs_ra_dec_ranges(self):
        """Test that sampled RA and dec values are in valid ranges."""
        model = MilkyWayDensityJuric2008(n_grid=64)
        n = 100
        coords = model.sample_icrs(n_samples=n, rng=np.random.default_rng(99))
        ra = coords.ra.deg
        dec = coords.dec.deg
        assert np.all(ra >= 0.0)
        assert np.all(ra < 360.0)
        assert np.all(dec >= -90.0)
        assert np.all(dec <= 90.0)

    def test_weight_zero_component_removed(self):
        """Test that zeroing a component weight effectively removes that component."""
        model_no_halo = MilkyWayDensityJuric2008(n_grid=64, halo_weight=0.0)
        model_full = MilkyWayDensityJuric2008(n_grid=64)

        rho = np.array([15.0, 18.0])  # Far out where halo contribution is significant
        z = np.array([8.0, 8.0])
        d_no_halo = model_no_halo.dens(rho, 0.0, z)
        d_full = model_full.dens(rho, 0.0, z)
        assert np.all(d_full > d_no_halo)

    def test_is_subclass_of_base(self):
        """Test that MilkyWayDensityJuric2008 is a subclass of MilkyWayDensityBase."""
        assert issubclass(MilkyWayDensityJuric2008, MilkyWayDensityBase)

    def test_base_class_is_abstract(self):
        """Test that MilkyWayDensityBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MilkyWayDensityBase()
