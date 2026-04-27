"""Milky Way stellar density models for sampling sky positions.

This module provides density models that describe the spatial distribution of
stars in the Milky Way. These models can be used to generate realistic
sky positions for simulated stellar populations.

References
----------
* Juric et al., 2008, ApJ, 673, 864
  https://iopscience.iop.org/article/10.1086/523619/pdf
"""

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, Galactocentric


class MilkyWayDensityBase(ABC):
    """Abstract base class for Milky Way stellar density models.

    Subclasses must implement the :meth:`dens` method which returns the
    unnormalised stellar number density at a given cylindrical galactocentric
    position (rho, phi, z).

    The constructor precomputes a 2-D grid over (rho, z) and a flat cumulative
    distribution that is used for fast inverse-transform sampling.

    Attributes
    ----------
    sun_z_kpc : float
        Height of the Sun above the Galactic mid-plane in kpc.
    sun_rho_kpc : float
        Galactocentric distance of the Sun in the mid-plane in kpc.
    rho_min_kpc : float
        Minimum cylindrical radius for the sampling grid in kpc.
    rho_max_kpc : float
        Maximum cylindrical radius for the sampling grid in kpc.
    z_min_kpc : float
        Minimum height above/below the mid-plane for the sampling grid in kpc.
    z_max_kpc : float
        Maximum height above/below the mid-plane for the sampling grid in kpc.
    n_grid : int
        Number of grid points per axis.

    Parameters
    ----------
    n_grid : int, optional
        Number of grid points along each axis of the (rho, z) grid.
        Larger values give more accurate sampling at the cost of memory and
        initialisation time. Default: 1024
    """

    sun_z_kpc = None
    sun_rho_kpc = None

    rho_min_kpc = 0.0
    rho_max_kpc = 20.0
    z_min_kpc = -20.0
    z_max_kpc = 20.0

    def __init__(self, n_grid=1024):
        self.n_grid = n_grid

        # Build a 2-D grid in (rho, z).  Note: meshgrid returns arrays with
        # shapes (n_grid, n_grid); we store them for coordinate look-up later.
        rho_vals = np.linspace(self.rho_min_kpc, self.rho_max_kpc, n_grid)
        z_vals = np.linspace(self.z_min_kpc, self.z_max_kpc, n_grid)
        self.rho, self.z = np.meshgrid(rho_vals, z_vals)

        # Evaluate the density on the grid.  We multiply by rho because the
        # volume element in cylindrical coordinates is rho d(phi) d(rho) dz,
        # so integrating over phi gives 2*pi*rho * density(rho, phi, z).
        self._dens_grid = self.rho * self.dens(self.rho, 0.0, self.z)

        # Build the flat cumulative distribution (normalised to [0, 1]).
        flat_cum = np.cumsum(self._dens_grid.ravel())
        flat_cum /= flat_cum[-1]
        self._dens_flat_cum = flat_cum

    @abstractmethod
    def dens(self, rho, phi, z):
        """Return the unnormalised stellar density at cylindrical position.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Cylindrical galactocentric radius in kpc.
        phi : float or numpy.ndarray
            Azimuthal angle in radians (0 toward Sun).
        z : float or numpy.ndarray
            Height above the Galactic mid-plane in kpc.

        Returns
        -------
        density : float or numpy.ndarray
            Unnormalised stellar density (arbitrary units).
        """

    def sample_galactic_cylindrical(self, n_samples=1, rng=None):
        """Sample positions in cylindrical Galactic coordinates.

        Uses inverse-transform sampling on the precomputed cumulative
        distribution to draw (rho, phi, z) positions.

        Parameters
        ----------
        n_samples : int, optional
            Number of positions to draw. Default: 1
        rng : numpy.random.Generator, optional
            Random number generator to use. If *None* a fresh generator is
            created with ``numpy.random.default_rng()``.

        Returns
        -------
        rho : numpy.ndarray
            Cylindrical galactocentric radii in kpc, shape (n_samples,).
        phi : numpy.ndarray
            Azimuthal angles in radians, shape (n_samples,).
        z : numpy.ndarray
            Heights above the Galactic mid-plane in kpc, shape (n_samples,).
        """
        rng = np.random.default_rng(rng)

        # Draw uniform random numbers and find the corresponding grid indices
        # via binary search on the cumulative distribution.
        u_vals = rng.random(size=n_samples)
        idx = np.searchsorted(self._dens_flat_cum, u_vals)
        # Clamp to valid range (edge case when u_vals == 1.0).
        idx = np.clip(idx, 0, len(self._dens_flat_cum) - 1)

        rho = self.rho.ravel()[idx]
        z = self.z.ravel()[idx]
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)

        return rho, phi, z

    def sample_icrs(self, n_samples=1, rng=None):
        """Sample sky positions distributed according to the density model.

        Positions are drawn from the precomputed distribution and converted
        from Galactocentric cylindrical coordinates to ICRS (RA, dec).

        Parameters
        ----------
        n_samples : int, optional
            Number of positions to draw. Default: 1
        rng : numpy.random.Generator, optional
            Random number generator to use. If *None* a fresh generator is
            created with ``numpy.random.default_rng()``.

        Returns
        -------
        coords : astropy.coordinates.ICRS
            ICRS sky coordinates for the sampled positions.
        """
        rho, phi, z = self.sample_galactic_cylindrical(n_samples=n_samples, rng=rng)

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        gc = Galactocentric(
            x=x * u.kpc,
            y=y * u.kpc,
            z=z * u.kpc,
            z_sun=self.sun_z_kpc * u.kpc,
            galcen_distance=self.sun_rho_kpc * u.kpc,
        )
        return gc.transform_to(ICRS())


class MilkyWayDensityJuric2008(MilkyWayDensityBase):
    """Milky Way stellar density model from Juric et al. (2008).

    Implements the smooth stellar number-density model described in Table 10 of
    Juric et al. (2008), composed of a thin disk, a thick disk, and a spherical
    halo component.

    References
    ----------
    * Juric et al., 2008, ApJ, 673, 864
      https://iopscience.iop.org/article/10.1086/523619/pdf

    Attributes
    ----------
    sun_rho_kpc : float
        Galactocentric distance of the Sun, 8 kpc.
    sun_z_kpc : float
        Height of the Sun above the Galactic mid-plane, 0.024 kpc.
    l_thin_kpc : float
        Scale length of the thin disk in kpc.
    h_thin_kpc : float
        Scale height of the thin disk in kpc.
    dens_thick_to_thin : float
        Local normalisation of the thick disk relative to the thin disk.
    l_thick_kpc : float
        Scale length of the thick disk in kpc.
    h_thick_kpc : float
        Scale height of the thick disk in kpc.
    dens_halo_to_thin : float
        Local normalisation of the halo relative to the thin disk.
    ellipticity_halo : float
        Axis ratio (c/a) of the halo.
    power_order_halo : float
        Power-law index of the halo density profile.
    thin_disk_weight : float
        Multiplicative weight applied to the thin disk component. Default: 1.0
    thick_disk_weight : float
        Multiplicative weight applied to the thick disk component. Default: 1.0
    halo_weight : float
        Multiplicative weight applied to the halo component. Default: 1.0

    Parameters
    ----------
    n_grid : int, optional
        Number of grid points along each axis of the (rho, z) grid.
        Default: 1024
    thin_disk_weight : float, optional
        Multiplicative weight applied to the thin disk component. Default: 1.0
    thick_disk_weight : float, optional
        Multiplicative weight applied to the thick disk component. Default: 1.0
    halo_weight : float, optional
        Multiplicative weight applied to the halo component. Default: 1.0

    Examples
    --------
    >>> import numpy as np
    >>> model = MilkyWayDensityJuric2008(n_grid=64)
    >>> rho, phi, z = model.sample_galactic_cylindrical(n_samples=10, rng=np.random.default_rng(42))
    >>> len(rho)
    10
    """

    # Juric et al. 2008, Table 10
    sun_rho_kpc = 8.0
    sun_z_kpc = 0.024
    l_thin_kpc = 2.6
    h_thin_kpc = 0.3
    dens_thick_to_thin = 0.12
    l_thick_kpc = 3.6
    h_thick_kpc = 0.9
    dens_halo_to_thin = 0.0051
    ellipticity_halo = 0.64
    power_order_halo = 2.77

    rho_min_kpc = 1.0
    rho_max_kpc = 20.0
    z_min_kpc = -10.0
    z_max_kpc = 10.0

    def __init__(
        self,
        n_grid=1024,
        thin_disk_weight=1.0,
        thick_disk_weight=1.0,
        halo_weight=1.0,
    ):
        self.thin_disk_weight = thin_disk_weight
        self.thick_disk_weight = thick_disk_weight
        self.halo_weight = halo_weight
        super().__init__(n_grid=n_grid)

    def _disk_dens(self, rho, z, scale_length, scale_height):
        """Evaluate an exponential disk density profile.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Cylindrical galactocentric radius in kpc.
        z : float or numpy.ndarray
            Height above the Galactic mid-plane in kpc.
        scale_length : float
            Disk scale length in kpc.
        scale_height : float
            Disk scale height in kpc.

        Returns
        -------
        density : float or numpy.ndarray
            Normalised disk density.
        """
        return np.exp((self.sun_rho_kpc - rho) / scale_length - np.abs(z) / scale_height)

    def thin_disk_dens(self, rho, phi, z):
        """Return the thin disk component of the density.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Cylindrical galactocentric radius in kpc.
        phi : float or numpy.ndarray
            Azimuthal angle in radians (unused; model is axisymmetric).
        z : float or numpy.ndarray
            Height above the Galactic mid-plane in kpc.

        Returns
        -------
        density : float or numpy.ndarray
            Thin disk density (normalised to 1 at the Sun's position).
        """
        return self._disk_dens(rho, z, scale_length=self.l_thin_kpc, scale_height=self.h_thin_kpc)

    def thick_disk_dens(self, rho, phi, z):
        """Return the thick disk component of the density.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Cylindrical galactocentric radius in kpc.
        phi : float or numpy.ndarray
            Azimuthal angle in radians (unused; model is axisymmetric).
        z : float or numpy.ndarray
            Height above the Galactic mid-plane in kpc.

        Returns
        -------
        density : float or numpy.ndarray
            Thick disk density (normalised to ``dens_thick_to_thin`` at the
            Sun's position relative to the thin disk).
        """
        return self.dens_thick_to_thin * self._disk_dens(
            rho, z, scale_length=self.l_thick_kpc, scale_height=self.h_thick_kpc
        )

    def halo_dens(self, rho, phi, z):
        """Return the stellar halo component of the density.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Cylindrical galactocentric radius in kpc.
        phi : float or numpy.ndarray
            Azimuthal angle in radians (unused; model is axisymmetric).
        z : float or numpy.ndarray
            Height above the Galactic mid-plane in kpc.

        Returns
        -------
        density : float or numpy.ndarray
            Halo density (normalised to ``dens_halo_to_thin`` at the Sun's
            position relative to the thin disk).
        """
        return self.dens_halo_to_thin * np.power(
            self.sun_rho_kpc / np.hypot(rho, z / self.ellipticity_halo),
            self.power_order_halo,
        )

    def dens(self, rho, phi, z):
        """Return the total stellar density at a given position.

        The total density is the weighted sum of the thin disk, thick disk,
        and halo components.

        Parameters
        ----------
        rho : float or numpy.ndarray
            Cylindrical galactocentric radius in kpc.
        phi : float or numpy.ndarray
            Azimuthal angle in radians (unused; model is axisymmetric).
        z : float or numpy.ndarray
            Height above the Galactic mid-plane in kpc.

        Returns
        -------
        density : float or numpy.ndarray
            Total unnormalised stellar density.
        """
        return (
            self.thin_disk_weight * self.thin_disk_dens(rho, phi, z)
            + self.thick_disk_weight * self.thick_disk_dens(rho, phi, z)
            + self.halo_weight * self.halo_dens(rho, phi, z)
        )
