import astropy.units as u
import numpy as np
import scipy.integrate as integrate
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import norm
from scipy.stats.sampling import NumericalInversePolynomial

from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.math_nodes.scipy_random import NumericalInversePolynomialFunc


def snia_volumetric_rates(redshift, r0=2.27e-5, alpha=1.7):
    """
    SN Ia volumetric rate.
    r_v(z) = r0 * (1+z)^alpha （SNe Ia yr^-1 Mpc^-3 h_70^3）
    The default values are from Frohmaier et al. (2019).
    r0 = 2.27+/-0.19e-5
    alpha = 1.7+/-0.21

    Parameters
    ----------
    redshift: float or numpy.ndarray
        The redshift of the supernova
    r0: float
        The rate function parameter r0. Default is 2.27e-5.
    alpha: float
        The rate function parameter alpha. Default is 1.7

    Returns
    -------
    rate_vol: float or numpy.ndarray
        The volumetric rate of the supernova given the redshift
    """

    rate_vol = r0 * np.power(1.0 + redshift, alpha)

    return rate_vol


def num_snia_per_redshift_bin(
    zmin=0.001,
    zmax=10,
    znbins=20,
    solid_angle=None,
    H0=73.0,
    Omega_m=0.3,
    vol_rate_function=snia_volumetric_rates,
):
    """
    Calculate the number of SNe Ia in each redshift bin based on rates.

    r_v(z) = dN/dz
    V = comoving volume
    T = length of survey in years
    N = int r_v(z)dz * dV * dT

    Parameters
    ----------
    zmin: float
        Min redshift value for calculation.
    zmax: float
        Max redshift value for calculation.
    znbins: int
        Number of redshift bins for calculating SNe Ia numbers.
    solid_angle: float
        Solid angle for calculating the number of SNe (in sr).
    H0: float
        The Hubble Constant.
    Omega_m: float
        The matter density.
    vol_rate_function: Callable
        The function that defines the volumetric rate. Default is snia_volumetric_rates.

    Returns
    -------
    num_sn: numpy.ndarray
        Number of SNe Ia in each zbin per year.
    z_mean: numpy.ndarray
        Mean value for each redshift bin.
    """

    if solid_angle is None:
        solid_angle = 4 * np.pi

    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

    zarr = np.linspace(zmin, zmax, znbins + 1)

    int_arr = np.linspace(zarr[:-1], zarr[1:], 50, axis=1)
    z_mean = np.mean(int_arr, axis=1)

    dV = (
        solid_angle * cosmo.differential_comoving_volume(int_arr) * (H0 / 70.0) ** 3
    )  # * 4pi because differential_comoving_volume is per solid angle
    r_v = vol_rate_function(int_arr)
    dn_dz = r_v * dV.value

    num_sn = integrate.trapezoid(dn_dz, int_arr, axis=1)

    return num_sn, z_mean


class HostmassX1Distr:
    """
    A class that contains the pdf of the SALT x1 parameter given the hostmass

    Attributes
    ----------
    hostmass: float
        The hostmass value in units of log10(M/M_solar).

    Parameters
    ----------
    hostmass: float
        The hostmass value in units of log10(M/M_solar).
    """

    def __init__(self, hostmass):
        self.hostmass = hostmass

    def _p(self, x1, hostmass=9.0):
        """
        The probablity of having a value of x1 given a hostmass.

        Parameters
        ----------
        x1: numpy.ndarray
            The x1 value.
        hostmass: float
            The hostmass value in units of log10(M/M_solar).

        Returns
        -------
        p: numpy.ndarray
            The probablity.
        """

        p = np.exp(-(np.minimum(0, x1) ** 2))
        p = np.where(np.logical_and(x1 > -5, x1 < 5), p, 0.0)
        p = np.where(hostmass < 10.0, p, 1.0)

        return p

    def pdf(self, x1):
        """
        The pdf of x1 given hostmass.

        Parameters
        ----------
        x1: numpy.ndarray
            The x1 value.
        hostmass: float
            The hostmass value in units of log10(M/M_solar).

        Returns
        -------
        The pdf function of x1 given hostmass.

        """
        return self._p(x1, hostmass=self.hostmass) * norm.pdf(x1, loc=0, scale=1)


def _x0_from_distmod(distmod, x1, c, alpha, beta, m_abs):
    """Calculate the SALT3 x0 parameter given distance modulus based on Tripp relation.
    distmod = -2.5*log10(x0) + alpha * x1 - beta * c - m_abs + 10.635
    x0 = 10 ^ (-0.4* (distmod - alpha * x1 + beta * c + m_abs - 10.635))

    Parameters
    ----------
    distmod : float
        The distance modulus value (in mag).
    x1 : float
        The SALT3 x1 parameter.
    c : float
        The SALT3 c parameter.
    alpha : float
        The alpha parameter in the Tripp relation.
    beta : float
        The beta parameter in the Tripp relation.
    m_abs : float
        The absolute magnitude of SN Ia.

    Returns
    -------
    x0 : float
        The x0 parameter
    """
    x0 = np.power(10.0, -0.4 * (distmod - alpha * x1 + beta * c + m_abs - 10.635))

    return x0


class HostmassX1Func(NumericalInversePolynomialFunc):
    """A class for sampling from the HostmassX1Distr.

    Parameters
    ----------
    hostmass : function or constant
        The function or constant providing the hostmass value in units of log10(M/M_solar).
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, hostmass, **kwargs):
        # Since HostmassX1Distr can only take on two values (one for hostmass < 10.0 and one
        # for hostmass >= 10.0), we just create two distributions.
        self._dist_ge_10 = HostmassX1Distr(11.0)
        self._dist_lt_10 = HostmassX1Distr(9.0)
        super().__init__(
            dist=HostmassX1Distr,
            hostmass=hostmass,
            **kwargs,
        )

        # We override the inverse functions.
        self._inv_poly_ge_10 = NumericalInversePolynomial(self._dist_ge_10)
        self._inv_poly_lt_10 = NumericalInversePolynomial(self._dist_lt_10)
        self._vect_sample = None

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Sample from one of the two distributions depending on hostmass.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator or None, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        rng = rng_info if rng_info is not None else self._rng
        hostmass = self.get_param(graph_state, "hostmass")

        if graph_state.num_samples == 1:
            results = (
                self._inv_poly_ge_10.rvs(1, rng)[0] if hostmass >= 10 else self._inv_poly_lt_10.rvs(1, rng)[0]
            )
        else:
            results = np.zeros(graph_state.num_samples)

            # Batch generate samples for all points with hostmass >= 10.0
            ge_10_idx = hostmass >= 10.0
            results[ge_10_idx] = self._inv_poly_ge_10.rvs(np.count_nonzero(ge_10_idx), rng)

            # Batch generate samples for all points with hostmass < 10.0
            lt_10_idx = hostmass < 10.0
            results[lt_10_idx] = self._inv_poly_lt_10.rvs(np.count_nonzero(lt_10_idx), rng)

        self._save_results(results, graph_state)
        return results


class X0FromDistMod(FunctionNode):
    """A wrapper class for the _x0_from_distmod() function.

    Parameters
    ----------
    distmod : function or constant
        The function or constant providing the distance modulus value.
    x1 : function or constant
        The function or constant providing the x1 value.
    c : function or constant
        The function or constant providing the c value.
    alpha : function or constant
        The function or constant providing the alpha value.
    beta : function or constant
        The function or constant providing the beta value.
    m_abs : function or constant
        The function or constant providing the m_abs value.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, distmod, x1, c, alpha, beta, m_abs, **kwargs):
        # Call the super class's constructor with the needed information.
        super().__init__(
            func=_x0_from_distmod,
            distmod=distmod,
            x1=x1,
            c=c,
            alpha=alpha,
            beta=beta,
            m_abs=m_abs,
            **kwargs,
        )


class DistModFromRedshift(FunctionNode):
    """A wrapper class for the _distmod_from_redshift() function.

    Parameters
    ----------
    redshift : function or constant
        The function or constant providing the redshift value.
    H0 : constant
        The Hubble constant.
    Omega_m : constant
        The matter density Omega_m.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, redshift, H0=73.0, Omega_m=0.3, **kwargs):
        # Create the cosmology once for this node.
        if not isinstance(H0, float) or not isinstance(Omega_m, float):
            raise ValueError("H0 and Omega_m must be constants.")
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

        # Call the super class's constructor with the needed information.
        super().__init__(
            func=self._distmod_from_redshift,
            redshift=redshift,
            **kwargs,
        )

    def _distmod_from_redshift(self, redshift):
        """Compute distance modulus given redshift and cosmology.

        Parameters
        ----------
        redshift : float or numpy.ndarray
            The redshift value(s).

        Returns
        -------
        distmod : float or numpy.ndarray
            The distance modulus (in mag)
        """
        return self.cosmo.distmod(redshift).value


class SNCoordGivenPhysicalSep(FunctionNode):
    """A class for gernerating SN Coordinates given host coordinates and physical separations.

    Parameters
    ----------
    host_ra : function or constant
        Host galaxy RA in degree.
    host_dec : function or constant
        Host galaxy DEC in degree.
    physical_sep_kpc : function or constant
        The physical separation between host and SN in kpc.
    redshift : function or constant
        The function or constant providing the redshift value.
    H0 : constant
        The Hubble constant.
    Omega_m : constant
        The matter density Omega_m.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, host_ra, host_dec, physical_sep_kpc, redshift, H0=73.0, Omega_m=0.3, **kwargs):
        # Create the cosmology once for this node.
        if not isinstance(H0, float) or not isinstance(Omega_m, float):  # pragma: no cover
            raise ValueError("H0 and Omega_m must be constants.")
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

        # Call the super class's constructor with the needed information.
        super().__init__(
            func=self._sn_coord,
            host_ra=host_ra,
            host_dec=host_dec,
            physical_sep_kpc=physical_sep_kpc,
            redshift=redshift,
            outputs=["ra", "dec"],
            **kwargs,
        )

    def _host_sn_angular_separation(self, physical_sep_kpc, redshift):
        """
        Function for calculating host-sn angular separation.

        Parameters
        ----------
        physical_sep_kpc : float or numpy.ndarray
            The physical host-sn separation(s) in kpc.
        redshift: float or numpy.ndarray
            The redshift value(s).

        Returns
        -------
        angular_sep_arcsec : float or numpy.ndarray
            The angular host-sn separation(s) in arc seconds.
        """

        angular_diameter_distance = self.cosmo.angular_diameter_distance(redshift)
        angular_sep_rad = (physical_sep_kpc * u.kpc / angular_diameter_distance) * u.rad

        return angular_sep_rad

    def _sn_coord(self, host_ra, host_dec, physical_sep_kpc, redshift):
        """
        Function to generate SN coordinates given the host coordinates and angular separation.

        Parameters
        ----------
        host_ra : float or numpy.ndarray
            Host galaxy RA values (in degrees).
        host_dec : float or numpy.ndarray
            Host galaxy DEC values (in degrees).
        physical_sep_kpc : float or numpy.ndarray
            The physical host-sn separation(s) in kpc.
        redshift : float or numpy.ndarray
            The redshift to convert physical separation to angular separation.

        Returns
        -------
        ra : float or numpy.ndarray
            SN RA values (in degrees).
        dec : float or numpy.ndarray
            SN DEC values (in degrees).
        """

        self.angular_sep_rad = self._host_sn_angular_separation(physical_sep_kpc, redshift)
        center = SkyCoord(host_ra * u.deg, host_dec * u.deg, frame="icrs")
        rand_pa = 2 * np.pi * np.random.random(center.size) * u.rad  # random position angle

        sn_coord = center.directional_offset_by(rand_pa, self.angular_sep_rad)
        ra = sn_coord.ra.deg
        dec = sn_coord.dec.deg

        return ra, dec
