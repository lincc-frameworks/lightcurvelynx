import numpy as np
from astropy.cosmology import FlatLambdaCDM

from lightcurvelynx.astro_utils.snia_utils import num_snia_per_redshift_bin, snia_volumetric_rates


def test_snia_volumetric_rates():
    """
    Test that the volumetric rates are correctly calculated.
    """

    z = np.array([0.0, 1.0])
    expected_rates = [2.27e-5, 7.375e-5]
    rates = snia_volumetric_rates(z)

    assert np.allclose(rates, expected_rates)


def test_num_snia_per_redshift_bin():
    """
    Test that the number of SN Ia calculated matches what it would be in a small bin without integration.
    """

    H0 = 70.0
    Omega_m = 0.3

    # zmean = 0.073
    zmin = 0.07
    zmax = 0.08
    znbins = 1
    nsn, z = num_snia_per_redshift_bin(zmin, zmax, znbins, H0=H0, Omega_m=Omega_m)

    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

    expected_rate = 2.42e-5
    expected_nsn = expected_rate * (cosmo.comoving_volume(zmax) - cosmo.comoving_volume(zmin))
    expected_nsn = expected_nsn.value

    nsn_total = np.sum(nsn)

    assert z.size == znbins
    assert nsn.size == znbins
    assert np.allclose(nsn_total, expected_nsn, rtol=0.1)
