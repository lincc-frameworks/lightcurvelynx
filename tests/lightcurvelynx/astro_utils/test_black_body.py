import astropy.constants as const
import numpy as np
from scipy.integrate import trapezoid

from lightcurvelynx.astro_utils.black_body import black_body_luminosity_density_per_solid


def test_bolometric_luminosity():
    """Test that integral of black-body luminosity density is equal to the bolomertric luminosity"""
    speed_of_light = const.c.cgs.value
    bolometric_luminosity = const.L_sun.cgs.value
    radius = const.R_sun.cgs.value
    teff = (bolometric_luminosity / (4 * np.pi * radius**2 * const.sigma_sb.cgs.value)) ** 0.25

    # From 10 to 100,000 Angstroms, in cm
    wavelengths = np.geomspace(100_000, 10, 1001) * 1e-8
    frequencies = speed_of_light / wavelengths

    # Multiply dL_nu / d nu / d Omega by 4pi and integrate over frequency
    luminosity_density = 4 * np.pi * black_body_luminosity_density_per_solid(teff, radius, wavelengths)
    total_luminosity = trapezoid(y=luminosity_density, x=frequencies)

    np.testing.assert_allclose(total_luminosity, bolometric_luminosity, rtol=1e-3)
