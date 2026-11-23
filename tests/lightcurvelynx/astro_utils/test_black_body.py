import astropy.constants as const
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling.physical_models import BlackBody
from scipy.integrate import trapezoid

from lightcurvelynx.astro_utils.black_body import (
    black_body_luminosity_density_per_solid,
    eval_blackbody,
)


@pytest.mark.parametrize("temp", [1000, 3100, 5500, 6003, 8050, 9000])
def test_eval_blackbody(temp):
    """Test that eval_blackbody function returns the same values as we would
    get from astropy's BlackBody class.
    """
    waves = np.linspace(1000, 10000, 101)

    # Compute astropy's version of the SED.
    bb = BlackBody(temperature=temp * u.K)
    sed_astropy = bb(waves * u.AA)

    # Compute our custom implementation of the SED.
    sed_custom = eval_blackbody(temp, waves)

    # Compare the two SEDs.
    assert np.allclose(sed_custom, sed_astropy.value, rtol=1e-6)


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
