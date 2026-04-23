import numpy as np
from astropy import units as u
from astropy.modeling.physical_models import BlackBody


def black_body_luminosity_density_per_solid(temperature, radius, wavelengths):
    """Calculate the black-body luminosity density per solid angle.

    It is L_nu / (4 pi) for a spherical isotropic source.

    Parameters
    ----------
    temperature : float
        The effective temperature of the star, in kelvins.
    radius : float
        The radius of the star, in cm.
    wavelengths : numpy.ndarray
        A length N array of wavelengths, in cm.

    Returns
    -------
    luminosity_density : numpy.ndarray
        A length N array of luminosity density values.
        Output is in CGS units of erg/s/Hz/steradian.

    Note
    ----
    lightcurvelynx adopts nJy as the unit of flux density, so the output of this
    function may need to be converted to compatible units,
    e.g. multiply by 10^32.
    """
    black_body = BlackBody(temperature * u.K)
    # Convert intensity to units compatible with nJy flux density
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        intensity_per_freq = black_body(wavelengths * u.cm).to_value(
            u.erg / u.s / u.cm**2 / u.Hz / u.steradian
        )
    # In the Wien tail, finite-precision evaluation can overflow in intermediate
    # expm1 calculations; physically this corresponds to near-zero intensity.
    intensity_per_freq = np.nan_to_num(intensity_per_freq, nan=0.0, posinf=0.0, neginf=0.0)
    # Integral over solid angle, on surface of sphere
    surface_flux = intensity_per_freq * np.pi
    # Multiply to total area (4pi r^2) and divide by 4pi steradians,
    # e.g. multiply by r^2.
    surface_area_per_solid_angle = radius**2
    return surface_flux * surface_area_per_solid_angle
