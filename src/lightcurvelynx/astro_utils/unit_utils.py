from functools import lru_cache

import numpy as np
from astropy import constants as const


# Cache the conversion factors to avoid repeated calculations. We expect this
# to be more efficient because the models use the same units for each evaluation.
@lru_cache(maxsize=32)
def get_flam_to_fnu_multiplier(flam_unit, wave_unit, fnu_unit):
    """Get the multiplier for flam to fnu conversion.

    Parameters
    ----------
    flam_unit : astropy.units.Unit
        The unit for the input flux_flam values.
    wave_unit : astropy.units.Unit
        The unit for the wavelength values.
    fnu_unit : astropy.units.Unit
        The unit for the output flux_fnu values.

    Returns
    -------
    multiplier : float
        The multiplier for the conversion from flam to fnu.
    """
    input_units = (flam_unit * wave_unit * wave_unit) / const.c.unit
    multiplier = (1.0 * input_units).to_value(fnu_unit)
    return multiplier


@lru_cache(maxsize=32)
def get_fnu_to_flam_multiplier(fnu_unit, wave_unit, flam_unit):
    """Get the multiplier for fnu to flam conversion.

    Parameters
    ----------
    fnu_unit : astropy.units.Unit
        The unit for the input flux_fnu values.
    wave_unit : astropy.units.Unit
        The unit for the wavelength values.
    flam_unit : astropy.units.Unit
        The unit for the output flux_flam values.

    Returns
    -------
    multiplier : float
        The multiplier for the conversion from fnu to flam.
    """
    input_units = (fnu_unit * const.c.unit) / (wave_unit * wave_unit)
    multiplier = (1.0 * input_units).to_value(flam_unit)
    return multiplier


def flam_to_fnu(flux_flam, wavelengths, *, wave_unit, flam_unit, fnu_unit):
    """Covert flux from f_lambda unit to f_nu unit.

    Parameters
    ----------
    flux_flam : list or numpy.ndarray
        The flux values in flam units. This can be a single N-length array
        or an M x N matrix.
    wavelengths: list or numpy.ndarray
        The wavelength values associated with the input flux values.
        This can be a single N-length array or an M x N matrix. If it is an
        N-length array, the same wavelength values are used for each flux_flam.
    wave_unit: astropy.units.Unit
        The unit for the wavelength values.
    flam_unit: astropy.units.Unit
        The unit for the input flux_flam values.
    fnu_unit: astropy.units.Unit
        The unit for the output flux_fnu values.

    Returns
    -------
    flux_fnu : list or np.array
        The flux values in fnu units.
    """
    flux_flam = np.asarray(flux_flam)
    wavelengths = np.asarray(wavelengths)

    # Check if we need to reshape wavelengths to match the number
    # of rows in flux_flam.
    if flux_flam.ndim > 1 and wavelengths.ndim == 1:
        wavelengths = wavelengths[None, :]

    # Check that the shapes match.
    try:
        _ = np.broadcast_shapes(flux_flam.shape, wavelengths.shape)
    except ValueError as err:
        raise ValueError(
            f"Mismatched sizes for flux_flam={flux_flam.shape} and wavelengths={wavelengths.shape}."
        ) from err

    # convert flux in flam_unit (e.g. ergs/s/cm^2/A) to fnu_unit (e.g. nJy or ergs/s/cm^2/Hz)
    multiplier = get_flam_to_fnu_multiplier(flam_unit, wave_unit, fnu_unit)
    flux_fnu = flux_flam * (wavelengths**2) / const.c.value
    return multiplier * flux_fnu


def fnu_to_flam(flux_fnu, wavelengths, *, wave_unit, flam_unit, fnu_unit):
    """
    Covert flux from f_nu unit to f_lambda unit

    Parameters
    ----------
    flux_fnu : list or numpy.ndarray
        The flux values in fnu units. This can be a single N-length array
        or an M x N matrix.
    wavelengths: list or numpy.ndarray
        The wavelength values associated with the input flux values.
        This can be a single N-length array or an M x N matrix. If it is an
        N-length array, the same wavelength values are used for each flux_fnu.
    wave_unit: astropy.units.Unit
        The unit for the wavelength values.
    flam_unit: astropy.units.Unit
        The unit for the output flux_flam values.
    fnu_unit: astropy.units.Unit
        The unit for the input flux_fnu values.

    Returns
    -------
    flux_flam : list or np.array
        The flux values in flam units.
    """
    flux_fnu = np.asarray(flux_fnu)
    wavelengths = np.asarray(wavelengths)

    # Check if we need to reshape wavelengths to match the number
    # of rows in flux_fnu.
    if flux_fnu.ndim > 1 and wavelengths.ndim == 1:
        wavelengths = wavelengths[None, :]

    # Check that the shapes match.
    try:
        _ = np.broadcast_shapes(flux_fnu.shape, wavelengths.shape)
    except ValueError as err:
        raise ValueError(
            f"Mismatched sizes for flux_fnu={flux_fnu.shape} and wavelengths={wavelengths.shape}."
        ) from err

    # convert flux in fnu_unit (e.g. nJy or ergs/s/cm^2/Hz) to flam_unit (e.g. ergs/s/cm^2/A)
    multiplier = get_fnu_to_flam_multiplier(fnu_unit, wave_unit, flam_unit)
    flux_flam = flux_fnu * const.c.value / wavelengths**2
    return multiplier * flux_flam
