from __future__ import annotations  # "type1 | type2" syntax in Python <3.9

import numpy as np
import numpy.typing as npt

from lightcurvelynx.astro_utils.mag_flux import mag2flux


def sky_bg_adu_to_electrons(sky_bg_adu, gain):
    """Convert sky background from ADU/pixel to electrons/pixel.

    Parameters
    ----------
    sky_bg_adu : float or ndarray of float
        Sky background in ADU/pixel.
    gain : float or ndarray of float
        The CCD gain (in e-/ADU).

    Returns
    -------
    float or ndarray of float
        Sky background in electrons/pixel.
    """
    return sky_bg_adu * gain


def magnitude_electron_zeropoint(
    *,
    filter: npt.ArrayLike,
    airmass: npt.ArrayLike,
    exptime: npt.ArrayLike,
    instr_zp_mag: dict[str, float] | float | npt.ArrayLike,
    ext_coeff: dict[str, float] | float | npt.ArrayLike,
) -> npt.ArrayLike:
    """Photometric zeropoint (magnitude that produces 1 electron) for
    LSST bandpasses (v1.9), using a standard atmosphere scaled
    for different airmasses and scaled for exposure times.

    Parameters
    ----------
    filter : ndarray of str
        The filter for which to return the photometric zeropoint.
    airmass : ndarray of float
        The airmass at which to return the photometric zeropoint.
    exptime : ndarray of float
        The exposure time for which to return the photometric zeropoint.
    instr_zp_mag : dict[str, float], float, or ndarray of float
        The instrumental zeropoint for each bandpass,
        i.e. AB-magnitude that produces 1 electron in a 1-second exposure.
        Keys are the bandpass names, values are the zeropoints.
    ext_coeff : dict[str, float], float, or ndarray of float
        Atmospheric extinction coefficient for each bandpass.
        Keys are the bandpass names, values are the coefficients.

    Returns
    -------
    ndarray of float
        AB mags that produces 1 electron.

    Notes
    -----
    Typically, zeropoints are defined as the magnitude of a source
    which would produce 1 count in a 1 second exposure -
    here we use *electron* counts, not ADU counts.

    References
    ----------
    Lynne Jones - https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
    """
    # If we are given dictionaries mapping filter to value for either instr_zp_mag or ext_coeff,
    # convert them to the corresponding array of values for the input filter array.
    if isinstance(instr_zp_mag, dict):
        instr_zp_mag = np.vectorize(instr_zp_mag.get)(filter)
    if isinstance(ext_coeff, dict):
        ext_coeff = np.vectorize(ext_coeff.get)(filter)
    return instr_zp_mag + ext_coeff * (airmass - 1) + 2.5 * np.log10(exptime)


def flux_electron_zeropoint(
    *,
    instr_zp_mag: dict[str, float] | float | npt.ArrayLike,
    ext_coeff: dict[str, float] | float | npt.ArrayLike,
    filter: npt.ArrayLike,
    airmass: npt.ArrayLike,
    exptime: npt.ArrayLike,
) -> npt.ArrayLike:
    """Flux (nJy) producing 1 electron.

    Parameters
    ----------
    filter : npt.ArrayLike
        The filter for which to return the photometric zeropoint.
    airmass : npt.ArrayLike
        The airmass at which to return the photometric zeropoint.
    exptime : npt.ArrayLike
        The exposure time for which to return the photometric zeropoint.
    instr_zp_mag : dict[str, float], float, or ndarray of float
        The instrumental zeropoint for each bandpass in AB magnitudes,
        i.e. the magnitude that produces 1 electron in a 1-second exposure.
        Keys are the bandpass names, values are the zeropoints.
    ext_coeff : dict[str, float], float, or ndarray of float
        Atmospheric extinction coefficient for each bandpass.
        Keys are the bandpass names, values are the coefficients.

    Returns
    -------
    ndarray of float
        Flux (nJy) per electron.
    """
    mag_zp_electron = magnitude_electron_zeropoint(
        instr_zp_mag=instr_zp_mag,
        ext_coeff=ext_coeff,
        filter=filter,
        airmass=airmass,
        exptime=exptime,
    )
    return mag2flux(mag_zp_electron)


def calculate_zp_from_maglim(
    maglim=None,
    sky_bg_electrons=None,
    fwhm_px=None,
    read_noise=None,
    dark_current=None,
    exptime=None,
    nexposure=1,
):
    """Calculate zero points based on the 5-sigma mag limit.

    snr = flux/fluxerr
    fluxerr = sqrt(flux + sky*npix*gain + readnoise**2*nexposure*npix + darkcurrent*npix*exptime*nexposure)
    5 = flux/fluxerr
    25 = flux**2/(flux + sky*npix*Gain + readnoise**2*nexposure*npix + darkcurrent*npix*exptime*nexposure)
    flux**2 - 25*flux -25*( sky*npix*Gain
                            + readnoise**2*nexposure*npix
                            + darkcurrent*npix*exptime*nexposure)
                        = 0
    flux = 12.5 + 0.5*sqrt(625
                            + 100( sky*npix*Gain
                            + readnoise**2*nexposure*npix
                            + darkcurrent*npix*exptime*nexposure) )
    zp = 2.5*log10(flux) + maglim

    Parameters
    ----------
    maglim : float or ndarray
        Five-sigma magnitude limit.
    sky_bg_electrons : float or ndarray
        Sky background in electrons/pixel.
    fwhm_px : float or ndarray
        PSF in pixels.
    read_noise : float or ndarray
        Read noise (in e-/pixel).
    dark_current : float or ndarray
        Dark current (in e-/pixel/second).
    exptime : float or ndarray
        Exposure time (in seconds).
    nexposure : int or ndarray
        Number of exposure. Default is 1.

    Returns
    -------
    zp: float or ndarray
        Instrument zero point (that converts 1 e- to magnitude).
    """
    npix = 2.266 * fwhm_px**2  # = 4 * pi * sigma**2 = pi/2/ln2 * FWHM**2
    flux_at_5sigma_limit = 12.5 + 2.5 * np.sqrt(
        25.0
        + 4.0
        * (
            sky_bg_electrons * npix
            + read_noise**2 * nexposure * npix
            + dark_current * npix * exptime * nexposure
        )
    )
    zp = 2.5 * np.log10(flux_at_5sigma_limit) + maglim

    return zp
