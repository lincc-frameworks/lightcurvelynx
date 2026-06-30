"""Flux-magnitude conversion utilities."""

import numpy as np
import numpy.typing as npt

from lightcurvelynx.base_models import FunctionNode

# AB definition is zp=8.9 for 1 Jy
MAG_AB_ZP_NJY = 8.9 + 2.5 * 9

# Factor to convert flux error to magnitude error
MAGERR_FACTOR = 2.5 / np.log(10.0)


def mag2flux(mag: npt.ArrayLike, mag_err: npt.ArrayLike = None) -> npt.ArrayLike:
    """Convert AB magnitude to bandflux in nJy.

    Parameters
    ----------
    mag : ndarray of float
        The magnitude to convert to bandflux.
    mag_err : ndarray of float, optional
        The magnitude error to convert to bandflux error.

    Returns
    -------
    bandflux : ndarray of float
        The bandflux corresponding to the input magnitude.
    bandflux_err : ndarray of float, optional
        The bandflux error corresponding to the input magnitude error.
        Only returned if ``mag_err`` is provided (not None), making the
        return value a tuple ``(bandflux, bandflux_err)``.
    """
    bandflux = np.power(10.0, -0.4 * (mag - MAG_AB_ZP_NJY))

    if mag_err is None:
        return bandflux

    bandflux_err = bandflux * mag_err / MAGERR_FACTOR
    return bandflux, bandflux_err


def flux2mag(flux_njy: npt.ArrayLike, flux_err_njy: npt.ArrayLike = None) -> npt.ArrayLike:
    """Convert bandflux in nJy to AB magnitude

    Parameters
    ----------
    flux_njy : ndarray of float
        The bandflux to convert to magnitude.
    flux_err_njy : ndarray of float, optional
        The bandflux error to convert to magnitude error.

    Returns
    -------
    mag : ndarray of float
        The magnitude corresponding to the input bandflux.
    mag_err: ndarray of float
        The magnitude error corresponding to the input bandflux error.
        Only returned if flux_err_njy is provided(not None),
        making the return value a tuple (mag, mag_err).
    """
    mag = MAG_AB_ZP_NJY - 2.5 * np.log10(flux_njy)

    if flux_err_njy is None:
        return mag

    mag_err = MAGERR_FACTOR * flux_err_njy / flux_njy
    return mag, mag_err


class Mag2FluxNode(FunctionNode):
    """A wrapper class for the mag2flux() function.

    Parameters
    ----------
    mag : ndarray of float
        The magnitude to convert to bandflux.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, mag, **kwargs):
        # Call the super class's constructor with the needed information.
        super().__init__(func=mag2flux, mag=mag, **kwargs)


class Flux2MagNode(FunctionNode):
    """A wrapper class for the flux2mag() function.

    Parameters
    ----------
    flux_njy : float or array-like
        The flux in nJy to convert to magnitude.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, flux_njy, **kwargs):
        # Call the super class's constructor with the needed information.
        super().__init__(func=flux2mag, flux_njy=flux_njy, **kwargs)
