"""Flux-magnitude conversion utilities."""

import numpy as np
import numpy.typing as npt

from lightcurvelynx.base_models import FunctionNode

# AB definition is zp=8.9 for 1 Jy
MAG_AB_ZP_NJY = 8.9 + 2.5 * 9


def mag2flux(mag: npt.ArrayLike) -> npt.ArrayLike:
    """Convert AB magnitude to bandflux in nJy

    Parameters
    ----------
    mag : ndarray of float
        The magnitude to convert to bandflux.

    Returns
    -------
    bandflux : ndarray of float
        The bandflux corresponding to the input magnitude.
    """
    return np.power(10.0, -0.4 * (mag - MAG_AB_ZP_NJY))


def flux2mag(flux_njy: npt.ArrayLike) -> npt.ArrayLike:
    """Convert bandflux in nJy to AB magnitude

    Parameters
    ----------
    flux_njy : ndarray of float
        The bandflux to convert to magnitude.

    Returns
    -------
    mag : ndarray of float
        The magnitude corresponding to the input bandflux.
    """
    return MAG_AB_ZP_NJY - 2.5 * np.log10(flux_njy)


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
