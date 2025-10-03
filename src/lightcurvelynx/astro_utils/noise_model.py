from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def poisson_bandflux_std(
    bandflux: npt.ArrayLike,
    *,
    total_exposure_time: npt.ArrayLike,
    exposure_count: npt.ArrayLike,
    psf_footprint: npt.ArrayLike,
    sky: npt.ArrayLike,
    zp: npt.ArrayLike,
    readout_noise: npt.ArrayLike | Callable,
    dark_current: npt.ArrayLike,
    zp_err_mag: npt.ArrayLike = 0.0,
) -> npt.ArrayLike:
    """Simulate photon noise for bandflux measurements.

    Parameters
    ----------
    bandflux : array_like of float
        Source bandflux in energy units, e.g. nJy.
    total_exposure_time : array_like of float
        Total exposure time of all observation, in time units
        (e.g. seconds).
    exposure_count : array_like of int
        Number of exposures in the observation.
    sky : array_like of float
        Sky background per unit angular area,
        in the units of electrons / pixel^2.
    psf_footprint : array_like of float
        Point spread function effective area, in pixel^2.
    zp : array_like of float
        Zero point bandflux for the observation, i.e. bandflux
        giving a single electron during the total exposure time.
        Units are the same as the input bandflux over electron,
        e.g. nJy / electron.
    readout_noise : array_like of float, or Callable
        Standard deviation of the readout electrons per pixel per exposure.
    dark_current : array_like of float
        Mean dark current electrons per pixel per unit time.
    zp_err_mag : array_like of float
        Zero-point uncertainty in magnitude. Default is 0.

    Returns
    -------
    array_like
        Simulated bandflux noise, in the same units as the input bandflux.

    Notes
    -----

    1. We do not specify units for the input parameters, but they
    should be consistent with each other.

    2. Here we assume that the sky and source photon noises follow
    Poisson statistics in the limit of large number of photons,
    e.g. they are both considered to be normal distributed with
    variance equal to the number of photons. Readout noise is
    assumed to be Poisson distributed with variance (squared mean)
    equal to the square of the given value. Dark current is assumed
    to be Poisson distributed with variance (squared mean) equal
    to the product of the given value and the exposure time.
    The output is Poisson standard deviation of the sum of all
    these noises converted to the flux units.
    """
    # Get variances, in electrons^2

    source_variance = bandflux / zp
    sky_variance = sky * psf_footprint
    if callable(readout_noise):
        readout_variance = readout_noise(total_exposure_time) ** 2
    else:
        readout_variance = readout_noise**2 * psf_footprint * exposure_count
    dark_variance = dark_current * total_exposure_time * psf_footprint

    zp_variance = (bandflux * zp_err_mag * np.log(10.0) / 2.5 / zp) ** 2

    total_variance = source_variance + sky_variance + readout_variance + dark_variance + zp_variance

    return np.sqrt(total_variance) * zp


def apply_noise(bandflux, bandflux_err, rng=None):
    """Apply Gaussian noise to a bandflux measurement.

    Parameters
    ----------
    bandflux : ndarray of float
        The bandflux measurement.
    bandflux_err : ndarray of float
        The bandflux measurement error.
    rng : np.random.Generator, optional
        The random number generator.

    Returns
    -------
    ndarray of float
        The noisy bandflux measurement.
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(loc=bandflux, scale=bandflux_err)
