"""Utility functions for handling redshift and cosmological calculations."""

from functools import partial

import astropy.cosmology.units as cu
from astropy import units as u

from lightcurvelynx.base_models import FunctionNode


def obs_to_rest_times_waves(observer_frame_times, observer_frame_wavelengths, redshift, t0):
    """Calculate the rest frame times and wavelengths needed to give user the observer frame times
    and wavelengths (given the redshift).

    Parameters
    ----------
    observer_frame_times : numpy.ndarray
        The times at which the observation is made.
    observer_frame_wavelengths : numpy.ndarray
        The wavelengths at which the observation is made (in Angstroms).
    redshift : float
        The redshift of the object.
    t0 : float
        The reference epoch for the given object.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The rest frame times and wavelengths needed to generate the rest frame flux densities,
        which will later be redshifted  back to observer frame flux densities at the observer frame
        times and wavelengths.
    """
    observed_times_rel_to_t0 = observer_frame_times - t0
    rest_frame_times_rel_to_t0 = observed_times_rel_to_t0 / (1 + redshift)
    rest_frame_times = rest_frame_times_rel_to_t0 + t0
    rest_frame_wavelengths = observer_frame_wavelengths / (1 + redshift)
    return (rest_frame_times, rest_frame_wavelengths)


def rest_to_obs_flux(flux_density, redshift):
    """Convert rest-frame flux to obs-frame flux.
    The (1+redshift) factor is applied to preserve bolometric flux.
    The rest-frame flux is defined as F_nu = L_nu / 4*pi*D_L**2,
    where D_L is the luminosity distance.

    Parameters
    ----------
    flux_density : numpy.ndarray
        A length T X N matrix of flux density values (in nJy).
    redshift : float
        The redshift of the object associated with given flux densities.

    Returns
    -------
    flux_density : numpy.ndarray
        The observer frame flux (in nJy).
    """
    # Note that the multiplication by (1+z) is due to the fact we are working in f_nu units,
    # instead of f_lambda units and may be unintuitive for users who are used to working in f_lambda
    # units. This factor can be derived by equaling the integrated flux in f_nu unit before and after
    # redshift is applied.
    return flux_density * (1 + redshift)


def redshift_to_distance(redshift, cosmology):
    """Compute a source's luminosity distance given its redshift and a
    specified cosmology using astropy's redshift_distance().

    Parameters
    ----------
    redshift : float
        The redshift value.
    cosmology : astropy.cosmology
        The cosmology specification.

    Returns
    -------
    distance : float
        The luminosity distance (in pc)
    """
    z = redshift * cu.redshift
    distance = z.to(u.pc, cu.redshift_distance(cosmology, kind="luminosity"))
    return distance.value


class RedshiftDistFunc(FunctionNode):
    """A wrapper class for the redshift_to_distance() function.

    Attributes
    ----------
    cosmology : astropy.cosmology
        The cosmology specification.

    Parameters
    ----------
    redshift : function or constant
        The function or constant providing the redshift value.
    cosmology : astropy.cosmology
        The cosmology specification.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, redshift, cosmology, **kwargs):
        # Create a partial function with the cosmology fixed. We do this
        # so that cosmology is not added as a node parameter.
        func = partial(redshift_to_distance, cosmology=cosmology)
        func.__name__ = "redshift_to_distance"
        func.__doc__ = redshift_to_distance.__doc__

        # Call the super class's constructor with the needed information.
        super().__init__(
            func=func,
            redshift=redshift,
            **kwargs,
        )
