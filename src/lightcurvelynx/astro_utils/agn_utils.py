"""Utility functions for AGN models.

Adapted from https://github.com/RickKessler/SNANA/blob/master/src/gensed_AGN.py
with the authors' permission.
"""

import numpy as np
from citation_compass import cite_function


@cite_function
def eddington_ratio_dist_fun(edd_ratio, galaxy_type="Blue", rng=None, num_samples=1):
    """Sample from the Eddington Ratio Distribution Function for a given galaxy type.

    Based on notebook from: https://github.com/burke86/imbh_forecast/blob/master/var.ipynb
    and the paper: https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    with parameters selected from: https://iopscience.iop.org/article/10.3847/1538-4357/aa803b/pdf

    References
    ----------
    * Approach: Sartori et. al. 2019 - https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    * Parameters: Weigel et. al. 2017 - https://iopscience.iop.org/article/10.3847/1538-4357/aa803b/pdf

    Parameters
    ----------
    edd_ratio : float
        The Eddington ratio.
    galaxy_type : str, optional
        The type of galaxy, either 'Red' or 'Blue'.
        Default: 'Blue'
    rng : np.random.Generator, optional
        The random number generator.
        Default: None
    num_samples : int, optional
        The number of samples to draw. If 1 returns a float otherwise returns an array.
        Default: 1

    Returns
    -------
    result : float or np.array
        The Eddington ratio distribution.
    """
    if rng is None:  # pragma: no cover
        rng = np.random.default_rng()

    if galaxy_type.lower() == "red":
        xi = 10**-2.13
        lambda_br = 10 ** rng.normal(-2.81, 0.18, num_samples)
        delta1 = rng.normal(0.41 - 0.7, 0.02, num_samples)  # > -0.45 won't affect LF
        delta2 = rng.normal(1.22, 0.16, num_samples)
    elif galaxy_type.lower() == "blue":
        xi = 10**-1.65
        lambda_br = 10 ** rng.normal(-1.84, 0.335, num_samples)
        delta1 = rng.normal(0.471 - 0.7, 0.02, num_samples)  # > -0.45 won't affect LF
        delta2 = rng.normal(2.53, 0.53, num_samples)
    else:
        raise ValueError("galaxy_type must be either 'Red' or 'Blue'.")

    result = xi * ((edd_ratio / lambda_br) ** delta1 + (edd_ratio / lambda_br) ** delta2)

    if num_samples == 1:
        return result[0]
    return result
