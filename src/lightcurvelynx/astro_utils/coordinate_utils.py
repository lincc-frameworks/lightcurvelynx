import numpy as np
from scipy.spatial import KDTree


def ra_dec_to_cartesian(ra, dec):
    """
    Batch convert right ascension and declination to Cartesian coordinates.

    We use this custom function over Astropy's built-in conversion for performance reasons.
    Because we know the data is in degrees and we only need Cartesian coordinates, we can skip
    object creation and units. The results are roughly 50x faster.

    Parameters
    ----------
    ra: float or numpy.ndarray
        Right ascension in degrees.
    dec: float or numpy.ndarray
        Declination in degrees.

    Returns
    -------
    x: float or numpy.ndarray
        X coordinate.
    y: float or numpy.ndarray
        Y coordinate.
    z: float or numpy.ndarray
        Z coordinate.
    """
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    return x, y, z


def dedup_coords(ra, dec, threshold=1e-5):
    """
    Remove duplicate coordinates within a specified threshold.

    Parameters
    ----------
    ra: numpy.ndarray
        Array of right ascension values in degrees.
    dec: numpy.ndarray
        Array of declination values in degrees.
    threshold: float
        Minimum separation in degrees to consider two points as distinct.

    Returns
    -------
    unique_ra: numpy.ndarray
        Array of unique right ascension values.
    unique_dec: numpy.ndarray
        Array of unique declination values.
    unique_indices: numpy.ndarray
        Indices of the unique coordinates in the original arrays.
    """
    # Create a KD-tree for efficient nearest neighbor search.
    x, y, z = ra_dec_to_cartesian(ra, dec)
    cart_coords = np.array([x, y, z]).T
    kd_tree = KDTree(cart_coords)

    # Do a range search with the same points to find all neighbors.
    adjusted_radius = 2.0 * np.sin(0.5 * np.radians(threshold))
    close_points = kd_tree.query_ball_point(cart_coords, adjusted_radius)

    # Find unique coordinates. We keep the first occurrence of each unique point.
    # Note there will always be at least one match (the point itself).
    unique_indices = []
    for idx, matches in enumerate(close_points):
        if len(matches) == 1 or idx == np.min(matches):
            unique_indices.append(idx)
    unique_indices = np.array(unique_indices)

    unique_ra = ra[unique_indices]
    unique_dec = dec[unique_indices]
    return unique_ra, unique_dec, unique_indices
