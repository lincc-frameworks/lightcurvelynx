import astropy.units as u
import numpy as np
import pytest
from lightcurvelynx.astro_utils.coordinate_utils import (
    build_moc_from_coords,
    dedup_coords,
    ra_dec_to_cartesian,
)


def test_ra_dec_to_cartesian():
    """Test the conversion from RA/Dec to Cartesian coordinates."""
    # Test known values
    ra = np.array([0, 90, 180, 270, 90, 0, 45.0, 405.0])
    dec = np.array([0, 0, 0, 0, 90, -90, 30.0, 30.0])
    x, y, z = ra_dec_to_cartesian(ra, dec)

    expected_x = np.array([1, 0, -1, 0, 0, 0, np.sqrt(6) / 4.0, np.sqrt(6) / 4.0])
    expected_y = np.array([0, 1, 0, -1, 0, 0, np.sqrt(6) / 4.0, np.sqrt(6) / 4.0])
    expected_z = np.array([0, 0, 0, 0, 1, -1, 1.0 / 2.0, 1.0 / 2.0])

    np.testing.assert_allclose(x, expected_x, atol=1e-5)
    np.testing.assert_allclose(y, expected_y, atol=1e-5)
    np.testing.assert_allclose(z, expected_z, atol=1e-5)

    with pytest.raises(ValueError):
        ra_dec_to_cartesian([0], [100])  # Invalid declination


def test_dedup_coords():
    """Test the deduplication of coordinates."""
    pts = np.array(
        [
            [0, 0],  # First unique point
            [0.01, 0.01],  # Duplicate of first point
            [-0.01, -0.01],  # Duplicate of first point
            [1, 1],  # Second unique point
            [100, -10],  # Third unique point
            [99.9999, -10.0001],  # Duplicate of third point
            [45, 90],  # Fourth unique point
            [-0.01, -0.01],  # Duplicate of first point
            [1.0001, 1.0001],  # Duplicate of second point
            [100.0001, -9.9999],  # Duplicate of third point
            [270.0, -90.0],  # Fifth unique point
        ]
    )
    unique_ra, unique_dec, unique_indices = dedup_coords(pts[:, 0], pts[:, 1], threshold=0.02)
    expected_inds = np.array([0, 3, 4, 6, 10])
    np.testing.assert_array_equal(unique_indices, expected_inds)
    np.testing.assert_allclose(unique_ra, pts[expected_inds, 0], atol=1e-8)
    np.testing.assert_allclose(unique_dec, pts[expected_inds, 1], atol=1e-8)

    # We fail if the lists are different lengths.
    with pytest.raises(ValueError):
        dedup_coords([0, 1], [0])  # Mismatched lengths


def test_build_moc_from_coords():
    """Test building a MOC from coordinates."""
    ra = np.array([0, 90, 180, 270, 45])
    dec = np.array([0, 0, 0, 0, 30])
    moc = build_moc_from_coords(ra, dec, depth=8)
    assert len(moc.flatten()) == 5

    # Test that the MOC covers the input points
    for r, d in zip(ra, dec, strict=False):
        assert moc.contains_lonlat(r * u.deg, d * u.deg)

    # Duplicates are naturally dropped when we dedup the healpix ids.
    ra_dup = np.array([0, 90, 180, 270, 45, 0, 0.0001, 45.0001])
    dec_dup = np.array([0, 0, 0, 0, 30, 0, 0.0001, 29.9999])
    moc_dup = build_moc_from_coords(ra_dup, dec_dup, depth=8)
    assert len(moc_dup.flatten()) == 5

    # We fail if the lists are different lengths.
    with pytest.raises(ValueError):
        _ = build_moc_from_coords([0, 90], [0])
