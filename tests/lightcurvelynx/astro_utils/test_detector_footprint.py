import matplotlib

matplotlib.use("Agg")  # Suppress the plots during testing

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord, RectanglePixelRegion
from regions.core.compound import CompoundPixelRegion


def test_rotate_to_center():
    """Test the static rotate_to_center method."""
    # Define the test data as a numpy array with one row for each test case,
    # and columns for ra, dec, center_ra, center_dec, expected_ra, expected_dec.
    tests = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # No shift
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Small shift in RA only
            [57.0, 0.0, 0.0, 0.0, 57.0, 0.0],  # Large shift in RA only
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Shift in Dec
            [91.0, -10.0, 91.0, -10.0, 0.0, 0.0],  # Shift to same point
            [45.0, -89.0, 45.0, -90.0, 0.0, 1.0],  # Same RA, different DEC
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],  # Shift in both
            [45.0, 45.0, 40.0, 40.0, 3.54734593, 5.09948400],  # Large shift in both
        ]
    )

    # Test the internal transform method. The points should be relative to the
    # center and in radians.
    lon_t, lat_t = DetectorFootprint.rotate_to_center(tests[:, 0], tests[:, 1], tests[:, 2], tests[:, 3])
    assert np.allclose(lon_t, tests[:, 4], atol=1e-6)
    assert np.allclose(lat_t, tests[:, 5], atol=1e-6)

    # Perform some additional tests with rotation. The test data is now
    # ra, dec, center_ra, center_dec, rotation, expected_ra, expected_dec
    tests = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 90.0, 0.0, 1.0],  # Rotate RA=1 to dec=1
            [0.0, 1.0, 0.0, 0.0, 90.0, -1.0, 0.0],  # Rotate dec=1 to RA=-1
            [1.0, 0.0, 0.0, 0.0, 45.0, 0.7071, 0.7071],  # Rotate RA=1 to RA=0.7,dec=0.7
            [1.0, 1.0, 0.0, 0.0, 90.0, -1.0, 1.0],  # Rotate RA=1,dec=1 to RA=-1,dec=1
            [1.0, 1.0, 0.0, 0.0, -45.0, 1.4142, 0.0],  # Rotate RA=1,dec=1 to RA=1.4,dec=0
            [45.0, 45.0, 40.0, 40.0, 45.0, -1.111, 6.1095],  # Large shift in both with rotation
        ]
    )

    # Test the internal transform method.
    lon_t, lat_t = DetectorFootprint.rotate_to_center(
        tests[:, 0],
        tests[:, 1],
        tests[:, 2],
        tests[:, 3],
        rotation=tests[:, 4],
    )
    assert np.allclose(lon_t, tests[:, 5], atol=1e-3)
    assert np.allclose(lat_t, tests[:, 6], atol=1e-3)


def test_create_detector_footprint():
    """Test creating a DetectorFootprint."""
    center = SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs")
    circle_region = CircleSkyRegion(center=center, radius=1.0 * u.deg)
    fp = DetectorFootprint(circle_region, pixel_scale=1.0)  # 1 arcsec/pixel
    assert isinstance(fp, DetectorFootprint)
    assert isinstance(fp.region, CirclePixelRegion)
    assert fp.wcs is not None

    # Test that we can compute a bounding radius. Since this is based
    # on the bounding box, it will be larger than the actual radius.
    radius = fp.compute_radius()
    assert radius > 1.0
    assert radius < np.sqrt(2) + 0.001  # Small tolerance for pixel padding

    # Test that the contains method works.
    ra = np.array([0.0, 0.5, 1.0, 1.5, 0.0, 1.5, 0.9])  # degrees
    dec = np.array([0.0, 0.5, 1.0, 1.5, 1.5, 0.0, 0.0])  # degrees
    result = fp.contains(ra, dec, center_ra=0.0, center_dec=0.0)
    expected = np.array([True, True, False, False, False, False, True])
    assert np.array_equal(result, expected)

    # We can pass a vector of center positions as well.  Everything matches up.
    result = fp.contains(ra, dec, center_ra=ra, center_dec=dec)
    assert np.all(result)

    # We can try a circular region that is offset from the center.
    result = fp.contains(ra + 45.0, dec - 10.0, center_ra=45.0, center_dec=-10.0)
    expected = np.array([True, True, False, False, False, False, True])
    assert np.array_equal(result, expected)

    # We have an error if the array shapes are not compatible.
    with pytest.raises(ValueError):
        fp.contains(ra, dec, center_ra=np.array([0.0, 1.0]), center_dec=0.0)
    with pytest.raises(ValueError):
        fp.contains(ra, dec, center_ra=0.0, center_dec=np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        fp.contains(ra, dec, center_ra=0.0, center_dec=0.0, rotation=np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        fp.contains(ra, np.array([0.0, 1.0]), center_ra=0.0, center_dec=0.0)

    # We fail to create a footprint if we do not pass in either wcs or pixel scale.
    # We also fail if we pass a negative pixel scale.
    with pytest.raises(ValueError):
        _ = DetectorFootprint(circle_region)
    with pytest.raises(ValueError):
        _ = DetectorFootprint(circle_region, pixel_scale=-1.0)


def test_rectangular_sky_footprint():
    """Test the DetectorFootprint's from_sky_rect function."""
    width = 2.0  # degrees = 200 pixels at 0.01 deg/pix
    height = 1.0  # degrees = 100 pixels at 0.01 deg/pix
    fp = DetectorFootprint.from_sky_rect(width=width, height=height, pixel_scale=36.0)  # 0.01 deg/pix

    # Test that we can compute an approximate radius.
    radius = fp.compute_radius()
    approx_radius = np.sqrt((height / 2.0) ** 2 + (width / 2.0) ** 2)
    assert radius >= approx_radius
    assert np.isclose(radius, approx_radius, rtol=0.1)

    # Generate data for the contains tests.
    ra = np.array([90.0, 91.0, 90.5, 91.5, 92.0, 90.5, 90.0])
    center_ra = np.array([90.0, 90.0, 90.0, 90.0, 92.0, 93.0, 90.0])
    dec = np.array([-10.0, -13.0, -10.0, -10.0, -8.0, -10.25, -9.25])
    center_dec = np.array([-10.0, -10.0, -10.0, -10.0, -8.0, -10.0, -10.0])

    # Test contains without rotation.
    contained = fp.contains(ra, dec, center_ra, center_dec)
    assert np.all(contained == np.array([True, False, True, False, True, False, False]))

    # Test that we can generate a SkyRegion, WCS, and MOC at each center and they
    # give the same results for contains.
    for idx, (c_ra, c_dec) in enumerate(zip(center_ra, center_dec, strict=False)):
        sky_region, sky_wcs = fp.compute_sky_region(c_ra, c_dec)
        assert sky_region.contains(SkyCoord(ra=ra[idx], dec=dec[idx], unit="deg"), sky_wcs) == contained[idx]

    # Test contains with rotation. The last point should now be contained.
    rotation = np.full(ra.shape, 90.0)
    contained = fp.contains(ra, dec, center_ra, center_dec, rotation=rotation)
    assert np.all(contained == np.array([True, False, True, False, True, False, True]))

    # Test that we can generate a SkyRegion, WCS, and MOC at each center and they
    # give the same results for contains.
    for idx, (c_ra, c_dec) in enumerate(zip(center_ra, center_dec, strict=False)):
        print(f"Testing index {idx} with center RA={c_ra}, DEC={c_dec}")
        sky_region, sky_wcs = fp.compute_sky_region(c_ra, c_dec, rotation=90.0)
        assert sky_region.contains(SkyCoord(ra=ra[idx], dec=dec[idx], unit="deg"), sky_wcs) == contained[idx]

    # We can query scalars as well.
    assert fp.contains(90.5, -10.25, 90.0, -10.0)
    assert not fp.contains(91.5, -10.25, 90.0, -10.0)
    assert fp.contains(89.75, -9.75, 90.0, -10.0)
    assert not fp.contains(89.75, -9.25, 90.0, -10.0)

    assert not fp.contains(91.5, -10.25, 90.0, -10.0, rotation=45.0)
    assert fp.contains(89.75, -9.75, 90.0, -10.0, rotation=-45.0)
    assert not fp.contains(89.75, -9.25, 90.0, -10.0, rotation=-45.0)

    # Test a 45 degree rotation.
    assert not fp.contains(0.7, -0.7, 0, 0.0, rotation=0.0)
    assert fp.contains(0.7, -0.7, 0, 0.0, rotation=45.0)

    # Try some points around the rectangles border when the rectangle is
    # is centered at different locations to ensure we scale RA correctly.
    for c_ra, c_dec in [(45.0, 20), (60.0, -30.0), (20.0, 60.0), (-30.0, -75.0)]:
        center = SkyCoord(ra=c_ra, dec=c_dec, unit="deg", frame="icrs")
        for dec_offset in [-0.51, -0.49, 0.49, 0.51]:
            for ra_offset in [-1.01, -0.99, 0.99, 1.01]:
                # Shift by DEC then RA.
                query_pt = center.directional_offset_by(0.0 * u.deg, dec_offset * u.deg)
                query_pt = query_pt.directional_offset_by(90 * u.deg, ra_offset * u.deg)
                expected = abs(ra_offset) < 1.0 and abs(dec_offset) < 0.5
                assert fp.contains(query_pt.ra.deg, query_pt.dec.deg, c_ra, c_dec) == expected

    # We fail to create a footprint if it is not centered on (0,0).
    center = PixCoord(x=100.0, y=0.0)
    offset_region = RectanglePixelRegion(center=center, width=2.0, height=10.0, angle=0.0 * u.deg)
    with pytest.raises(ValueError):
        DetectorFootprint(offset_region, pixel_scale=36.0)  # 0.01 deg/pix


def test_rectangular_pixel_footprint():
    """Test the DetectorFootprint's from_pixel_rect function."""
    width = 200.0  # pixels = 2 degrees at 0.01 deg/pix
    height = 100.0  # pixels = 1 degree at 0.01 deg/pix
    fp = DetectorFootprint.from_pixel_rect(width=width, height=height, pixel_scale=36.0)  # 0.01 deg/pix
    ra = np.array([90.0, 91.0, 90.5, 91.5, 92.0, 90.5, 90.0])
    center_ra = np.array([90.0, 90.0, 90.0, 90.0, 92.0, 93.0, 90.0])
    dec = np.array([-10.0, -13.0, -10.0, -10.0, -8.0, -10.25, -9.25])
    center_dec = np.array([-10.0, -10.0, -10.0, -10.0, -8.0, -10.0, -10.0])

    # Test that we can compute an approximate radius.
    radius = fp.compute_radius()
    approx_radius = np.sqrt(1.0**2 + 0.5**2)
    assert radius >= approx_radius
    assert np.isclose(radius, approx_radius, rtol=0.1)

    # Test contains without rotation.
    contained = fp.contains(ra, dec, center_ra, center_dec)
    assert np.all(contained == np.array([True, False, True, False, True, False, False]))

    # Test contains with rotation. The last point should now be contained.
    rotation = np.full(ra.shape, 90.0)
    contained = fp.contains(ra, dec, center_ra, center_dec, rotation=rotation)
    assert np.all(contained == np.array([True, False, True, False, True, False, True]))

    # We can query scalars as well.
    assert fp.contains(90.5, -10.25, 90.0, -10.0)
    assert not fp.contains(91.5, -10.25, 90.0, -10.0)
    assert fp.contains(89.75, -9.75, 90.0, -10.0)
    assert not fp.contains(89.75, -9.25, 90.0, -10.0)

    assert not fp.contains(91.5, -10.25, 90.0, -10.0, rotation=45.0)
    assert fp.contains(89.75, -9.75, 90.0, -10.0, rotation=-45.0)
    assert not fp.contains(89.75, -9.25, 90.0, -10.0, rotation=-45.0)

    # Test a 45 degree rotation.
    assert not fp.contains(0.7, -0.7, 0, 0.0, rotation=0.0)
    assert fp.contains(0.7, -0.7, 0, 0.0, rotation=45.0)

    # We fail if a NaN is passed in for rotation.
    with pytest.raises(ValueError):
        fp.contains(90.5, -10.25, 90.0, -10.0, rotation=np.nan)

    # Try some points around the rectangles border when the rectangle is
    # is centered at different locations to ensure we scale RA correctly.
    for c_ra, c_dec in [(45.0, 20), (60.0, -30.0), (20.0, 60.0), (-30.0, -75.0)]:
        center = SkyCoord(ra=c_ra, dec=c_dec, unit="deg", frame="icrs")
        for dec_offset in [-0.51, -0.49, 0.49, 0.51]:
            for ra_offset in [-1.01, -0.99, 0.99, 1.01]:
                # Shift by DEC then RA.
                query_pt = center.directional_offset_by(0.0 * u.deg, dec_offset * u.deg)
                query_pt = query_pt.directional_offset_by(90 * u.deg, ra_offset * u.deg)
                expected = abs(ra_offset) < 1.0 and abs(dec_offset) < 0.5
                assert fp.contains(query_pt.ra.deg, query_pt.dec.deg, c_ra, c_dec) == expected

    # We fail to create a footprint if it is not centered on (0,0).
    center = PixCoord(x=100.0, y=0.0)
    offset_region = RectanglePixelRegion(center=center, width=2.0, height=10.0, angle=0.0 * u.deg)
    with pytest.raises(ValueError):
        DetectorFootprint(offset_region, pixel_scale=36.0)  # 0.01 deg/pix


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_detector_footprint_plot():
    """Test that plot does not throw an error."""
    width = 200.0  # pixels = 2 degrees at 0.01 deg/pix
    height = 100.0  # pixels = 1 degree at 0.01 deg/pix
    fp = DetectorFootprint.from_pixel_rect(width=width, height=height, pixel_scale=36.0)  # 0.01 deg/pix

    # Test that we can plot the footprint.
    fp.plot(point_ra=[0.0], point_dec=[0.0], center_ra=0.0, center_dec=0.0)


@pytest.mark.filterwarnings("ignore::UserWarning")  # Ignore plotting warning
def test_detector_footprint_plot_nested_compound():
    """Test that nested compound regions can be plotted."""
    first = RectanglePixelRegion(center=PixCoord(x=-2.0, y=0.0), width=2.0, height=2.0)
    second = RectanglePixelRegion(center=PixCoord(x=2.0, y=0.0), width=2.0, height=2.0)
    third = RectanglePixelRegion(center=PixCoord(x=0.0, y=3.0), width=2.0, height=2.0)
    compound_region = (first.union(second)).union(third)
    assert isinstance(compound_region, CompoundPixelRegion)

    fp = DetectorFootprint(compound_region, pixel_scale=36.0)
    fp.plot()


def test_detector_footprint_from_sorcha_corners(test_data_dir):
    """Test that we can load a detector footprint from a Sorcha corners file."""
    filename = test_data_dir / "test_corners.csv"
    footprint = DetectorFootprint.from_sorcha_corners_file(filename, pixel_scale=0.2)
    assert footprint is not None

    # The test file is a 3 x 1 array of detectors along the ra dimension.
    # Each detector is a square of ~800 arcseconds by ~800 arcseconds.
    # We test it centered on (0.0, 0.0) for ease of computations.

    # 1) Test a bunch of points that should all be inside.
    assert np.all(
        footprint.contains(
            np.array([0.0, 0.333, -0.333, 0.01, 0.300, 0.1, -0.1, 0.2, -0.2]),
            np.array([0.0, 0.01, -0.01, 0.01, 0.01, 0.0, 0.01, -0.01, 0.01]),
            0.0,
            0.0,
        )
    )

    # 2) Test the row above (all out).
    assert not np.any(
        footprint.contains(
            np.array([0.0, 0.333, -0.333, 0.01, 0.300]),
            np.array([0.333, 0.330, 0.330, 0.329, 0.333]),
            0.0,
            0.0,
        )
    )

    # 3) Test the row below (all out).
    assert not np.any(
        footprint.contains(
            np.array([0.0, 0.333, -0.333, 0.01, 0.300]),
            np.array([-0.333, -0.330, -0.330, -0.329, -0.333]),
            0.0,
            0.0,
        )
    )

    # Test outside the RA bounds (all out).
    assert not np.any(
        footprint.contains(
            np.array([1.0, 1.01, 0.99, 1.01, 1.01]),
            np.array([0.0, 0.01, -0.01, 0.01, 0.01]),
            0.0,
            0.0,
        )
    )


def test_detector_footprint_from_preset_lsst_approx():
    """Test that we can create a detector footprint from the 'lsst-approx' preset."""
    footprint = DetectorFootprint.from_preset("lsst-approx")
    assert footprint is not None
    assert np.array_equal(
        footprint.contains(
            np.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5]),
            np.array([1.5, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, -1.5, -1.5, -1.5, -1.5]),
            0.0,
            0.0,
        ),
        np.array([False, True, True, False, True, True, True, True, False, True, True, False]),
    )


def test_detector_footprint_from_preset_lsst_ccd():
    """Test that we can create a detector footprint from the 'lsst-ccd' preset."""
    footprint = DetectorFootprint.from_preset("lsst-ccd")
    assert footprint is not None
    # Test a few points that should be inside the single CCD 0.222 x 0.222 deg.
    assert np.all(
        footprint.contains(
            np.array([-0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1]),
            np.array([-0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1]),
            0.0,
            0.0,
        )
    )
    # Test a few points that should be outside the single CCD.
    assert not np.any(
        footprint.contains(
            np.array([-0.2, 0.2, 0.0, 0.0]),
            np.array([0.0, 0.0, -0.2, 0.2]),
            0.0,
            0.0,
        )
    )


def test_detector_footprint_from_preset_unknown():
    """Test that we can raise and error if given an unknown survey name."""
    with pytest.raises(ValueError):
        DetectorFootprint.from_preset("unknown-survey")
