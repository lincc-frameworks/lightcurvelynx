import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from lightcurvelynx.models.periodic_variable_star import EclipsingBinaryStar
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation


def test_circle_overlap_area_total_eclipse():
    """Test that for r1=r2 and d=0, the overlap area is equal to the area of the circles"""
    r = 3.5
    d = 0.0
    overlap_area = EclipsingBinaryStar._circle_overlap_area(d, r, r)
    assert_allclose(overlap_area, np.pi * r**2)


def test_circle_overlap_area_primary_minimum():
    """Test that for r1>r2 and d < r1-r2, the overlap area is equal to the area of the secondary circle"""
    r1 = 3.5
    r2 = 1.5
    d = 1.0
    overlap_area = EclipsingBinaryStar._circle_overlap_area(d, r1, r2)
    assert_allclose(overlap_area, np.pi * r2**2)


def test_circle_overlap_area_no_overlap():
    """Test that for d>=r1+r2, the overlap area is zero"""
    r1 = 3.5
    r2 = 1.5
    d = r1 + r2
    overlap_area = EclipsingBinaryStar._circle_overlap_area(d, r1, r2)
    assert_allclose(overlap_area, 0.0)


@pytest.mark.parametrize(
    "d, r1, r2",
    [
        # Circle centers are inside circles.
        (1.0, 3.5, 3.0),
        # Intersections and secondary center lie on the same line.
        (3.0, 5.0, 4.0),
        # Center of the secondary is still inside the primary, but near the edge.
        (4.5, 5.0, 4.0),
        # Circle centers are outside each other.
        (5.0, 3.5, 3.0),
    ],
)
def test_circle_overlap_area_with_monte_carlo(d, r1, r2):
    """Test EclipsingBinaryStar._circle_overlap_area with MC simulation"""
    # The origin is the center of the primary star, the secondary star is at (d, 0).

    n_samples = 1_000_000
    rng = np.random.default_rng(0)

    # Generate random points in the bounding box
    x_range = [-r1, d + r2]
    assert r1 >= r2, "Check the test parameters, r1 should be not less than r2."
    y_range = [-r1, r1]
    x, y = rng.uniform([x_range[0], y_range[0]], [x_range[1], y_range[1]], (n_samples, 2)).T

    # Check if the points are inside the circles and find the overlap area
    inside_primary = np.hypot(x, y) < r1
    inside_secondary = np.hypot(x - d, y) < r2
    n_overlap = np.sum(inside_primary & inside_secondary)
    desired_overlap_area = n_overlap / n_samples * (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])

    actual_overlap_area = EclipsingBinaryStar._circle_overlap_area(d, r1, r2)
    # 2-sigma tolerance
    assert_allclose(actual_overlap_area, desired_overlap_area, rtol=2 / np.sqrt(n_overlap))


def test_norm_star_center_distance_zero():
    """Test EclipsingBinaryStar._norm_star_center_distance for phase_fraction=0.0, inclination_degree=90.0"""
    assert_allclose(EclipsingBinaryStar._norm_star_center_distance(0.0, 90.0), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "phase_fraction, inclination_degree",
    [
        # Inclination is 0, the distance is always 1
        (0.0, 0.0),
        (0.25, 0.0),
        (0.3, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
        (0.95, 0.0),
        # Phase is 0.25 or 0.75, the distance is always 1
        (0.25, 0.0),
        (0.75, 10.0),
        (0.25, 45.0),
        (0.75, 57.0),
        (0.25, 85.0),
    ],
)
def test_norm_star_center_distance_one(phase_fraction, inclination_degree):
    """Test EclipsingBinaryStar._norm_star_center_distance returns 1.0 for maximum separation"""
    assert_allclose(EclipsingBinaryStar._norm_star_center_distance(phase_fraction, inclination_degree), 1.0)


def test_norm_star_center_distance_with_spherical_geometry():
    """Test EclipsingBinaryStar._norm_star_center_distance with spherical geometry implementation"""
    # We don't need reproducible random numbers here, because test should work for any random input.
    n_samples = 100
    phase = np.random.uniform(0, 1, n_samples)
    inclination = np.random.uniform(0, 90, n_samples)
    phase_radians = 2 * np.pi * phase

    # Let secondary orbit define x' y' plane, then coordinates are:
    x_prime, y_prime, z_prime = np.cos(phase_radians), np.sin(phase_radians), np.zeros_like(phase_radians)

    # Rotate x' y' z' to x y z plane with matrix multiplication.
    # Angle between z and z' is inclination, y is y', so we rotate about y axis.
    # We need to reshape inclination to (n_samples, 1).
    rotation_amt = (90.0 - inclination).reshape(-1, 1)
    rotation = Rotation.from_euler("y", rotation_amt, degrees=True)

    x, y, z = rotation.apply(np.stack([x_prime, y_prime, z_prime], axis=-1)).T
    # x is looking to the observer, y and z are the observer's plane.
    expected_distance = np.hypot(y, z)
    actual_distance = EclipsingBinaryStar._norm_star_center_distance(phase, inclination)
    assert_allclose(actual_distance, expected_distance)


def test_eclipsing_binary_star():
    """Test EclipsingBinaryStar light curve satisfies astrophysical expectations"""
    points_per_period = 1000

    distance_pc = 10
    primary_mass = 1.0 * u.M_sun
    secondary_mass = 0.6 * u.M_sun
    major_semiaxis = 10 * u.R_sun

    # Mass-radius relation for main-sequence stars
    primary_radius = 1.0 * u.R_sun * (primary_mass / const.M_sun) ** 0.8
    secondary_radius = 1.0 * u.R_sun * (secondary_mass / const.M_sun) ** 0.8
    assert primary_radius > secondary_radius, "Primary should be larger than secondary"
    # Luminosity-mass relation for main-sequence stars
    primary_lum = 1.0 * u.L_sun * (primary_mass / const.M_sun) ** 4
    secondary_lum = 1.0 * u.L_sun * (secondary_mass / const.M_sun) ** 4
    # Effective temperature from Stefan-Boltzmann law
    primary_temperature = (primary_lum / (4 * np.pi * primary_radius**2 * const.sigma_sb)) ** 0.25
    secondary_temperature = (secondary_lum / (4 * np.pi * secondary_radius**2 * const.sigma_sb)) ** 0.25
    assert primary_temperature > secondary_temperature, "Primary should be hotter than secondary"

    # Third law of Kepler, with gravitation constant
    period = np.sqrt(4 * np.pi**2 * major_semiaxis**3 / (const.G * (primary_mass + secondary_mass)))

    source = EclipsingBinaryStar(
        distance=distance_pc,
        period=period.to_value(u.day),
        t0=0.0,
        major_semiaxis=major_semiaxis.cgs.value,
        inclination=89.0,
        primary_radius=primary_radius.cgs.value,
        primary_temperature=primary_temperature.cgs.value,
        secondary_radius=secondary_radius.cgs.value,
        secondary_temperature=secondary_temperature.cgs.value,
    )
    # Times in days
    times = np.linspace(0, 2.0 * period.to_value(u.day), 2 * points_per_period + 1)
    # Wavelengths in cm
    wavelengths_aa = np.array([4500, 6000])

    fluxes = source.evaluate_sed(times, wavelengths_aa)

    # Check the fluxes are positive
    assert np.all(fluxes >= 0)
    # Check the fluxes are periodic
    assert_allclose(fluxes[0], fluxes[-1])
    assert_allclose(fluxes[:points_per_period], fluxes[points_per_period : 2 * points_per_period])
    # Check the fluxes are symmetric
    assert_allclose(fluxes[: points_per_period + 1], fluxes[points_per_period::-1])

    # Check the primary minimum is deeper than everything else
    assert np.all(fluxes[0] <= fluxes)

    # Check if primary minimum is redder than everything else
    color = -2.5 * np.log10(fluxes[:, 0] / fluxes[:, 1])
    assert np.all(color[0] >= color)
