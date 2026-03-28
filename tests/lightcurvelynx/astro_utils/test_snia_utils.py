import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from lightcurvelynx.astro_utils.snia_utils import (
    DistModFromRedshift,
    HostmassX1Distr,
    HostmassX1Func,
    SNCoordGivenPhysicalSep,
)
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from scipy.stats import norm


def test_hostmass_x1_distr():
    """Test that we can create and evaluate a hostmass to X1 distribution."""
    x1_vals = np.arange(-6.0, 6.0, 0.5)

    # At hostmass < 10.0, the distribution should be truncated at -5.0 and 5.0.
    expected = np.array(
        [
            0.0,
            0.0,
            0.0,
            2.5657549651628195e-14,
            1.5060607839169267e-11,
            4.175889141873483e-09,
            5.469335442591314e-07,
            3.383758018885159e-05,
            0.000988879045909292,
            0.01365105415030207,
            0.08901605491595148,
            0.2741887521763265,
            0.3989422804014327,
            0.3520653267642995,
            0.24197072451914337,
            0.12951759566589174,
            0.05399096651318806,
            0.01752830049356854,
            0.0044318484119380075,
            0.0008726826950457602,
            0.00013383022576488537,
            1.5983741106905478e-05,
            0.0,
            0.0,
        ]
    )
    for hostmass in np.arange(5.0, 9.0):
        dist = HostmassX1Distr(hostmass=hostmass)
        pdf_vals = dist.pdf(x1_vals)
        assert np.allclose(pdf_vals, expected)

    # At hostmass >= 10.0 the result should just be a normal with mean 0.0 and scale=1.0.
    expected2 = norm.pdf(x1_vals, loc=0, scale=1)
    for hostmass in np.arange(10.0, 12.0):
        dist = HostmassX1Distr(hostmass=hostmass)
        pdf_vals2 = dist.pdf(x1_vals)
        assert np.allclose(pdf_vals2, expected2)


def test_sample_hostmass_x1c():
    """Test that we can sample correctly from HostmassX1Func."""
    num_samples = 5

    hm_node1 = HostmassX1Func(
        hostmass=NumpyRandomFunc("uniform", low=7, high=12, seed=100),
        seed=101,
    )
    states1 = hm_node1.sample_parameters(num_samples=num_samples)
    values1 = hm_node1.get_param(states1, "function_node_result")
    assert len(values1) == num_samples
    assert len(np.unique(values1)) == num_samples

    # If we are only querying a single sample, we get a float.
    states_single = hm_node1.sample_parameters(num_samples=1)
    values_single = hm_node1.get_param(states_single, "function_node_result")
    assert isinstance(values_single, float)

    # If we create a new node with the same hostmas and the same seeds, we get the
    # same results and the same hostmasses.
    hm_node2 = HostmassX1Func(
        hostmass=NumpyRandomFunc("uniform", low=7, high=12, seed=100),
        seed=101,
    )
    states2 = hm_node2.sample_parameters(num_samples=num_samples)
    values2 = hm_node2.get_param(states2, "function_node_result")
    assert np.allclose(values1, values2)
    assert np.allclose(
        hm_node1.get_param(states1, "hostmass"),
        hm_node2.get_param(states2, "hostmass"),
    )

    # If we use a different seed for the hostmass function only, we get
    # different results (i.e. the hostmass parameter is being resampled).
    hm_node3 = HostmassX1Func(
        hostmass=NumpyRandomFunc("uniform", low=7, high=12, seed=102),
        seed=101,
    )
    states3 = hm_node3.sample_parameters(num_samples=num_samples)
    values3 = hm_node3.get_param(states3, "function_node_result")
    assert not np.allclose(values1, values3)
    assert not np.allclose(
        hm_node1.get_param(states1, "hostmass"),
        hm_node3.get_param(states3, "hostmass"),
    )


def test_dist_mod_from_redshift():
    """Test the computation of dist_mod from the redshift."""
    redshifts = [0.01, 0.02, 0.05, 0.5]
    expected = [33.08419428, 34.60580484, 36.64346629, 42.17006132]

    for idx, z in enumerate(redshifts):
        node = DistModFromRedshift(redshift=z, H0=73.0, Omega_m=0.3)
        state = node.sample_parameters(num_samples=1)
        assert node.get_param(state, "function_node_result") == pytest.approx(expected[idx])

    # Both H0 and Omega_m must be constants.
    rand_node = NumpyRandomFunc("uniform", low=0.0, high=1.0)
    with pytest.raises(ValueError):
        DistModFromRedshift(redshift=0.3, H0=rand_node, Omega_m=0.3)
    with pytest.raises(ValueError):
        DistModFromRedshift(redshift=0.3, H0=73.0, Omega_m=rand_node)


def test_sn_coord_given_physical_separation():
    """Test correct sn coor is returned."""

    host_ra = 0.0
    host_dec = 0.0
    physical_sep_kpc = 1.0
    redshift = 0.5
    H0 = 70.0
    Omega_m = 0.3

    sncoord = SNCoordGivenPhysicalSep(
        host_ra, host_dec, physical_sep_kpc, redshift, H0=H0, Omega_m=Omega_m, node_label="sncoord"
    )
    state = sncoord.sample_parameters()
    sn_ra = state["sncoord.ra"]
    sn_dec = state["sncoord.dec"]
    # calculate physical separation from host coord, sn coor, and redshift
    host = SkyCoord(host_ra * u.deg, host_dec * u.deg)
    sn = SkyCoord(sn_ra * u.deg, sn_dec * u.deg)

    sep = host.separation(sn)

    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

    DA = cosmo.angular_diameter_distance(redshift).to(u.kpc).value

    calculated_physical_sep = sep.to(u.rad).value * DA

    assert np.isclose(calculated_physical_sep, physical_sep_kpc)


def test_sn_coord_given_physical_separation_fixed():
    """Test that we always get the same sn coor if we fix the position angle."""
    sncoord = SNCoordGivenPhysicalSep(
        0.0,  # host_ra
        0.0,  # host_dec
        1.0,  # physical_sep_kpc
        0.5,  # redshift
        pos_angle=1.0,
        H0=70.0,
        Omega_m=0.3,
        node_label="sncoord_fixed_pa",
    )
    state = sncoord.sample_parameters(num_samples=100)
    assert len(np.unique(state["sncoord_fixed_pa.ra"])) == 1
    assert len(np.unique(state["sncoord_fixed_pa.dec"])) == 1


def test_sn_coord_given_physical_separation_seeded():
    """Test that we can control the randomness with a seeded random number generator."""
    sncoord = SNCoordGivenPhysicalSep(
        0.0,  # host_ra
        0.0,  # host_dec
        1.0,  # physical_sep_kpc
        0.5,  # redshift
        H0=70.0,
        Omega_m=0.3,
        node_label="sncoord_fixed_pa",
    )

    # Given the same seed, we should get the same results.
    rng1 = np.random.default_rng(seed=234)
    state1 = sncoord.sample_parameters(num_samples=1000, rng_info=rng1)

    rng2 = np.random.default_rng(seed=234)
    state2 = sncoord.sample_parameters(num_samples=1000, rng_info=rng2)

    assert np.allclose(state1["sncoord_fixed_pa.ra"], state2["sncoord_fixed_pa.ra"])
    assert np.allclose(state1["sncoord_fixed_pa.dec"], state2["sncoord_fixed_pa.dec"])

    # Given a different seed, we should get different results.
    rng3 = np.random.default_rng(seed=235)
    state3 = sncoord.sample_parameters(num_samples=1000, rng_info=rng3)
    assert not np.allclose(state1["sncoord_fixed_pa.ra"], state3["sncoord_fixed_pa.ra"])
    assert not np.allclose(state1["sncoord_fixed_pa.dec"], state3["sncoord_fixed_pa.dec"])
