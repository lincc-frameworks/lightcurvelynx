"""Test the utility functions for computing AGN parameters. These are mostly change
detection tests to make sure the results match the code from SNANA:
https://github.com/RickKessler/SNANA/blob/master/src/gensed_AGN.py
"""

import numpy as np
import pytest
from lightcurvelynx.consts import M_SUN_G
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.agn import AGN, sample_damped_random_walk


def test_agn_compute_critical_accretion_rate():
    """Test that we can compute the accretion rate from mass of the black hole."""
    # This is a change detection test to make sure the results match previous code.

    # No BH = no accretion.
    assert AGN.compute_critical_accretion_rate(0.0) == 0.0

    # For a 1000 solar mass black hole, the accretion rate should be 1.4e21 g/s.
    assert AGN.compute_critical_accretion_rate(1000.0 * M_SUN_G) == pytest.approx(1.4e21)

    # Test that the operation is vectorized.
    num_suns = np.array([0.5, 1.0, 10.0, 50.0])
    masses = M_SUN_G * num_suns
    expected = 1.4e18 * num_suns
    assert np.allclose(AGN.compute_critical_accretion_rate(masses), expected)


def test_agn_black_hole_accretion_rate():
    """Test that we can compute the black hole's accretion rate."""
    # This is a change detection test to make sure the results match previous code.

    # No accretion => no blackhole accretion.
    assert AGN.compute_blackhole_accretion_rate(0.0, 0.5) == 0.0
    assert AGN.compute_blackhole_accretion_rate(0.0, 1.0) == 0.0
    assert AGN.compute_blackhole_accretion_rate(0.0, 2.0) == 0.0

    # For a total accretion rate of 1.4e21 g/s and different ratios,
    # compute the black hole accretion rate.
    assert AGN.compute_blackhole_accretion_rate(1.4e21, 1.0) == pytest.approx(1.4e21)
    assert AGN.compute_blackhole_accretion_rate(1.4e21, 0.5) == pytest.approx(7.0e20)
    assert AGN.compute_blackhole_accretion_rate(1.4e21, 2.0) == pytest.approx(2.8e21)

    # Test that the operation is vectorized.
    rates = np.array([0.5, 1.0, 10.0, 50.0])
    ratios = np.array([1.0, 1.0, 2.0, 2.0])
    assert np.allclose(AGN.compute_blackhole_accretion_rate(rates, ratios), rates * ratios)


def test_agn_compute_bolometric_luminosity():
    """Test that we can compute the bolometric luminosity from mass of the black hole."""
    # This is a change detection test to make sure the results match previous code.

    # No BH = no luminosity.
    assert AGN.compute_bolometric_luminosity(1.0, 0.0) == 0.0

    # For a 100 solar mass black hole and 1.0 ratio, the bolometric luminosity should be 1.26e40
    assert AGN.compute_bolometric_luminosity(1.0, 100.0 * M_SUN_G) == pytest.approx(1.26e40)

    # For a 100 solar mass black hole and 0.5 ratio, the bolometric luminosity should be 6.3e39
    assert AGN.compute_bolometric_luminosity(0.5, 100.0 * M_SUN_G) == pytest.approx(6.3e39)

    # Test that the operation is vectorized.
    num_suns = np.array([0.5, 1.0, 10.0, 50.0])
    ratios = np.array([1.0, 1.0, 2.0, 2.0])
    masses = M_SUN_G * num_suns

    expected = 1.26e38 * num_suns * ratios
    assert np.allclose(AGN.compute_bolometric_luminosity(ratios, masses), expected)


def test_agn_compute_mag_i():
    """Test that we can compute the i band magnitude from the bolometric luminosity."""
    # This is a change detection test to make sure the results match previous code.

    assert AGN.compute_mag_i(1.0) == pytest.approx(90.0)

    # Test that the operation is vectorized.
    l_bol = np.array([0.5, 1.0, 10.0, 50.0])
    expected = np.array([90.75257499, 90.0, 87.5, 85.75257499])
    assert np.allclose(AGN.compute_mag_i(l_bol), expected)


def test_agn_compute_r_0():
    """Test that we can compute an AGN's r_0 from its r_in."""
    # This is a change detection test to make sure the results match previous code.

    assert AGN.compute_r_0(1.0) == pytest.approx(1.361111111111111)

    # Test that the operation is vectorized.
    r_in = np.array([0.5, 1.0, 10.0, 50.0])
    expected = 1.361111111111111 * r_in
    assert np.allclose(AGN.compute_r_0(r_in), expected)


def test_agn_compute_temp_at_r_0():
    """Test that we can compute the effective temperature at r0."""
    # This is a change detection test to make sure the results match previous code.

    assert AGN.compute_temp_at_r_0(1000.0, 100.0, 5.0) == pytest.approx(0.28248581706023856)


def test_compute_structure_function_at_inf():
    """Test that we can compute the structure function at infinity."""
    # This is a change detection test to make sure the results match previous code.
    sf_inf4000 = AGN.compute_sf_inf_4000aa()
    sf_inf = AGN.compute_structure_function_at_inf(
        sf_inf4000, np.array([1000.0, 2000.0, 3000.0, 4000.0]) * 1e-8
    )
    expected = np.array([0.02772937, 0.05545874, 0.08318811, 0.11091748])
    assert np.allclose(sf_inf, expected)


def test_compute_tau():
    """Test that we can compute the DRW tau."""
    # This is a change detection test to make sure the results match previous code.
    tau_4000 = AGN.compute_tau_4000aa()
    tau = AGN.compute_tau(tau_4000, np.array([1000.0, 2000.0, 3000.0, 4000.0]) * 1e-8)
    expected = np.array([146.19752104, 292.39504207, 438.59256311, 584.79008414])
    assert np.allclose(tau, expected)


def test_sample_damped_random_walk():
    """Test that we can sample from a damped random walk."""
    rng = np.random.default_rng(100)

    tau_v = np.array([1.0, 1.0, 10.0, 10.0, 100.0])
    sf_inf = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
    times = np.arange(0.0, 20.0, 1.0)
    delta_m = sample_damped_random_walk(times, tau_v, sf_inf, 0.0, rng=rng)
    assert delta_m.shape == (20, 5)

    # The mean of delta_m for each wavelength should be around sf_inf.
    assert np.allclose(np.mean(delta_m, axis=0), sf_inf, atol=0.1)

    # We fail with non-monotonically increasing times.
    with pytest.raises(ValueError):
        _ = sample_damped_random_walk(np.array([0.0, 1.0, 0.0]), tau_v, sf_inf, 0.0)

    # We fail with mismatched tau_v and sf_inf.
    with pytest.raises(ValueError):
        _ = sample_damped_random_walk(times, tau_v[:3], sf_inf, 0.0)


def test_create_agn():
    """Test that we can create an AGN object and the derived parameters are correct."""
    # Select the black hole mass uniformly between 1e9 and 2e9 solar masses.
    bh_mass_sampler = NumpyRandomFunc("uniform", low=1e9, high=2e9, seed=100)
    agn_node = AGN(
        t0=0.0,
        blackhole_mass=bh_mass_sampler,
        redshift=1.0,
        edd_ratio=0.9,
        cosmology="Planck18",
        node_label="AGN_0",
    )
    state = agn_node.sample_parameters(num_samples=10_000)

    # Check that the parameters are within the expected ranges.
    bh_masses = state["AGN_0"]["blackhole_mass"]
    assert np.all(bh_masses >= 1e9)
    assert np.all(bh_masses <= 2e9)
    assert np.unique(bh_masses).size > 1_000

    assert np.allclose(state["AGN_0"]["critical_accretion_rate"], 1.4e18 * bh_masses)
    assert np.allclose(
        state["AGN_0"]["blackhole_accretion_rate"], 0.9 * state["AGN_0"]["critical_accretion_rate"]
    )
    assert np.allclose(
        state["AGN_0"]["bolometric_luminosity"], AGN.compute_bolometric_luminosity(0.9, bh_masses * M_SUN_G)
    )
    assert np.allclose(state["AGN_0"]["mag_i"], AGN.compute_mag_i(state["AGN_0"]["bolometric_luminosity"]))

    # Check that we can sample from the AGN.
    single_state = agn_node.sample_parameters(num_samples=1)
    times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    wavelengths = np.array([1000.0, 2000.0, 3000.0, 4000.0])

    fluxes = agn_node.compute_sed(times, wavelengths, single_state)
    assert fluxes.shape == (5, 4)
    assert np.all(fluxes > 0.0)


def test_create_agn_seed():
    """Test that we can set the seed for an AGN object and get the same results."""
    # Select the black hole mass uniformly between 1e9 and 2e9 solar masses.
    agn_node1 = AGN(
        t0=0.0,
        blackhole_mass=1e9,
        redshift=1.0,
        edd_ratio=0.9,
        inclination_rad=np.pi / 4,
        cosmology="Planck18",
        node_label="AGN_1",
        seed=100,
    )
    state1 = agn_node1.sample_parameters(num_samples=1)
    assert state1["AGN_1"]["agn_seed"] == 100

    # Check that we can sample from the AGN.
    times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    wavelengths = np.array([1000.0, 2000.0, 3000.0, 4000.0])

    fluxes1 = agn_node1.evaluate_sed(times, wavelengths, state1)
    assert fluxes1.shape == (5, 4)
    assert np.all(fluxes1 > 0.0)

    # Create another AGN with the same seed and check that we get the same results.
    agn_node2 = AGN(
        t0=0.0,
        blackhole_mass=1e9,
        redshift=1.0,
        edd_ratio=0.9,
        inclination_rad=np.pi / 4,
        cosmology="Planck18",
        node_label="AGN_2",
        seed=100,
    )
    state2 = agn_node2.sample_parameters(num_samples=1)
    assert state2["AGN_2"]["agn_seed"] == 100
    fluxes2 = agn_node2.evaluate_sed(times, wavelengths, state2)

    assert np.allclose(fluxes1, fluxes2)

    # Create another AGN without a set seed and check that we get different results.
    agn_node3 = AGN(
        t0=0.0,
        blackhole_mass=1e9,
        redshift=1.0,
        edd_ratio=0.9,
        inclination_rad=np.pi / 4,
        cosmology="Planck18",
        node_label="AGN_3",
    )
    state3 = agn_node3.sample_parameters(num_samples=1)
    assert state3["AGN_3"]["agn_seed"] != 100

    fluxes3 = agn_node3.evaluate_sed(times, wavelengths, state3)
    assert not np.allclose(fluxes1, fluxes3)
    assert not np.allclose(fluxes2, fluxes3)
