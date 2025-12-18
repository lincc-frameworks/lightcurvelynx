"""Test magnitude-flux conversion utilities."""

import numpy as np
import pytest
from lightcurvelynx.astro_utils.mag_flux import (
    Flux2MagNode,
    Mag2FluxNode,
    flux2mag,
    mag2flux,
)
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc


def test_flux2mag():
    """Test that mag2flux is correct."""
    flux = np.array([3631e9, 1e9, 3631])
    desired_mag = np.array([0, 8.9, 22.5])
    np.testing.assert_allclose(flux2mag(flux), desired_mag, atol=1e-3)


def test_mag2flux():
    """Test that flux2mag is correct."""
    mag = np.array([0, 8.9, 8.9 + 2.5 * 9])
    desired_flux = np.array([3631e9, 1e9, 1])
    np.testing.assert_allclose(mag2flux(mag), desired_flux, rtol=1e-3)


def test_mag2flux2mag():
    """Test that mag2flux inverts flux2mag."""
    rng = np.random.default_rng(42)
    mag = rng.uniform(-10, 30, 1024)
    flux = mag2flux(mag)
    mag2 = flux2mag(flux)
    np.testing.assert_allclose(mag, mag2, rtol=1e-10)


def test_flux2mag2flux():
    """Test that flux2mag inverts mag2flux."""
    rng = np.random.default_rng(43)
    flux = rng.uniform(1e-3, 1e3, 1024)
    mag = flux2mag(flux)
    flux2 = mag2flux(mag)
    np.testing.assert_allclose(flux, flux2, rtol=1e-10)


def test_flux2magnode():
    """Test the computation of the Flux2MagNode."""
    fluxes = np.array([3631e9, 1e9, 3631])
    expected = flux2mag(fluxes)

    for idx, f in enumerate(fluxes):
        node = Flux2MagNode(flux_njy=f)
        state = node.sample_parameters(num_samples=1)
        assert node.get_param(state, "function_node_result") == pytest.approx(expected[idx])


def test_mag2fluxnode():
    """Test the computation of the Mag2FluxNode."""
    mags = np.array([0, 8.9, 8.9 + 2.5 * 9])
    expected = mag2flux(mags)

    for idx, m in enumerate(mags):
        node = Mag2FluxNode(mag=m)
        state = node.sample_parameters(num_samples=1)
        assert node.get_param(state, "function_node_result") == pytest.approx(expected[idx])


def test_flux2magnode_chained():
    """Test chaining Flux2MagNode and Mag2FluxNode."""
    flux_node = NumpyRandomFunc("uniform", low=100.0, high=1e6, seed=101, node_label="node1")
    mag_node = Flux2MagNode(flux_njy=flux_node, node_label="node2")

    num_samples = 10
    state_flux = mag_node.sample_parameters(num_samples=num_samples)
    assert np.allclose(
        state_flux["node2"]["function_node_result"],
        flux2mag(state_flux["node1"]["function_node_result"]),
    )
