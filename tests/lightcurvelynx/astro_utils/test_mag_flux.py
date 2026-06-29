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
    """Test conversion from nJy flux/error to AB magnitude/error."""
    flux = np.array([3631e9, 1e9, 3631.0])
    desired_mag = np.array([0.0, 8.9, 22.5])

    # No flux error: return magnitudes only, not a tuple.
    mag_only = flux2mag(flux, flux_err_njy=None)

    assert not isinstance(mag_only, tuple)
    np.testing.assert_allclose(mag_only, desired_mag, atol=1e-3)

    # Use 10%, 20%, and 1% fractional flux uncertainties, respectively.
    flux_err = np.array(
        [
            0.10 * 3631e9,
            0.20 * 1e9,
            0.01 * 3631.0,
        ]
    )

    # mag_err = (2.5 / ln(10)) * (flux_err / flux)
    desired_mag_err = np.array(
        [
            0.1085736205,  # 1.085736205 * 0.10
            0.2171472410,  # 1.085736205 * 0.20
            0.0108573620,  # 1.085736205 * 0.01
        ]
    )

    result = flux2mag(flux, flux_err)

    assert isinstance(result, tuple)
    assert len(result) == 2

    mag, mag_err = result
    np.testing.assert_allclose(mag, desired_mag, atol=1e-3)
    np.testing.assert_allclose(mag_err, desired_mag_err, atol=1e-3)


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
