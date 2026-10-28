import importlib

import numpy as np
import pytest
from citation_compass import find_in_citations

from lightcurvelynx import _LIGHTCURVELYNX_TEST_DATA_DIR
from lightcurvelynx.astro_utils.dustmap import ConstantHemisphereDustMap, DustmapWrapper, SFDMap
from lightcurvelynx.math_nodes.given_sampler import GivenValueList


def test_constant_hemisphere_dust_map():
    """Test that we can create and directly query a ConstantHemisphereDustMap object."""
    dust_map = ConstantHemisphereDustMap(north_ebv=0.5, south_ebv=0.4)
    assert dust_map.north_ebv == 0.5
    assert dust_map.south_ebv == 0.4

    # Check that the dust map returns the correct values for different locations.
    assert dust_map.compute_ebv(0.0, 45.0) == 0.5
    assert dust_map.compute_ebv(-10.0, 70.0) == 0.5
    assert dust_map.compute_ebv(5.0, 0.001) == 0.5
    assert dust_map.compute_ebv(0.0, -45.0) == 0.4
    assert dust_map.compute_ebv(20.0, -0.001) == 0.4

    ebvs = dust_map.compute_ebv(
        [0.0, 0.0, 20.0, 340.0, 250.0, 180.0],
        [45.0, -45.0, 15.0, -70.0, -90.0, 0.0],
    )
    assert np.array_equal(ebvs, [0.5, 0.4, 0.5, 0.4, 0.4, 0.5])


def test_dust_map_wrapper():
    """Test that we can create and sample a wrapped dust map."""
    dec_vals = np.arange(-90.0, 90.0, 15.0)
    ra_vals = np.arange(0.0, 360.0, 30.0)
    fake_dust_map = ConstantHemisphereDustMap(north_ebv=0.8, south_ebv=0.1)

    dust_node1 = DustmapWrapper(
        dust_map=fake_dust_map,
        ra=GivenValueList(ra_vals, stateful=False),
        dec=GivenValueList(dec_vals, stateful=False),
        node_label="dust",
    )
    samples = dust_node1.sample_parameters(num_samples=len(dec_vals))
    assert np.all(samples["dust"]["ebv"][dec_vals >= 0] == 0.8)
    assert np.all(samples["dust"]["ebv"][dec_vals < 0] == 0.1)

    # Try adding an adjustment function to the dust map.
    def _ebv_adjust(value):
        return value + 0.1

    dust_node2 = DustmapWrapper(
        dust_map=fake_dust_map,
        ebv_func=_ebv_adjust,
        ra=GivenValueList(ra_vals, stateful=False),
        dec=GivenValueList(dec_vals, stateful=False),
        node_label="dust",
    )
    samples = dust_node2.sample_parameters(num_samples=len(dec_vals))
    assert np.all(samples["dust"]["ebv"][dec_vals >= 0] == 0.9)
    assert np.all(samples["dust"]["ebv"][dec_vals < 0] == 0.2)


def test_sfdmap():
    """Test that we can create and sample an SFDObject."""
    if importlib.util.find_spec("sfdmap2") is None:
        pytest.skip("sfdmap2 is not installed, skipping SFDMap tests.")

    # Use data from a fake map with zero dust in the high North latitudes
    # and high dust in the South.
    dust = SFDMap(
        _LIGHTCURVELYNX_TEST_DATA_DIR / "dustmaps" / "sfdmap2",
        ra=GivenValueList([125.0, -45.0, 10.0], stateful=False),
        dec=GivenValueList([60.0, -30.0, -45.0], stateful=False),
        node_label="dust",
    )

    # Test that we can call the compute_ebv function directly with specific coordinates
    # (scalar and array) without sampling RA and dec.
    assert dust.compute_ebv(125.0, 60.0) == pytest.approx(0.0, abs=0.001)
    assert dust.compute_ebv(-45.0, -30.0) == pytest.approx(86.0, abs=0.001)

    ebvs = dust.compute_ebv(np.array([125.0, -45.0, -10.0]), np.array([60.0, -30.0, 70.0]))
    assert np.allclose(ebvs, [0.0, 86.0, 0.0], atol=0.001)

    # Test that we can sample SFDMap as a ParameterNode, this will pull
    # RA and Dec values from the given lists.

    samples = dust.sample_parameters(num_samples=3)
    assert np.allclose(samples["dust"]["ebv"], [0.0, 86.0, 86.0], atol=0.01)

    # We fail if we try to create a SFDMap without RA and dec setters.
    with pytest.raises(ValueError):
        _ = SFDMap(_LIGHTCURVELYNX_TEST_DATA_DIR / "dustmaps" / "sfdmap2")


def test_dustmap_citation():
    """Test the citations for the DustMapWrapper model."""
    fake_dust_map = ConstantHemisphereDustMap(north_ebv=0.8, south_ebv=0.1)
    _ = DustmapWrapper(dust_map=fake_dust_map, ra=0.0, dec=0.0)

    citations = find_in_citations("DustmapWrapper")
    for citation in citations:
        assert "Green 2018, JOSS, 3(26), 695" in citation
        assert "https://github.com/gregreen/dustmaps" in citation
