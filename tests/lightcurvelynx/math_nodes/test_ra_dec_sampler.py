import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import Latitude, Longitude, SkyCoord, angular_separation
from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint
from lightcurvelynx.math_nodes.ra_dec_sampler import (
    ApproximateMOCSampler,
    ObsTableRADECSampler,
    ObsTableUniformRADECSampler,
    UniformRADEC,
)
from lightcurvelynx.math_nodes.single_value_node import SingleVariableNode
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.utils.io_utils import write_results_as_hats
from mocpy import MOC
from nested_pandas import NestedFrame


def test_uniform_ra_dec():
    """Test that we can generate numbers from a uniform distribution on a sphere."""
    sampler_node = UniformRADEC(seed=100, node_label="sampler")

    # Test we can generate a single value.
    (ra, dec) = sampler_node.generate(num_samples=1)
    assert 0.0 <= ra <= 360.0
    assert -90.0 <= dec <= 90.0

    # Generate many samples.
    num_samples = 20_000
    state = sampler_node.sample_parameters(num_samples=num_samples)

    all_ra = state["sampler"]["ra"]
    assert len(all_ra) == num_samples
    assert np.all(all_ra >= 0.0)
    assert np.all(all_ra <= 360.0)

    all_dec = state["sampler"]["dec"]
    assert len(all_dec) == num_samples
    assert np.all(all_dec >= -90.0)
    assert np.all(all_dec <= 90.0)

    # Compute histograms of RA and dec values.
    ra_bins = np.zeros(36)
    dec_bins = np.zeros(18)
    for idx in range(num_samples):
        ra_bins[int(all_ra[idx] / 10.0)] += 1
        dec_bins[int((all_dec[idx] + 90.0) / 10.0)] += 1

    # Check that all RA bins have approximately equal samples.
    expected_count = num_samples / 36
    for bin_count in ra_bins:
        assert 0.8 <= bin_count / expected_count <= 1.2

    # Check that the dec bins around the poles have less samples
    # than the bins around the equator.
    assert dec_bins[0] < 0.25 * dec_bins[9]
    assert dec_bins[17] < 0.25 * dec_bins[10]

    # Check that we can generate uniform samples in radians.
    sampler_node2 = UniformRADEC(seed=100, node_label="sampler2", use_degrees=False)
    state2 = sampler_node2.sample_parameters(num_samples=num_samples)

    all_ra = state2["sampler2"]["ra"]
    assert len(all_ra) == num_samples
    assert np.all(all_ra >= 0.0)
    assert np.all(all_ra <= 2.0 * np.pi)

    all_dec = state2["sampler2"]["dec"]
    assert len(all_dec) == num_samples
    assert np.all(all_dec >= -np.pi)
    assert np.all(all_dec <= np.pi)


def test_obstable_ra_dec_sampler():
    """Test that we can sample from am OpSim object."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    ops_data = OpSim(values)
    assert len(ops_data) == 5

    sampler_node = ObsTableRADECSampler(ops_data, radius=0.0, in_order=True)
    assert sampler_node.radius == 0.0

    # Test we can generate a single value.
    (ra, dec, time) = sampler_node.generate(num_samples=1)
    assert ra == 15.0
    assert dec == -10.0
    assert time == 0.0

    # Test we can generate multiple observations
    (ra, dec, time) = sampler_node.generate(num_samples=2)
    assert np.allclose(ra, [30.0, 15.0])
    assert np.allclose(dec, [-5.0, 0.0])
    assert np.allclose(time, [1.0, 2.0])

    # Do randomized sampling (with no offset).
    sampler_node2 = ObsTableRADECSampler(ops_data, in_order=False, radius=0.0, seed=100, node_label="sampler")
    state = sampler_node2.sample_parameters(num_samples=5000)

    # Check that the samples are uniform and consistent.
    int_times = state["sampler"]["time"].astype(int)
    assert np.allclose(state["sampler"]["ra"], values["fieldRA"][int_times])
    assert np.allclose(state["sampler"]["dec"], values["fieldDec"][int_times])
    assert len(int_times[int_times == 0]) > 750
    assert len(int_times[int_times == 1]) > 750
    assert len(int_times[int_times == 2]) > 750
    assert len(int_times[int_times == 3]) > 750
    assert len(int_times[int_times == 4]) > 750

    # Do randomized sampling with offsets (using the default OpSim radius).
    sampler_node3 = ObsTableRADECSampler(ops_data, in_order=False, seed=100, node_label="sampler")
    state = sampler_node3.sample_parameters(num_samples=5000)

    # Check that the samples are not all the centers (unique values > 500) but are within
    # the sampling radius of the corresponding pointing. We use int(time) to match the index
    # of the center pointing.
    int_times = state["sampler"]["time"].astype(int)
    assert len(np.unique(state["sampler"]["ra"])) > 500
    assert len(np.unique(state["sampler"]["dec"])) > 500
    for idx in range(5):
        dist = angular_separation(
            state["sampler"]["ra"][int_times == idx] * u.deg,
            state["sampler"]["dec"][int_times == idx] * u.deg,
            values["fieldRA"][idx] * u.deg,
            values["fieldDec"][idx] * u.deg,
        )
        assert np.all(dist <= sampler_node3.radius * u.deg)

    # We fail if no radius is provided by the OpSim or the parameters
    ops_data = OpSim(values, radius=None)
    with pytest.raises(ValueError):
        _ = ObsTableRADECSampler(ops_data)

    # But we can use the OpSim's radius if that is provided.
    ops_data = OpSim(values, radius=1.0)
    sampler_node = ObsTableRADECSampler(ops_data)
    assert sampler_node.radius == 1.0


def test_obstable_ra_dec_sampler_extra():
    """Test that we can sample from an ObsTable-like object with extra columns."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp": np.ones(5),
        "extra": np.array([10, 20, 30, 40, 50]),
    }
    ops_df = pd.DataFrame(values)
    sampler_node = ObsTableRADECSampler(ops_df, radius=0.0, extra_cols=["extra"], in_order=True)
    assert sampler_node.radius == 0.0

    # Test we can generate a single value.
    sample = sampler_node.generate(num_samples=1)
    assert len(sample) == 4
    assert sample[0] == 15.0  # ra
    assert sample[1] == -10.0  # dec
    assert sample[2] == 0.0  # time
    assert sample[3] == 10  # extra

    # We can chain on any of the column names.
    single_node = SingleVariableNode("extra", sampler_node.extra, node_label="single")
    state = single_node.sample_parameters(num_samples=2)
    assert np.allclose(state["single.extra"], [20.0, 30.0])


def test_obstable_ra_dec_sampler_from_hats(test_data_dir):
    """Test that we can sample from a HATS catalog on disk."""
    outer_dict = {
        "id": [0, 1, 2],
        "ra": [10.0, 10.1, 10.2],
        "dec": [-10.0, -9.9, -10.1],
        "nobs": [3, 2, 1],
        "z": [0.1, 0.2, 0.3],
    }
    inner_dict = {
        "mjd": [59000, 59001, 59002, 59000, 59001, 59000],
        "flux": [10.0, 12.0, 11.0, 15.0, 14.0, 13.0],
        "fluxerr": [1.0, 1.0, 1.0, 1.5, 1.5, 1.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
    }
    nested_inds = [0, 0, 0, 1, 1, 2]
    results = NestedFrame(data=outer_dict, index=[0, 1, 2])
    nested_1 = pd.DataFrame(data=inner_dict, index=nested_inds)
    results = results.add_nested(nested_1, "lightcurve")

    with tempfile.TemporaryDirectory() as dir_name:
        dir_path = Path(dir_name, "test_hats")
        write_results_as_hats(dir_path, results)

        sampler_node = ObsTableRADECSampler.from_hats(
            dir_path,
            radius=0.0,
            in_order=True,
            node_label="sampler",
            extra_cols=["z"],
        )
        assert sampler_node.radius == 0.0

        states = sampler_node.sample_parameters(num_samples=3)
        assert np.allclose(states["sampler"]["ra"], [10.0, 10.1, 10.2])
        assert np.allclose(states["sampler"]["dec"], [-10.0, -9.9, -10.1])
        assert np.allclose(states["sampler"]["z"], [0.1, 0.2, 0.3])


def test_opsim_uniform_ra_dec_sampler():
    """Test that we can sample uniformly from am OpSim object."""
    # Create an opsim with two points in different hemispheres.
    values = {
        "observationStartMJD": np.array([0.0, 1.0]),
        "fieldRA": np.array([15.0, 195.0]),
        "fieldDec": np.array([75.0, -75.0]),
        "zp_nJy": np.ones(2),
    }
    ops_data = OpSim(values)
    assert len(ops_data) == 2

    # Use a very large radius so we do not reject too many samples.
    sampler_node = ObsTableUniformRADECSampler(ops_data, radius=30.0, seed=100, node_label="sampler")
    assert sampler_node.radius == 30.0

    # Test we can generate a single value.
    ra, dec = sampler_node.generate(num_samples=1)
    assert ops_data.is_observed(ra, dec, radius=30.0)

    # Test we can generate many observations
    num_samples = 10_000
    ra, dec = sampler_node.generate(num_samples=num_samples)
    assert np.all(ops_data.is_observed(ra, dec, radius=30.0))

    # We should sample roughly uniformly from the two regions.
    northern_mask = dec > 0.0
    assert np.sum(northern_mask) > 0.4 * num_samples
    assert np.sum(northern_mask) < 0.6 * num_samples

    northern_center = SkyCoord(ra=15.0, dec=75.0, unit="deg")
    northern_pts = SkyCoord(ra[northern_mask], dec[northern_mask], unit="deg")
    assert np.all(northern_pts.separation(northern_center) <= 30.0 * u.deg)

    southern_center = SkyCoord(ra=195.0, dec=-75.0, unit="deg")
    southern_pts = SkyCoord(ra[~northern_mask], dec[~northern_mask], unit="deg")
    assert np.all(southern_pts.separation(southern_center) <= 30.0 * u.deg)

    # We fail if neither the OpSim or the sampler have a radius
    ops_data = OpSim(values, radius=None)
    with pytest.raises(ValueError):
        _ = ObsTableUniformRADECSampler(ops_data)

    # But we succeed if the OpSim has a radius.
    ops_data = OpSim(values, radius=10.0)
    sampler_node = ObsTableUniformRADECSampler(ops_data)
    assert sampler_node.radius == 10.0


def test_opsim_uniform_ra_dec_sampler_footprint():
    """Test that we can sample uniformly from am OpSim object with a footprint."""
    # Create an opsim with two points in different hemispheres.
    values = {
        "observationStartMJD": np.array([0.0, 1.0]),
        "fieldRA": np.array([15.0, 195.0]),
        "fieldDec": np.array([75.0, -75.0]),
        "zp_nJy": np.ones(2),
    }

    # Detector is a rectangle of 20 x 30 degrees.
    fp = DetectorFootprint.from_sky_rect(width=20, height=30, pixel_scale=0.1)
    ops_data = OpSim(values, detector_footprint=fp)

    # Test we can generate multiple observations
    num_samples = 200
    sampler_node = ObsTableUniformRADECSampler(ops_data, radius=50.0, seed=100, node_label="sampler")
    ra, dec = sampler_node.generate(num_samples=num_samples)

    # We should sample roughly uniformly from the two regions.
    northern_mask = dec > 0.0
    assert np.sum(northern_mask) > 0.4 * num_samples
    assert np.sum(northern_mask) < 0.6 * num_samples

    # All the points should be visible.
    assert np.all(ops_data.is_observed(ra, dec, radius=50.0))


def test_approximate_moc_sampler():
    """Test that we can create and sample from an ApproximateMOCSampler."""
    longitudes = Longitude([15.0, 90.0], unit="deg")
    latitudes = Latitude([-20.0, 20.0], unit="deg")
    moc = MOC.from_cones(
        lon=longitudes,
        lat=latitudes,
        radius=1.0 * u.deg,
        max_depth=12,
        union_strategy="large_cones",
    )
    moc_sampler = ApproximateMOCSampler(moc)

    # Test we can generate a single value.
    ra, dec = moc_sampler.generate(num_samples=1)
    assert moc.contains_skycoords(SkyCoord(ra, dec, unit="deg"))

    # Test we can generate many observations
    num_samples = 10_000
    ra, dec = moc_sampler.generate(num_samples=num_samples)
    assert np.all(moc.contains_skycoords(SkyCoord(ra, dec, unit="deg")))

    # We should sample roughly uniformly from the two regions.
    northern_mask = dec > 0.0
    assert np.sum(northern_mask) > 0.4 * num_samples
    assert np.sum(northern_mask) < 0.6 * num_samples

    assert np.all(ra[northern_mask] > 88.0)
    assert np.all(ra[northern_mask] < 92.0)
    assert np.all(ra[~northern_mask] > 13.0)
    assert np.all(ra[~northern_mask] < 17.0)
    assert np.all(dec[northern_mask] > 18.0)
    assert np.all(dec[northern_mask] < 22.0)
    assert np.all(dec[~northern_mask] > -22.0)
    assert np.all(dec[~northern_mask] < -18.0)


def test_approximate_moc_sampler_from_file():
    """Test that we can create an ApproximateMOCSampler from a MOC file."""
    longitudes = Longitude([15.0, 90.0], unit="deg")
    latitudes = Latitude([-20.0, 20.0], unit="deg")
    moc = MOC.from_cones(
        lon=longitudes,
        lat=latitudes,
        radius=1.0 * u.deg,
        max_depth=8,
        union_strategy="large_cones",
    )

    with tempfile.TemporaryDirectory() as dir_name:
        for fmt in ["fits", "json", "ascii"]:
            file_path = Path(dir_name, f"test_mock.{fmt}")

            # We can't load a non-existent file.
            with pytest.raises(FileNotFoundError):
                _ = ApproximateMOCSampler.from_file(file_path, format=fmt)

            moc.save(file_path, format=fmt)
            moc_sampler = ApproximateMOCSampler.from_file(file_path, format=fmt)

            assert moc_sampler.healpix_list is not None
