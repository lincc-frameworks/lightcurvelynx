import numpy as np
import pytest
from lightcurvelynx.astro_utils.spectrograph import Spectrograph


def test_create_spectrograph_from_regular_grid():
    """Test that we can create and query a Spectrograph object."""
    spgraph = Spectrograph.from_regular_grid(wave_start=3000, wave_end=11000, bin_width=5.0)
    assert spgraph.instrument == "Spectrograph"
    assert spgraph.num_bins == 1600
    assert len(spgraph) == 1600  # (11000 - 3000) / 5 = 1600 bins
    assert np.array_equal(spgraph.query_waves, np.arange(3000 + 2.5, 11000, 5.0))

    l_val, h_val = spgraph.wave_bounds()
    assert l_val == 3000
    assert h_val == 11000

    for i in range(len(spgraph)):
        assert spgraph.bin_widths[i] == pytest.approx(5.0)

    assert str(spgraph) == "Spectrograph (spectra) [3000.0A - 11000.0A]"

    # Two dimensional fluxes to spec_fluxes
    values = np.random.random((10, len(spgraph)))
    spec_fluxes = spgraph.evaluate(values)
    expected = values * spgraph.bin_widths[np.newaxis, :]
    assert np.allclose(spec_fluxes, expected)

    # Three dimensional fluxes to spec_fluxes
    values_3d = np.random.random((4, 10, len(spgraph)))
    spec_fluxes_3d = spgraph.evaluate(values_3d)
    expected_3d = values_3d * spgraph.bin_widths[np.newaxis, np.newaxis, :]
    assert np.allclose(spec_fluxes_3d, expected_3d)

    # Test that we fail to create a Spectrograph object with invalid parameters.
    with pytest.raises(ValueError):
        _ = Spectrograph.from_regular_grid(wave_start=5000, wave_end=4000, bin_width=5.0)
    with pytest.raises(ValueError):
        _ = Spectrograph.from_regular_grid(wave_start=4000, wave_end=5000, bin_width=-5.0)


def test_create_spectrograph_from_irregular_grid():
    """Test that we can create and query a Spectrograph object."""
    waves_min = np.array([3500.0, 4000.0, 5000.0, 7000.0, 7500.0, 8000.0])
    waves_max = np.array([4000.0, 5000.0, 7000.0, 7500.0, 8000.0, 8500.0])
    spgraph = Spectrograph(waves_min, waves_max, instrument="custom_spectrograph")
    assert spgraph.instrument == "custom_spectrograph"
    assert spgraph.num_bins == 6
    assert len(spgraph) == 6
    assert np.array_equal(spgraph.query_waves, (waves_min + waves_max) / 2)

    l_val, h_val = spgraph.wave_bounds()
    assert l_val == 3500.0
    assert h_val == 8500.0
    assert np.allclose(spgraph.bin_widths, [500.0, 1000.0, 2000.0, 500.0, 500.0, 500.0])

    assert str(spgraph) == "custom_spectrograph (spectra) [3500.0A - 8500.0A]"
    # Two dimensional fluxes to spec_fluxes
    values = np.random.random((10, len(spgraph)))
    spec_fluxes = spgraph.evaluate(values)
    expected = values * spgraph.bin_widths[np.newaxis, :]
    assert np.allclose(spec_fluxes, expected)

    # Three dimensional fluxes to spec_fluxes
    values_3d = np.random.random((4, 10, len(spgraph)))
    spec_fluxes_3d = spgraph.evaluate(values_3d)
    expected_3d = values_3d * spgraph.bin_widths[np.newaxis, np.newaxis, :]
    assert np.allclose(spec_fluxes_3d, expected_3d)

    # We fail a query if the flux density matrix is empty.
    with pytest.raises(ValueError):
        _ = spgraph.evaluate(np.array([]))

    # We fail a query if the flux density matrix is more than 3-dimensional.
    with pytest.raises(ValueError):
        _ = spgraph.evaluate(np.array([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]]))

    # Test that we fail to create a Spectrograph object with invalid parameters.
    with pytest.raises(ValueError):
        # waves_min and waves_max have different lengths
        _ = Spectrograph([1000.0, 2000.0], [2000.0, 3000.0, 4000.0])
    with pytest.raises(ValueError):
        # waves_max is not greater than waves_min
        _ = Spectrograph([1000.0, 2000.0, 3000.0], [2000.0, 1500.0, 4000.0])


def test_create_spectrograph_from_bad_bins():
    """Test that we fail if the bins overlap or are not in increasing order."""
    waves_min_overlap = np.array([3500.0, 4000.0, 5000.0, 7000.0, 7500.0])
    waves_max_overlap = np.array([4000.0, 5000.0, 7100.0, 7500.0, 8000.0])
    with pytest.raises(ValueError):
        _ = Spectrograph(waves_min_overlap, waves_max_overlap)

    waves_min_ooo = np.array([3500.0, 4000.0, 7000.0, 5000.0])
    waves_max_ooo = np.array([4000.0, 5000.0, 7500.0, 6000.0])
    with pytest.raises(ValueError):
        _ = Spectrograph(waves_min_ooo, waves_max_ooo)


def test_spectrograph_equals():
    """Test that we can compare two Spectrograph objects for equality."""
    spgraph1 = Spectrograph.from_regular_grid(wave_start=4000, wave_end=8000, bin_width=5.0)
    spgraph2 = Spectrograph.from_regular_grid(wave_start=4000, wave_end=8000, bin_width=5.0)
    spgraph3 = Spectrograph.from_regular_grid(wave_start=4000, wave_end=8000, bin_width=10.0)
    spgraph4 = Spectrograph.from_regular_grid(wave_start=3000, wave_end=8000, bin_width=5.0)
    spgraph5 = Spectrograph.from_regular_grid(wave_start=4000, wave_end=9000, bin_width=5.0)

    assert spgraph1 == spgraph2
    assert spgraph1 != spgraph3
    assert spgraph1 != spgraph4
    assert spgraph1 != spgraph5


def test_create_spectrograph_with_scale():
    """Test that we can create and query a Spectrograph object."""
    scale = np.array([0.5, 1.0, 1.0, 1.0, 0.8])
    spgraph = Spectrograph.from_regular_grid(wave_start=4000, wave_end=5000, bin_width=200.0, scale=scale)
    assert np.allclose(spgraph.query_waves, np.array([4100.0, 4300.0, 4500.0, 4700.0, 4900.0]))
    assert spgraph.num_bins == 5

    # One dimensional fluxes to spec_fluxes
    measurement = np.array([50.0, 40.0, 20.0, 20.0, 10.0])
    expected = np.array([25.0, 40.0, 20.0, 20.0, 8.0]) * spgraph.bin_widths
    results = spgraph.evaluate(measurement)
    assert np.allclose(results, expected)

    # Two dimensional fluxes to spec_fluxes
    measurement = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5.0, 15.0, 25.0, 35.0, 45.0],
        ]
    )
    expected = (
        np.array(
            [
                [5.0, 20.0, 30.0, 40.0, 40.0],
                [2.5, 15.0, 25.0, 35.0, 36.0],
            ]
        )
        * spgraph.bin_widths[np.newaxis, :]
    )
    results = spgraph.evaluate(measurement)
    assert np.allclose(results, expected)

    # Three dimensional fluxes to bandfluxes
    measurement = np.array(
        [
            [
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [5.0, 15.0, 25.0, 35.0, 45.0],
            ],
            [
                [8.0, 18.0, 28.0, 38.0, 48.0],
                [4.0, 14.0, 24.0, 34.0, 44.0],
            ],
        ]
    )
    expected = (
        np.array(
            [
                [
                    [5.0, 20.0, 30.0, 40.0, 40.0],
                    [2.5, 15.0, 25.0, 35.0, 36.0],
                ],
                [
                    [4.0, 18.0, 28.0, 38.0, 38.4],
                    [2.0, 14.0, 24.0, 34.0, 35.2],
                ],
            ]
        )
        * spgraph.bin_widths[np.newaxis, np.newaxis, :]
    )
    results = spgraph.evaluate(measurement)
    assert np.allclose(results, expected)

    # Test equality with scales.
    spgraph2 = Spectrograph(spgraph.waves_min, spgraph.waves_max, scale=scale)
    assert spgraph == spgraph2

    different_scale = np.array([0.5, 1.0, 1.0, 1.0, 0.9])
    spgraph3 = Spectrograph(spgraph.waves_min, spgraph.waves_max, scale=different_scale)
    assert spgraph != spgraph3

    spgraph4 = Spectrograph(spgraph.waves_min, spgraph.waves_max, scale=None)
    assert spgraph != spgraph4

    # Test with a mismatched scale length.
    bad_scale = np.array([1.0, 0.8])
    with pytest.raises(ValueError):
        _ = Spectrograph(spgraph.waves_min, spgraph.waves_max, scale=bad_scale)


def test_create_spectrograph_max_wave_step():
    """Test that we can create and query a Spectrograph object with a max_wave_step."""
    wave_min = np.array([3500.0, 3600.0, 3650.0, 3700.0, 3800.0])
    wave_max = np.array([3600.0, 3650.0, 3700.0, 3800.0, 4000.0])
    spgraph = Spectrograph(wave_min, wave_max, max_wave_step=50.0)

    # The bin boundaries and widths should be unchanged.
    assert np.allclose(spgraph.waves_min, wave_min)
    assert np.allclose(spgraph.waves_max, wave_max)
    assert np.allclose(spgraph.bin_widths, wave_max - wave_min)
    assert spgraph.num_bins == 5

    # The query waves should be modified to respect the max_wave_step. Note that the samples
    # are chosen so they are evenly spaced within each bin (instead of evenly spaced across
    # the entire wavelength range).
    sample_waves = [
        3533.3,  # Bin 0 - Sample 0
        3566.7,  # Bin 0 - Sample 1
        3625.0,  # Bin 1 - Sample 0
        3675.0,  # Bin 2 - Sample 0
        3733.3,  # Bin 3 - Sample 0
        3766.7,  # Bin 3 - Sample 1
        3840.0,  # Bin 4 - Sample 0
        3880.0,  # Bin 4 - Sample 1
        3920.0,  # Bin 4 - Sample 2
        3960.0,  # Bin 4 - Sample 3
    ]
    assert np.allclose(spgraph.query_waves, sample_waves, atol=0.2)

    # Test one dimensional fluxes to spec_fluxes
    measurement1 = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    expected1 = np.array([5.0, 20.0, 30.0, 45.0, 75.0]) * spgraph.bin_widths
    results1 = spgraph.evaluate(measurement1)
    assert np.allclose(results1, expected1)
    assert results1.shape == (spgraph.num_bins,)

    # Test two dimensional flux densities to spec_fluxes
    measurement2 = np.array(
        [
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0],
        ]
    )  # Number query waves by num times
    expected2 = (
        np.array(
            [
                [5.0, 20.0, 30.0, 45.0, 75.0],
                [10.0, 25.0, 35.0, 50.0, 80.0],
            ]
        )
        * spgraph.bin_widths[np.newaxis, :]
    )
    results2 = spgraph.evaluate(measurement2)
    assert np.allclose(results2, expected2)
    assert results2.shape == (2, spgraph.num_bins)

    # Test three dimensional fluxes to bandfluxes
    measurement3 = np.array([measurement2, measurement2 + 2.0])
    expected3 = (
        np.array(
            [
                [
                    [5.0, 20.0, 30.0, 45.0, 75.0],
                    [10.0, 25.0, 35.0, 50.0, 80.0],
                ],
                [
                    [7.0, 22.0, 32.0, 47.0, 77.0],
                    [12.0, 27.0, 37.0, 52.0, 82.0],
                ],
            ]
        )
        * spgraph.bin_widths[np.newaxis, np.newaxis, :]
    )
    results3 = spgraph.evaluate(measurement3)
    assert np.allclose(results3, expected3)
    assert results3.shape == (2, 2, spgraph.num_bins)

    # We fail to create a Spectrograph object with an invalid max_wave_step (<= 0.0).
    with pytest.raises(ValueError):
        _ = Spectrograph(wave_min, wave_max, max_wave_step=-50.0)


def test_create_spectrograph_unneeded_max_wave_step():
    """Test that if we set max_wave_step high enough, we just use the midpoints."""
    wave_min = np.array([3500.0, 3600.0, 3650.0, 3700.0, 3800.0])
    wave_max = np.array([3600.0, 3650.0, 3700.0, 3800.0, 4000.0])
    spgraph = Spectrograph(wave_min, wave_max, max_wave_step=2000.0)

    # The bin boundaries and widths should be unchanged.
    assert np.allclose(spgraph.waves_min, wave_min)
    assert np.allclose(spgraph.waves_max, wave_max)
    assert np.allclose(spgraph.bin_widths, wave_max - wave_min)
    assert spgraph.num_bins == 5

    # The query waves should still be the bin midpoints.
    assert np.allclose(spgraph.query_waves, (wave_min + wave_max) / 2)
