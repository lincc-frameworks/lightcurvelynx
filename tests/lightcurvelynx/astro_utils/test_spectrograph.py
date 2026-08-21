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
    assert np.allclose(spec_fluxes, values)

    # Three dimensional fluxes to spec_fluxes
    values_3d = np.random.random((4, 10, len(spgraph)))
    spec_fluxes_3d = spgraph.evaluate(values_3d)
    assert np.allclose(spec_fluxes_3d, values_3d)

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
    assert np.allclose(spec_fluxes, values)

    # Three dimensional fluxes to spec_fluxes
    values_3d = np.random.random((4, 10, len(spgraph)))
    spec_fluxes_3d = spgraph.evaluate(values_3d)
    assert np.allclose(spec_fluxes_3d, values_3d)

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
        # waves_min is not in strictly increasing order
        _ = Spectrograph([1000.0, 3000.0, 2999.0], [2000.0, 3000.0, 4000.0])
    with pytest.raises(ValueError):
        # waves_max is not in strictly increasing order
        _ = Spectrograph([1000.0, 2000.0, 3000.0], [2000.0, 1000.0, 4000.0])
    with pytest.raises(ValueError):
        # waves_max is not greater than waves_min
        _ = Spectrograph([1000.0, 2000.0, 3000.0], [2000.0, 1500.0, 4000.0])


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

    # Two dimensional fluxes to spec_fluxes
    measurement = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5.0, 15.0, 25.0, 35.0, 45.0],
        ]
    )
    expected = np.array(
        [
            [5.0, 20.0, 30.0, 40.0, 40.0],
            [2.5, 15.0, 25.0, 35.0, 36.0],
        ]
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
    expected = np.array(
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
    results = spgraph.evaluate(measurement)
    assert np.allclose(results, expected)

    # Test equality with scales.
    spgraph2 = Spectrograph(spgraph.waves_min, spgraph.waves_max, scale=scale)
    assert spgraph == spgraph2

    different_scale = np.array([0.5, 1.0, 1.0, 1.0, 0.9])
    spgraph3 = Spectrograph(spgraph.waves_min, spgraph.waves_max, scale=different_scale)
    assert spgraph != spgraph3

    # Test with a mismatched scale length.
    bad_scale = np.array([1.0, 0.8])
    with pytest.raises(ValueError):
        _ = Spectrograph(spgraph.waves_min, spgraph.waves_max, scale=bad_scale)
