import numpy as np
import pytest
from lightcurvelynx.astro_utils.spectrograph import Spectrograph


def test_create_spectrograph_from_regular_grid():
    """Test that we can create and query a Spectrograph object."""
    spgraph = Spectrograph.from_regular_grid(wave_start=3000, wave_end=11000, bin_width=5.0)
    assert spgraph.instrument == "Spectrograph"
    assert len(spgraph) == 1600  # (11000 - 3000) / 5 = 1600 bins
    assert np.array_equal(spgraph.waves, np.arange(3000 + 2.5, 11000, 5.0))

    l_val, h_val = spgraph.wave_bounds()
    assert l_val == 3000
    assert h_val == 11000

    for i in range(len(spgraph)):
        assert spgraph.bin_width(i) == pytest.approx(5.0)

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
    waves = np.array([3500.0, 4000.0, 5000.0, 7000.0, 7500.0, 8000.0])
    spgraph = Spectrograph(waves, instrument="custom_spectrograph")
    assert spgraph.instrument == "custom_spectrograph"
    assert len(spgraph) == 6
    assert np.array_equal(spgraph.waves, waves)

    l_val, h_val = spgraph.wave_bounds()
    assert l_val == 3250.0
    assert h_val == 8250.0

    assert spgraph.bin_width(0) == pytest.approx(500.0)
    assert spgraph.bin_width(1) == pytest.approx(750.0)
    assert spgraph.bin_width(2) == pytest.approx(1500.0)
    assert spgraph.bin_width(3) == pytest.approx(1250.0)
    assert spgraph.bin_width(4) == pytest.approx(500.0)
    assert spgraph.bin_width(5) == pytest.approx(500.0)

    assert str(spgraph) == "custom_spectrograph (spectra) [3250.0A - 8250.0A]"
    # Two dimensional fluxes to spec_fluxes
    values = np.random.random((10, len(spgraph)))
    spec_fluxes = spgraph.evaluate(values)
    assert np.allclose(spec_fluxes, values)

    # Three dimensional fluxes to spec_fluxes
    values_3d = np.random.random((4, 10, len(spgraph)))
    spec_fluxes_3d = spgraph.evaluate(values_3d)
    assert np.allclose(spec_fluxes_3d, values_3d)

    # Test that we fail to create a Spectrograph object with invalid parameters.
    bad_waves = np.array([4000.0, 3500.0, 5000.0])
    with pytest.raises(ValueError):
        _ = Spectrograph(bad_waves)


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
    assert np.allclose(spgraph.waves, np.array([4100.0, 4300.0, 4500.0, 4700.0, 4900.0]))

    # Two dimensional fluxes to bandfluxes
    input = np.array(
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
    results = spgraph.evaluate(input)
    assert np.allclose(results, expected)

    # Three dimensional fluxes to bandfluxes
    input = np.array(
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
    results = spgraph.evaluate(input)
    assert np.allclose(results, expected)

    # Test equality with scales.
    waves = np.copy(spgraph.waves)
    spgraph2 = Spectrograph(waves, scale=scale)
    assert spgraph == spgraph2

    different_scale = np.array([0.5, 1.0, 1.0, 1.0, 0.9])
    spgraph3 = Spectrograph(waves, scale=different_scale)
    assert spgraph != spgraph3

    # Test with a mismatched scale length.
    bad_scale = np.array([1.0, 0.8])
    with pytest.raises(ValueError):
        _ = Spectrograph(waves, scale=bad_scale)
