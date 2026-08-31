import numpy as np
import pytest
from astropy import units as u
from lightcurvelynx.astro_utils.spectrograph import Spectrograph, gaussian_integral
from lightcurvelynx.astro_utils.unit_utils import fnu_to_flam


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

    # One dimensional fluxes to spec_fluxes
    values_1d = np.random.random(len(spgraph))
    spec_fluxes_1d = spgraph.evaluate(values_1d)
    expected_density_per_bin_1d = fnu_to_flam(
        values_1d,
        spgraph.bin_centers,
        wave_unit=u.AA,
        flam_unit=u.erg / u.s / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    expected_flux_1d = expected_density_per_bin_1d * spgraph.bin_widths
    assert spec_fluxes_1d.shape == (len(spgraph),)
    assert np.allclose(spec_fluxes_1d, expected_flux_1d)

    # Two dimensional fluxes to spec_fluxes
    values = np.random.random((10, len(spgraph)))
    spec_fluxes = spgraph.evaluate(values)
    expected_density_per_bin = fnu_to_flam(
        values,
        spgraph.bin_centers,
        wave_unit=u.AA,
        flam_unit=u.erg / u.s / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    expected_flux = expected_density_per_bin * spgraph.bin_widths[np.newaxis, :]
    assert spec_fluxes.shape == (10, len(spgraph))
    assert np.allclose(spec_fluxes, expected_flux)

    # Three dimensional fluxes to spec_fluxes
    values_3d = np.random.random((4, 10, len(spgraph)))
    spec_fluxes_3d = spgraph.evaluate(values_3d)
    expected_density_per_bin_3d = fnu_to_flam(
        values_3d,
        spgraph.bin_centers,
        wave_unit=u.AA,
        flam_unit=u.erg / u.s / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    expected_flux_3d = expected_density_per_bin_3d * spgraph.bin_widths[np.newaxis, np.newaxis, :]
    assert spec_fluxes_3d.shape == (4, 10, len(spgraph))
    assert np.allclose(spec_fluxes_3d, expected_flux_3d)

    # We fail evaluation with incorrect input shape, empty arrays, or mismatched dimensions
    with pytest.raises(ValueError):
        spgraph.evaluate(np.random.random((10, len(spgraph) + 1)))
    with pytest.raises(ValueError):
        spgraph.evaluate(np.random.random((4, 10, len(spgraph) + 1)))
    with pytest.raises(ValueError):
        spgraph.evaluate(np.random.random((0, len(spgraph))))
    with pytest.raises(ValueError):
        spgraph.evaluate(np.random.random((4, 0, len(spgraph))))

    # Test that we fail to create a Spectrograph object with invalid parameters.
    with pytest.raises(ValueError):
        _ = Spectrograph.from_regular_grid(wave_start=5000, wave_end=4000, bin_width=5.0)
    with pytest.raises(ValueError):
        _ = Spectrograph.from_regular_grid(wave_start=4000, wave_end=5000, bin_width=-5.0)


def test_spectrograph_evaluate_converts_njy_to_flam():
    """Test that nJy flux densities are converted to F_lambda before integrating."""
    spgraph = Spectrograph.from_regular_grid(wave_start=4000.0, wave_end=5000.0, bin_width=200.0)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    expected = (
        fnu_to_flam(
            values,
            spgraph.query_waves,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph._query_widths
    )
    results = spgraph.evaluate(values)

    assert np.allclose(results, expected)


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

    # One dimensional fluxes to spec_fluxes
    values_1d = np.random.random(len(spgraph))
    spec_fluxes_1d = spgraph.evaluate(values_1d)
    expected_1d = (
        fnu_to_flam(
            values_1d,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph.bin_widths
    )
    assert spec_fluxes_1d.shape == (len(spgraph),)
    assert np.allclose(spec_fluxes_1d, expected_1d)

    # Two dimensional fluxes to spec_fluxes
    values = np.random.random((10, len(spgraph)))
    spec_fluxes = spgraph.evaluate(values)
    expected = (
        fnu_to_flam(
            values,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph.bin_widths[np.newaxis, :]
    )
    assert np.allclose(spec_fluxes, expected)

    # Three dimensional fluxes to spec_fluxes
    values_3d = np.random.random((4, 10, len(spgraph)))
    spec_fluxes_3d = spgraph.evaluate(values_3d)
    expected_3d = (
        fnu_to_flam(
            values_3d,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph.bin_widths[np.newaxis, np.newaxis, :]
    )
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
    with pytest.raises(ValueError):
        # A non-positive max_wave_step.
        _ = Spectrograph(
            [1000.0, 2000.0, 3000.0],
            [2000.0, 3000.0, 4000.0],
            max_wave_step=0.0,
        )


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
    spgraph6 = Spectrograph.from_regular_grid(
        wave_start=4000, wave_end=8000, bin_width=5.0,
        wavelength_resolution=np.full(spgraph1.num_bins, 10.0),
    )

    assert spgraph1 == spgraph2
    assert spgraph1 != spgraph3
    assert spgraph1 != spgraph4
    assert spgraph1 != spgraph5
    assert spgraph1 != spgraph6


def test_create_spectrograph_with_scale():
    """Test that we can create and query a Spectrograph object with a scale."""
    scale = np.array([0.5, 1.0, 1.0, 1.0, 0.8])
    spgraph = Spectrograph.from_regular_grid(wave_start=4000, wave_end=5000, bin_width=200.0, scale=scale)
    assert np.allclose(spgraph.query_waves, np.array([4100.0, 4300.0, 4500.0, 4700.0, 4900.0]))
    assert spgraph.num_bins == 5

    # One dimensional fluxes to spec_fluxes
    measurement = np.array([50.0, 40.0, 20.0, 20.0, 10.0])
    expected_fnu = np.array([25.0, 40.0, 20.0, 20.0, 8.0])
    expected = (
        fnu_to_flam(
            expected_fnu,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph.bin_widths
    )
    results = spgraph.evaluate(measurement)
    assert np.allclose(results, expected)

    # Two dimensional fluxes to spec_fluxes
    measurement = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5.0, 15.0, 25.0, 35.0, 45.0],
        ]
    )
    expected_fnu = np.array(
        [
            [5.0, 20.0, 30.0, 40.0, 40.0],
            [2.5, 15.0, 25.0, 35.0, 36.0],
        ]
    )
    expected = (
        fnu_to_flam(
            expected_fnu,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
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
    expected_fnu = np.array(
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
    expected = (
        fnu_to_flam(
            expected_fnu,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
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


def test_create_spectrograph_with_float_scale():
    """Test that we can create and query a Spectrograph object with a scalar float scale."""
    spgraph = Spectrograph.from_regular_grid(wave_start=4000, wave_end=5000, bin_width=200.0, scale=0.5)
    assert np.allclose(spgraph.query_waves, np.array([4100.0, 4300.0, 4500.0, 4700.0, 4900.0]))
    assert spgraph.num_bins == 5

    # Make sure the scale is applied correctly to the fluxes.
    measurement = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [5.0, 15.0, 25.0, 35.0, 45.0],
        ]
    )
    expected_fnu = np.array(
        [
            [5.0, 10.0, 15.0, 20.0, 25.0],
            [2.5, 7.5, 12.5, 17.5, 22.5],
        ]
    )
    expected = (
        fnu_to_flam(
            expected_fnu,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph.bin_widths[np.newaxis, :]
    )
    results = spgraph.evaluate(measurement)
    assert np.allclose(results, expected)


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
        3525.0,  # Bin 0 - Sample 0
        3575.0,  # Bin 0 - Sample 1
        3625.0,  # Bin 1 - Sample 0
        3675.0,  # Bin 2 - Sample 0
        3725.0,  # Bin 3 - Sample 0
        3775.0,  # Bin 3 - Sample 1
        3825.0,  # Bin 4 - Sample 0
        3875.0,  # Bin 4 - Sample 1
        3925.0,  # Bin 4 - Sample 2
        3975.0,  # Bin 4 - Sample 3
    ]
    assert np.allclose(spgraph.query_waves, sample_waves, atol=0.2)

    # Test one dimensional fluxes to spec_fluxes
    measurement1 = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    expected1_fnu = np.array([5.0, 20.0, 30.0, 45.0, 75.0])
    expected1 = (
        fnu_to_flam(
            expected1_fnu,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph.bin_widths
    )
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
    expected2_fnu = np.array(
        [
            [5.0, 20.0, 30.0, 45.0, 75.0],
            [10.0, 25.0, 35.0, 50.0, 80.0],
        ]
    )
    expected2 = (
        fnu_to_flam(
            expected2_fnu,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph.bin_widths[np.newaxis, :]
    )
    results2 = spgraph.evaluate(measurement2)
    assert np.allclose(results2, expected2)
    assert results2.shape == (2, spgraph.num_bins)

    # Test three dimensional fluxes to bandfluxes
    measurement3 = np.array([measurement2, measurement2 + 2.0])
    expected3_fnu = np.array(
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
    expected3 = (
        fnu_to_flam(
            expected3_fnu,
            spgraph.bin_centers,
            wave_unit=u.AA,
            flam_unit=u.erg / u.s / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        * spgraph.bin_widths[np.newaxis, np.newaxis, :]
    )
    results3 = spgraph.evaluate(measurement3)
    assert np.allclose(results3, expected3)
    assert results3.shape == (2, 2, spgraph.num_bins)

    # check padding introduced by wavelength_resolution.
    resolution = np.full(5, 30.0)
    spgraph_padded = Spectrograph(
        wave_min, wave_max, wavelength_resolution=resolution
    )
    assert spgraph_padded.num_padded_bins > spgraph_padded.num_bins
    assert len(spgraph_padded.query_waves) == spgraph_padded.num_padded_bins
    assert np.allclose(
        spgraph_padded.query_waves, (spgraph_padded.padded_min + spgraph_padded.padded_max) / 2
    )
    result = spgraph_padded.evaluate(np.random.random(spgraph_padded.query_waves.shape))
    assert result.shape == (spgraph_padded.num_bins,)

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


def test_create_spectrograph_max_wave_step_aggregates_subbins():
    """Test that split bins are correctly aggregated back into original spectrograph bins."""
    wave_min = np.array([3500.0, 3600.0, 3650.0, 3700.0, 3800.0])
    wave_max = np.array([3600.0, 3650.0, 3700.0, 3800.0, 4000.0])
    spgraph = Spectrograph(wave_min, wave_max, max_wave_step=50.0)

    # Each bin is split into multiple sub-bins; we make the per-sample fluxes
    # distinct so the aggregation is easy to validate.
    values = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    result = spgraph.evaluate(values)

    # Expectation: each original spectrograph bin gets the sum of the converted
    # sub-bin contributions within it.
    expected = np.array(
        [
            fnu_to_flam(
                5.0,
                spgraph.bin_centers[0],
                wave_unit=u.AA,
                flam_unit=u.erg / u.s / u.cm**2 / u.AA,
                fnu_unit=u.nJy,
            )
            * spgraph.bin_widths[0],
            fnu_to_flam(
                20.0,
                spgraph.bin_centers[1],
                wave_unit=u.AA,
                flam_unit=u.erg / u.s / u.cm**2 / u.AA,
                fnu_unit=u.nJy,
            )
            * spgraph.bin_widths[1],
            fnu_to_flam(
                30.0,
                spgraph.bin_centers[2],
                wave_unit=u.AA,
                flam_unit=u.erg / u.s / u.cm**2 / u.AA,
                fnu_unit=u.nJy,
            )
            * spgraph.bin_widths[2],
            fnu_to_flam(
                45.0,
                spgraph.bin_centers[3],
                wave_unit=u.AA,
                flam_unit=u.erg / u.s / u.cm**2 / u.AA,
                fnu_unit=u.nJy,
            )
            * spgraph.bin_widths[3],
            fnu_to_flam(
                75.0,
                spgraph.bin_centers[4],
                wave_unit=u.AA,
                flam_unit=u.erg / u.s / u.cm**2 / u.AA,
                fnu_unit=u.nJy,
            )
            * spgraph.bin_widths[4],
        ]
    )
    assert np.allclose(result, expected)

def test_create_spectrograph_with_wavelength_resolution():
    """Test that a Spectrograph with per-bin wavelength resolution pads its bins
    on both sides to support smearing, and that no resolution means no padding."""
    waves_min = np.array([4000.0, 4020.0, 4040.0])
    waves_max = np.array([4020.0, 4040.0, 4060.0])
    resolution = np.full(3, 10.0)
    spgraph = Spectrograph(waves_min, waves_max, wavelength_resolution=resolution)

    assert spgraph.num_padded_bins > spgraph.num_bins
    assert np.sum(spgraph.is_padding) == spgraph.num_padded_bins - spgraph.num_bins
    assert len(spgraph.padded_min) == spgraph.num_padded_bins
    assert len(spgraph.padded_max) == spgraph.num_padded_bins
    assert np.allclose(spgraph.waves_min, waves_min)
    assert np.allclose(spgraph.waves_max, waves_max)

    # With no resolution given, there is no padding at all.
    spgraph_no_res = Spectrograph(waves_min, waves_max)
    assert spgraph_no_res.num_padded_bins == spgraph_no_res.num_bins
    assert not np.any(spgraph_no_res.is_padding)


def test_spectrograph_smear_matrix_values():
    """Test that the smear matrix redistributes flux using the expected Gaussian
    fractions for a simple, hand-computable case."""
    waves_min = np.array([4000.0, 4020.0, 4040.0])
    waves_max = np.array([4020.0, 4040.0, 4060.0])
    resolution = np.full(3, 10.0)  # sigma = bin_width / 2
    spgraph = Spectrograph(waves_min, waves_max, wavelength_resolution=resolution, compute_smear=True)

    # Index of the middle observed bin (center 4030) within the padded array.
    middle = int(np.argmax(~spgraph.is_padding)) + 1

    within_1_sigma = gaussian_integral(-1, 1)
    between_1_and_3_sigma = gaussian_integral(1, 3)
    assert spgraph.smear_matrix[middle, middle] == pytest.approx(within_1_sigma)
    assert spgraph.smear_matrix[middle, middle - 1] == pytest.approx(between_1_and_3_sigma)
    assert spgraph.smear_matrix[middle, middle + 1] == pytest.approx(between_1_and_3_sigma)

    # Padding bins can never receive smeared flux.
    assert np.all(spgraph.smear_matrix[:, spgraph.is_padding] == 0.0)

    # With zero resolution the smear matrix is the identity (no cross-bin mixing).
    zero_resolution = np.full(3, 0.0)
    spgraph_no_res = Spectrograph(waves_min, waves_max, wavelength_resolution=zero_resolution, compute_smear=True)
    assert np.allclose(spgraph_no_res.smear_matrix, np.eye(spgraph_no_res.num_bins))


def test_spectrograph_evaluate_with_smearing():
    """Test that evaluate() smears flux across bins and that smear=False bypasses it."""
    waves_min = np.arange(4000.0, 4400.0, 20.0)
    waves_max = np.arange(4020.0, 4420.0, 20.0)
    resolution = np.full(len(waves_min), 30.0)
    spgraph = Spectrograph(waves_min, waves_max, wavelength_resolution=resolution, compute_smear=True)
    spgraph_unsmeared = Spectrograph(waves_min, waves_max, wavelength_resolution=resolution)

    # A flat flux density should stay (approximately) flat after smearing, since
    # padding supplies the flux that would otherwise be lost at the edges.
    flux_density = 3.7
    flat_input = np.full(spgraph.query_waves.shape, flux_density)
    unsmeared = spgraph_unsmeared.evaluate(flat_input)
    smeared = spgraph.evaluate(flat_input)
    assert smeared.shape == (spgraph.num_bins,)
    assert np.allclose(smeared, unsmeared, rtol=5e-3)

    # A single narrow spike of flux should spread into neighboring bins when smeared,
    # while approximately conserving total flux.
    spike_input = np.zeros_like(spgraph.query_waves)
    spike_input[len(spike_input) // 2] = 100.0
    spiked = spgraph.evaluate(spike_input)
    not_spiked = spgraph_unsmeared.evaluate(spike_input)
    assert np.sum(spiked > 0) > np.sum(not_spiked > 0)
    assert np.sum(spiked) == pytest.approx(np.sum(not_spiked), rel=0.05)
