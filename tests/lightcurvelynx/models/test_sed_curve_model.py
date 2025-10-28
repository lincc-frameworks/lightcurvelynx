import numpy as np
import pytest
from lightcurvelynx.models.sed_curve_model import LightcurveSEDData, SEDCurveModel


def test_three_column_to_matrix() -> None:
    """Test that we can transform a 3-column SED data to a matrix form."""
    data = np.array(
        [
            [1.0, 1000.0, 10.0],
            [1.0, 2000.0, 20.0],
            [2.0, 1000.0, 15.0],
            [2.0, 2000.0, 25.0],
            [3.0, 1000.0, 20.0],
            [3.0, 2000.0, 30.0],
        ]
    )
    unique_times, unique_wavelengths, lightcurve_matrix = LightcurveSEDData.three_column_to_matrix(data)

    expected_wavelengths = np.array([1000.0, 2000.0])
    expected_times = np.array([1.0, 2.0, 3.0])
    expected_matrix = np.array(
        [
            [10.0, 20.0],
            [15.0, 25.0],
            [20.0, 30.0],
        ]
    )

    np.testing.assert_array_equal(unique_wavelengths, expected_wavelengths)
    np.testing.assert_array_equal(unique_times, expected_times)
    np.testing.assert_array_equal(lightcurve_matrix, expected_matrix)


def test_linear_lightcurve_sed_data() -> None:
    """Test that we can create a LightcurveSEDData object with linear interpolation."""
    data = np.array(
        [
            [1.0, 1000.0, 10.0],
            [1.0, 2000.0, 20.0],
            [2.0, 1000.0, 15.0],
            [2.0, 2000.0, 25.0],
            [3.0, 1000.0, 20.0],
            [3.0, 2000.0, 30.0],
        ]
    )
    data_obj = LightcurveSEDData(data, interpolation_type="linear", periodic=False)

    eval_times = np.array([-1.5, 1.5, 2.5, 3.5])
    eval_waves = np.array([1000.0, 2000.0])
    sed_values = data_obj.evaluate_sed(eval_times, eval_waves)
    expected_values = np.array(
        [
            [0.0, 0.0],  # 0.0 when not baseline provided
            [12.5, 22.5],
            [17.5, 27.5],
            [0.0, 0.0],  # 0.0 when not baseline provided
        ]
    )
    assert np.allclose(sed_values, expected_values)

    # We correct for lc_data_t0.
    data_obj2 = LightcurveSEDData(
        data,
        interpolation_type="linear",
        periodic=False,
        lc_data_t0=1.0,
    )
    sed_values = data_obj2.evaluate_sed(eval_times, eval_waves)
    expected_values = np.array(
        [
            [0.0, 0.0],  # 0.0 when not baseline provided
            [17.5, 27.5],
            [0.0, 0.0],  # 0.0 when not baseline provided
            [0.0, 0.0],  # 0.0 when not baseline provided
        ]
    )
    assert np.allclose(sed_values, expected_values)

    # Add a baseline and check that it is used outside the time range.
    baseline = np.array([1.0, 2.0])
    data_obj3 = LightcurveSEDData(
        data,
        interpolation_type="linear",
        periodic=False,
        baseline=baseline,
    )
    sed_values = data_obj3.evaluate_sed(eval_times, eval_waves)
    expected_values = np.array(
        [
            [1.0, 2.0],  # Baseline values when not in time range
            [12.5, 22.5],
            [17.5, 27.5],
            [1.0, 2.0],  # Baseline values when not in time range
        ]
    )
    assert np.allclose(sed_values, expected_values)

    # We fail if we use a incorrectly shaped baseline.
    with pytest.raises(ValueError):
        LightcurveSEDData(
            data,
            interpolation_type="linear",
            periodic=False,
            baseline=np.array([1.0, 2.0, 3.0]),
        )


def test_linear_lightcurve_sed_data_periodic() -> None:
    """Test that we can create periodic LightcurveSEDData object with linear interpolation."""
    data = np.array(
        [
            [1.0, 1000.0, 10.0],
            [1.0, 2000.0, 20.0],
            [2.0, 1000.0, 15.0],
            [2.0, 2000.0, 25.0],
            [3.0, 1000.0, 10.0],
            [3.0, 2000.0, 20.0],
        ]
    )
    data_obj = LightcurveSEDData(data, interpolation_type="linear", periodic=True)

    eval_times = np.array([0.5, 1.5, 2.25, 3.25])
    eval_waves = np.array([1000.0, 2000.0])
    sed_values = data_obj.evaluate_sed(eval_times, eval_waves)
    expected_values = np.array(
        [
            [12.5, 22.5],
            [12.5, 22.5],
            [11.25, 21.25],
            [13.75, 23.75],
        ]
    )
    assert np.allclose(sed_values, expected_values)


def test_create_sed_curve_model() -> None:
    """Test that we can create an SEDCurveModel object."""
    data = np.array(
        [
            [0.0, 1000.0, 5.0],
            [0.0, 2000.0, 15.0],
            [0.0, 3000.0, 5.0],
            [1.0, 1000.0, 10.0],
            [1.0, 2000.0, 20.0],
            [1.0, 3000.0, 5.0],
            [2.0, 1000.0, 15.0],
            [2.0, 2000.0, 25.0],
            [2.0, 3000.0, 5.0],
            [3.0, 1000.0, 10.0],
            [3.0, 2000.0, 15.0],
            [3.0, 3000.0, 5.0],
        ]
    )
    model = SEDCurveModel(data, 0.0, interpolation_type="linear", periodic=False, t0=0.0)
    assert len(model.times) == 4
    assert len(model.wavelengths) == 3

    # Evaluate the model at some times and wavelengths.
    eval_times = np.array([1.5, 2.5, 3.5])
    eval_waves = np.array([1000.0, 2000.0, 2500.0])

    state = model.sample_parameters(num_samples=1)
    sed_values = model.evaluate_sed(eval_times, eval_waves, graph_state=state)
    expected_values = np.array(
        [
            [12.5, 22.5, 13.75],
            [12.5, 20.0, 12.5],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(sed_values, expected_values)

    # Set a non-zero t0. An evaluation time of 2.0 now corresponds to phase 0.0 in the curve.
    model2 = SEDCurveModel(data, 0.0, interpolation_type="linear", periodic=False, t0=2.0)
    state2 = model2.sample_parameters(num_samples=1)
    sed_values2 = model2.evaluate_sed(eval_times, eval_waves, graph_state=state2)
    expected_values2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [7.5, 17.5, 11.25],
            [12.5, 22.5, 13.75],
        ]
    )
    assert np.allclose(sed_values2, expected_values2)
