import numpy as np
import pytest
from citation_compass import find_in_citations
from lightcurvelynx.astro_utils.mag_flux import flux2mag
from lightcurvelynx.models.sed_template_model import (
    MultiSEDTemplateModel,
    SEDTemplate,
    SEDTemplateModel,
    SIMSEDModel,
)


def test_linear_sed_template_data() -> None:
    """Test that we can create a SEDTemplate object with linear interpolation."""
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
    data_obj = SEDTemplate(data, interpolation_type="linear", periodic=False)
    assert not data_obj.is_periodic
    assert data_obj.period is None

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

    # We correct for sed_data_t0.
    data_obj2 = SEDTemplate(
        data,
        interpolation_type="linear",
        periodic=False,
        sed_data_t0=1.0,
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
    data_obj3 = SEDTemplate(
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

    # We fail if we have the wrong number of columns.
    with pytest.raises(ValueError):
        SEDTemplate(
            np.array([[1.0, 1000.0], [2.0, 2000.0]]),
            interpolation_type="linear",
            periodic=False,
        )

    # We fail if we use a incorrectly shaped baseline.
    with pytest.raises(ValueError):
        SEDTemplate(
            data,
            interpolation_type="linear",
            periodic=False,
            baseline=np.array([1.0, 2.0, 3.0]),
        )


def test_linear_sed_template_data_periodic() -> None:
    """Test that we can create periodic SEDTemplate object with linear interpolation."""
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
    data_obj = SEDTemplate(data, interpolation_type="linear", periodic=True)
    assert data_obj.is_periodic
    assert data_obj.period == 2.0

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

    # We fail if we have a periodic model with only one time point or
    # if the first and last time points are not the same.
    with pytest.raises(ValueError):
        SEDTemplate(
            np.array([[1.0, 1000.0, 10.0], [1.0, 2000.0, 10.0]]),
            interpolation_type="linear",
            periodic=True,
        )
    with pytest.raises(ValueError):
        SEDTemplate(
            np.array([[1.0, 1000.0, 10.0], [2.0, 1000.0, 15.0]]),
            interpolation_type="linear",
            periodic=True,
        )


def test_sed_template_data_from_components() -> None:
    """Test that we can create a SEDTemplate object from separate time, wavelength, and flux arrays."""
    times = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([1000.0, 2000.0])
    fluxes = np.array(
        [
            [10.0, 20.0],
            [15.0, 25.0],
            [20.0, 30.0],
        ]
    )
    data_obj = SEDTemplate.from_components(
        times,
        wavelengths,
        fluxes,
        interpolation_type="linear",
        periodic=False,
    )

    eval_times = np.array([1.5, 2.5])
    eval_waves = np.array([1000.0, 2000.0])
    sed_values = data_obj.evaluate_sed(eval_times, eval_waves)
    expected_values = np.array(
        [
            [12.5, 22.5],
            [17.5, 27.5],
        ]
    )
    assert np.allclose(sed_values, expected_values)

    # We fail if the shapes of the input arrays are inconsistent.
    with pytest.raises(ValueError):
        SEDTemplate.from_components(
            np.array([1.0, 2.0]),
            wavelengths,
            fluxes,
            interpolation_type="linear",
            periodic=False,
        )


def test_sed_template_data_from_file(test_data_dir):
    """Test that we can create a SEDTemplate from a file."""
    filename = test_data_dir / "truncated-salt2-h17" / "salt2_template_0.dat"
    data = SEDTemplate.from_file(filename)
    assert len(data.times) == 26
    assert len(data.wavelengths) == 401

    with pytest.raises(FileNotFoundError):
        SEDTemplate.from_file(test_data_dir / "nonexistent-file.dat")


def test_create_sed_template_model() -> None:
    """Test that we can create an SEDTemplateModel object."""
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
    model = SEDTemplateModel(data, sed_data_t0=0.0, interpolation_type="linear", periodic=False, t0=0.0)
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

    # We fail is we provide the data in numpy array form without sed_data_t0.
    with pytest.raises(ValueError):
        _ = SEDTemplateModel(data, interpolation_type="linear", periodic=False, t0=0.0)

    # Set a non-zero t0. An evaluation time of 2.0 now corresponds to phase 0.0 in the curve.
    model2 = SEDTemplateModel(data, sed_data_t0=0.0, interpolation_type="linear", periodic=False, t0=2.0)
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


def test_create_sed_template_model_from_template() -> None:
    """Test that we can create an SEDTemplateModel from an SEDTemplate."""
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
    data_obj = SEDTemplate(data, interpolation_type="linear", periodic=True)
    model = SEDTemplateModel(data_obj, t0=0.0)
    assert len(model.times) == 3
    assert len(model.wavelengths) == 2
    assert model.template.is_periodic
    assert model.template.period == 2.0


def test_sed_model_data_from_file(test_data_dir):
    """Test that we can create a SEDTemplateModel from a file."""
    filename = test_data_dir / "truncated-salt2-h17" / "salt2_template_0.dat"
    model = SEDTemplateModel.from_file(filename, t0=0.0)
    assert len(model.template.times) == 26
    assert len(model.template.wavelengths) == 401

    # We fail if we don't have t0.
    with pytest.raises(ValueError):
        _ = SEDTemplateModel.from_file(filename)


def test_create_multi_sed_template_model() -> None:
    """Test that we can create a MultiSEDTemplateModel object."""
    data1 = np.array(
        [
            [0.0, 1000.0, 5.0],
            [0.0, 2000.0, 15.0],
            [1.0, 1000.0, 10.0],
            [1.0, 2000.0, 20.0],
        ]
    )
    lc1 = SEDTemplate(data1, interpolation_type="linear", periodic=False)

    data2 = np.array(
        [
            [0.0, 1000.0, 20.0],
            [0.0, 2000.0, 30.0],
            [1.0, 1000.0, 25.0],
            [1.0, 2000.0, 35.0],
        ]
    )
    lc2 = SEDTemplate(data2, interpolation_type="linear", periodic=False)
    model = MultiSEDTemplateModel([lc1, lc2], weights=[0.25, 0.75], t0=0.0, node_label="model")
    assert len(model) == 2

    # Evaluate the model at some times and wavelengths.
    eval_times = np.array([0.5])
    eval_waves = np.array([1000.0])

    state = model.sample_parameters(num_samples=10_000)
    sed_values = model.evaluate_sed(eval_times, eval_waves, graph_state=state)
    chose_first = state["model"]["selected_template"] == 0
    assert 0.15 < np.count_nonzero(chose_first) / 10000.0 < 0.35  # Check weights are roughly correct.
    assert np.allclose(sed_values[chose_first, 0, 0], 7.5)
    assert np.allclose(sed_values[~chose_first, 0, 0], 22.5)


def test_simsed_model_compute_sed(test_data_dir) -> None:
    """Test that we can load and compute SEDs from a SIMSEDModel."""
    model = SIMSEDModel.from_dir(test_data_dir / "fake_simsed", t0=0.0, distance=10.0, node_label="model")
    assert len(model) == 2
    assert model.flux_scale == 2.0

    # Compute the SLSN SED at a specific time and wavelength.
    times = np.array([0.0, 1.0, 2.0])
    wavelengths = np.array([4500.0, 5000.0])
    sed_values = model.evaluate_sed(times, wavelengths)
    assert sed_values.shape == (3, 2)
    assert np.all(sed_values > 0.0)

    # Confirm that we have noted the data in the citations registry.
    citations = find_in_citations("SIMSED Data")
    assert len(citations) == 1
    assert str(test_data_dir / "fake_simsed") in citations[0]

    # We fail if we try to load a SIMSEDModel without a valid distance.
    with pytest.raises(ValueError):
        SIMSEDModel.from_dir(test_data_dir / "fake_simsed", t0=0.0, distance=None, node_label="model")


def test_slsn_simsed_model(test_data_dir) -> None:
    """Test that we can compute the fluxes from a fake SLSN SIMSEDModel."""
    # Use fake data that sits around the peak of the SLSN lightcurve defined in
    # SIMSED.SLSN-I-MOSFIT/slsn0.dat.gz from https://zenodo.org/records/2612896
    # and use a similar flux_scale as given in SIMSED.SLSN-I-MOSFIT/SED.INFO to give
    # reasonable fluxes.
    data = np.array(
        [
            [0.0, 1000.0, 5.0e41],
            [0.0, 2000.0, 5.0e41],
            [1.0, 1000.0, 5.0e41],
            [1.0, 2000.0, 5.0e41],
            [2.0, 1000.0, 5.0e41],
            [3.0, 1000.0, 5.0e41],
            [2.0, 1000.0, 5.0e41],
            [3.0, 2000.0, 5.0e41],
        ]
    )
    template = SEDTemplate(data, interpolation_type="linear", periodic=False)
    model = SIMSEDModel([template], flux_scale=8.4e-41, t0=0.0, distance=10.0)
    assert len(model) == 1
    assert model.flux_scale == 8.4e-41

    # Compute the SLSN SED at a specific time and wavelength.
    times = np.array([0.5, 1.0])
    wavelengths = np.array([1200.0])
    sed_values = model.evaluate_sed(times, wavelengths)
    assert sed_values.shape == (2, 1)
    assert np.all(sed_values > 0.0)

    # We know that the expected magntiudes for a SLSN at 10 pc will be around -22.
    mag_vals = flux2mag(sed_values.flatten())
    assert np.all((mag_vals > -23.0) & (mag_vals < -21.0))
