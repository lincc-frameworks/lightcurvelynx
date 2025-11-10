import numpy as np
import pytest
from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.effects.basic_effects import ScaleFluxEffect
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.basic_models import ConstantSEDModel, StepModel
from lightcurvelynx.models.multi_object_model import AdditiveMultiObjectModel, RandomMultiObjectModel
from lightcurvelynx.models.static_sed_model import StaticBandfluxModel, StaticSEDModel


def test_additive_multi_object_node() -> None:
    """Test that we can create and evaluate a AdditiveMultiObjectModel."""
    object1 = ConstantSEDModel(brightness=10.0, node_label="my_static_object")
    object2 = StepModel(brightness=15.0, t0=1.0, t1=2.0, node_label="my_step_object")
    model = AdditiveMultiObjectModel([object1, object2], node_label="my_multi_object")

    state = model.sample_parameters()
    assert state["my_static_object"]["brightness"] == 10.0
    assert state["my_step_object"]["brightness"] == 15.0

    times = np.array([0.0, 1.5, 3.0])
    wavelengths = np.array([1000.0, 2000.0])

    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (3, 2)
    assert np.allclose(values, [[10.0, 10.0], [25.0, 25.0], [10.0, 10.0]])


def test_additive_multi_object_node_passband() -> None:
    """Test that we can evaluate a AdditiveMultiObjectModel at the passband level."""
    a_band = Passband(np.array([[900, 0.0], [1000, 0.5], [2000, 0.5], [2100, 0.0]]), "LSST", "a")
    b_band = Passband(np.array([[2900, 0.0], [3000, 0.5], [4000, 0.5], [4100, 0.0]]), "LSST", "b")
    c_band = Passband(np.array([[5900, 0.0], [6000, 0.5], [7000, 0.5], [7100, 0.0]]), "LSST", "c")
    pb_group = PassbandGroup([a_band, b_band, c_band])

    object1 = ConstantSEDModel(brightness=10.0, node_label="my_static_object")
    object2 = StepModel(brightness=15.0, t0=1.0, t1=2.0, node_label="my_step_object")
    model = AdditiveMultiObjectModel([object1, object2], node_label="my_multi_object")
    state = model.sample_parameters(num_samples=1)

    times = np.array([0.0, 0.5, 1.25, 1.5, 3.0, 4.0])
    filters = np.array(["a", "a", "a", "b", "c", "a"])

    bandflux1 = object1.evaluate_bandfluxes(pb_group, times, filters, state)
    assert np.allclose(bandflux1, [10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    bandflux2 = object2.evaluate_bandfluxes(pb_group, times, filters, state)
    assert np.allclose(bandflux2, [0.0, 0.0, 15.0, 15.0, 0.0, 0.0])

    bandflux_combined = model.evaluate_bandfluxes(pb_group, times, filters, state)
    assert np.allclose(bandflux_combined, [10.0, 10.0, 25.0, 25.0, 10.0, 10.0])

    # We can include a Bandflux only model in the computation.
    object3 = StaticBandfluxModel({"a": 1.0, "b": 2.0, "c": 0.0})
    model2 = AdditiveMultiObjectModel([object1, object2, object3], node_label="my_multi_object")
    state2 = model2.sample_parameters(num_samples=1)

    bandflux_combined = model2.evaluate_bandfluxes(pb_group, times, filters, state2)
    assert np.allclose(bandflux_combined, [11.0, 11.0, 26.0, 27.0, 10.0, 11.0])


def test_additive_multi_object_node_resample() -> None:
    """Test that we can correctly resample a AdditiveMultiObjectModel."""
    ra = NumpyRandomFunc("uniform", low=10.0, high=20.0)
    dec = NumpyRandomFunc("uniform", low=-10.0, high=10.0)

    # object1 and object2 share the same ra and dec for each sample, but have
    # different brightness values.
    object1 = ConstantSEDModel(
        brightness=NumpyRandomFunc("uniform", low=10.0, high=20.0),
        ra=ra,
        dec=dec,
        node_label="my_static_object",
    )
    object2 = StepModel(
        brightness=NumpyRandomFunc("uniform", low=30.0, high=40.0),
        ra=ra,
        dec=dec,
        t0=1.0,
        t1=2.0,
        node_label="my_step_object",
    )
    model = AdditiveMultiObjectModel([object1, object2], node_label="my_multi_object")

    num_samples = 1000
    rng_info = np.random.default_rng(100)
    state = model.sample_parameters(num_samples=num_samples, rng_info=rng_info)
    assert len(np.unique(state["my_static_object"]["ra"])) > num_samples / 2
    assert len(np.unique(state["my_static_object"]["dec"])) > num_samples / 2
    assert np.allclose(state["my_static_object"]["ra"], state["my_static_object"]["ra"])
    assert np.allclose(state["my_static_object"]["dec"], state["my_static_object"]["dec"])

    brightness_diff = state["my_step_object"]["brightness"] - state["my_static_object"]["brightness"]
    assert np.all(brightness_diff > 5.0)

    # We can evaluate the combined model at all of the times.
    times = np.array([0.0, 1.5, 3.0])
    wavelengths = np.array([1000.0, 2000.0])

    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (1000, 3, 2)

    assert np.allclose(values[:, 0, 0], state["my_static_object"]["brightness"])
    assert np.allclose(values[:, 0, 1], state["my_static_object"]["brightness"])
    assert np.allclose(
        values[:, 1, 0],
        state["my_static_object"]["brightness"] + state["my_step_object"]["brightness"],
    )
    assert np.allclose(
        values[:, 1, 1],
        state["my_static_object"]["brightness"] + state["my_step_object"]["brightness"],
    )
    assert np.allclose(values[:, 2, 0], state["my_static_object"]["brightness"])
    assert np.allclose(values[:, 2, 1], state["my_static_object"]["brightness"])


def test_additive_multi_object_node_redshift() -> None:
    """Test that we handle redshifts separately for each object."""
    object1 = StepModel(brightness=10.0, t0=1.0, t1=3.0, redshift=0.0, node_label="object1")
    object2 = StepModel(brightness=10.0, t0=2.0, t1=4.0, redshift=1.0, node_label="object2")
    model = AdditiveMultiObjectModel([object1, object2], node_label="my_multi_object")

    state = model.sample_parameters()
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    wavelengths = np.array([1000.0, 2000.0])

    # Compute the expected values:
    # The redshift applied to object1 is a no-op for both time and brightness.
    # The redshift applied to object2 shifts times to [1.25 1.75 2.25 2.75 3.25]
    # and scales up the brightness by a factor of (1 + z) = 2.0.
    contrib1 = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 10.0], [0.0, 0.0], [0.0, 0.0]])
    contrib2 = np.array([[0.0, 0.0], [0.0, 0.0], [20.0, 20.0], [20.0, 20.0], [20.0, 20.0]])

    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (5, 2)
    assert np.allclose(values, contrib1 + contrib2)


def test_additive_multi_object_node_effects_rest_frame() -> None:
    """Test that we handle rest frame effects separately for each object."""
    object1 = StepModel(brightness=10.0, t0=1.0, t1=3.0, node_label="object1")
    object2 = StepModel(brightness=10.0, t0=2.0, t1=4.0, node_label="object2")
    object1.add_effect(ScaleFluxEffect(flux_scale=0.5, rest_frame=True))
    object2.add_effect(ScaleFluxEffect(flux_scale=0.1, rest_frame=True))
    model = AdditiveMultiObjectModel([object1, object2], node_label="my_multi_object")

    # We added the effect to each submodel's rest frame list.
    assert len(object1.rest_frame_effects) == 1
    assert len(object2.rest_frame_effects) == 1
    assert len(object1.obs_frame_effects) == 0
    assert len(object2.obs_frame_effects) == 0
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 0

    state = model.sample_parameters()
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    wavelengths = np.array([1000.0, 2000.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (5, 2)

    contrib1 = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    contrib2 = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    assert np.allclose(values, contrib1 + contrib2)


def test_additive_multi_object_node_effects_rest_frame_add() -> None:
    """Test that we handle rest frame effects separately for each object."""
    object1 = StepModel(brightness=10.0, t0=1.0, t1=3.0, node_label="object1")
    object2 = StepModel(brightness=10.0, t0=2.0, t1=4.0, node_label="object2")
    model = AdditiveMultiObjectModel([object1, object2], node_label="my_multi_object")
    model.add_effect(ScaleFluxEffect(flux_scale=0.5, rest_frame=True))

    # We added the effect to each submodel's rest frame list.
    assert len(object1.rest_frame_effects) == 1
    assert len(object2.rest_frame_effects) == 1
    assert len(object1.obs_frame_effects) == 0
    assert len(object2.obs_frame_effects) == 0
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 0

    state = model.sample_parameters()
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    wavelengths = np.array([1000.0, 2000.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (5, 2)

    contrib1 = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    contrib2 = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0]])
    assert np.allclose(values, contrib1 + contrib2)


def test_additive_multi_object_node_effects_obs_frame() -> None:
    """Test that we handle observer frame effects together."""
    object1 = StepModel(brightness=10.0, t0=1.0, t1=3.0, node_label="object1")
    object2 = StepModel(brightness=10.0, t0=2.0, t1=4.0, node_label="object2")
    model = AdditiveMultiObjectModel([object1, object2], node_label="my_multi_object")
    model.add_effect(ScaleFluxEffect(flux_scale=0.5, rest_frame=False))

    # We added the effect to the joint model's rest frame list.
    assert len(object1.rest_frame_effects) == 0
    assert len(object2.rest_frame_effects) == 0
    assert len(object1.obs_frame_effects) == 0
    assert len(object2.obs_frame_effects) == 0
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 1

    state = model.sample_parameters()
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    wavelengths = np.array([1000.0, 2000.0])
    values = model.evaluate_sed(times, wavelengths, state)
    assert values.shape == (5, 2)

    contrib1 = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    contrib2 = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0]])
    assert np.allclose(values, contrib1 + contrib2)


def test_additive_multi_object_node_min_max() -> None:
    """Test that we can get the correct wavelength limits for a AdditiveMultiObjectModel."""
    sed0 = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    model0 = StaticSEDModel([sed0], node_label="sed0")

    sed1 = np.array(
        [
            [200.0, 300.0, 400.0, 500.0],  # Wavelengths
            [20.0, 40.0, 40.0, 20.0],  # fluxes
        ]
    )
    model1 = StaticSEDModel([sed1], node_label="sed1")

    # The reported min/max wavelengths are the overlap of the objects.
    model = AdditiveMultiObjectModel(
        [model0, model1],
        node_label="test",
    )
    states = model.sample_parameters(num_samples=1)
    assert np.array_equal(model.minwave(states), [100.0, 200.0])
    assert np.array_equal(model.maxwave(states), [400.0, 500.0])


def test_additive_multi_object_node_bandflux() -> None:
    """Test that we can create and evaluate an AdditiveMultiObjectModel with Bandflux models."""
    object1 = StaticBandfluxModel({"a": 1.0, "b": 2.0, "c": 0.0}, node_label="object1")
    object2 = StaticBandfluxModel({"a": 1.0, "b": 1.0, "c": 0.5}, node_label="object2")
    model = AdditiveMultiObjectModel([object1, object2], node_label="test")

    # When we evaluate the model, we should get the expected values.
    state = model.sample_parameters()
    times = np.array([0.0, 1.5, 3.0, 4.0, 5.0])
    filters = np.array(["a", "a", "b", "a", "c"])
    values = model.evaluate_bandfluxes(None, times, filters, state)
    assert np.array_equal(values, [2.0, 2.0, 3.0, 2.0, 0.5])

    # Check that we cannot turn on "apply_redshift" when using bandflux models.
    with pytest.raises(NotImplementedError):
        model.set_apply_redshift(True)

    # Check that we can add a rest frame effect. This are applied at the bandpass
    # level for Bandpass models.
    model.add_effect(ScaleFluxEffect(flux_scale=0.5, rest_frame=True))
    state = model.sample_parameters()
    values = model.evaluate_bandfluxes(None, times, filters, state)
    assert np.allclose(values, [1.0, 1.0, 1.5, 1.0, 0.25])

    # Check that we can apply a observer frame effect. The two effects stack.
    model.add_effect(ScaleFluxEffect(flux_scale=0.1, rest_frame=False))
    state = model.sample_parameters()
    values = model.evaluate_bandfluxes(None, times, filters, state)
    assert np.allclose(values, [0.1, 0.1, 0.15, 0.1, 0.025])

    # Check that we fail to evaluate the SEDs when we have a bandflux model.
    with pytest.raises(TypeError):
        _ = model.evaluate_sed(times, np.array([4000.0, 5000.0]), state)


def test_random_multi_object_node() -> None:
    """Test that we can create and evaluate a RandomMultiObjectModel."""
    object1 = ConstantSEDModel(brightness=10.0, node_label="object1")
    object2 = ConstantSEDModel(brightness=15.0, node_label="object2")
    model = RandomMultiObjectModel(
        [object1, object2],
        weights=[0.8, 0.2],
        node_label="my_multi_object",
    )

    state = model.sample_parameters(num_samples=10_000)
    assert np.all(state["object1"]["brightness"] == 10.0)
    assert np.all(state["object2"]["brightness"] == 15.0)

    # We should get approximately 80% of the samples from the first
    # object and 20% from the second object.
    object = np.array(state["my_multi_object"]["selected_object"], dtype=str)
    assert np.all((object == "object1") | (object == "object2"))
    assert np.sum(object == "object1") > 7000
    assert np.sum(object == "object2") > 1000

    # When we evaluate the model, we should get the expected values.
    times = np.array([0.0, 1.5, 3.0])
    wavelengths = np.array([1000.0, 2000.0])
    values = model.evaluate_sed(times, wavelengths, state)

    assert values.shape == (10_000, 3, 2)
    assert np.all((values == 10.0) | (values == 15.0))
    assert np.sum(values == 10.0) > 7000 * 5
    assert np.sum(values == 15.0) > 1000 * 5


def test_random_multi_object_node_bandflux() -> None:
    """Test that we can create and evaluate a RandomMultiObjectModel."""
    a_band = Passband(np.array([[900, 0.0], [1000, 0.5], [2000, 0.5], [2100, 0.0]]), "LSST", "a")
    b_band = Passband(np.array([[2900, 0.0], [3000, 0.5], [4000, 0.5], [4100, 0.0]]), "LSST", "b")
    c_band = Passband(np.array([[5900, 0.0], [6000, 0.5], [7000, 0.5], [7100, 0.0]]), "LSST", "c")
    pb_group = PassbandGroup([a_band, b_band, c_band])

    # Create a random model with 2 SED-based models and 1-bandflux based model.
    object1 = ConstantSEDModel(brightness=10.0, node_label="object1")
    object2 = ConstantSEDModel(brightness=15.0, node_label="object2")
    object3 = StaticBandfluxModel({"a": 1.0, "b": 2.0, "c": 0.0}, node_label="object3")
    model = RandomMultiObjectModel(
        [object1, object2, object3],
        weights=[0.25, 0.5, 0.25],
        node_label="my_multi_object",
    )

    rng_info = np.random.default_rng(100)
    state = model.sample_parameters(num_samples=1_000, rng_info=rng_info)

    model_names = np.array(state["my_multi_object"]["selected_object"], dtype=str)
    assert np.all((model_names == "object1") | (model_names == "object2") | (model_names == "object3"))
    assert np.sum(model_names == "object1") > 200
    assert np.sum(model_names == "object2") > 400
    assert np.sum(model_names == "object3") > 200

    # When we evaluate the model, we should get the expected values.
    times = np.array([0.0, 1.5, 3.0, 4.0])
    filters = np.array(["a", "a", "b", "a"])
    values = model.evaluate_bandfluxes(pb_group, times, filters, state)

    assert values.shape == (1_000, 4)
    for i in range(1_000):
        if model_names[i] == "object1":
            assert np.allclose([10.0, 10.0, 10.0, 10.0], values[i])
        elif model_names[i] == "object2":
            assert np.allclose([15.0, 15.0, 15.0, 15.0], values[i])
        elif model_names[i] == "object3":
            assert np.allclose([1.0, 1.0, 2.0, 1.0], values[i])

    # Check that we fail if we try to evaluate the SEDs.
    with pytest.raises(TypeError):
        _ = model.evaluate_sed(times, np.array([1000.0, 2000.0]), state)


def test_random_multi_object_node_min_max() -> None:
    """Test that we can get the correct wavelength limits for a RandomMultiObjectModel."""
    sed0 = np.array(
        [
            [100.0, 200.0, 300.0, 400.0],  # Wavelengths
            [10.0, 20.0, 20.0, 10.0],  # fluxes
        ]
    )
    model0 = StaticSEDModel([sed0], node_label="sed0")

    sed1 = np.array(
        [
            [200.0, 300.0, 400.0, 500.0],  # Wavelengths
            [20.0, 40.0, 40.0, 20.0],  # fluxes
        ]
    )
    model1 = StaticSEDModel([sed1], node_label="sed1")

    model = RandomMultiObjectModel(
        [model0, model1],
        weights=[0.5, 0.5],
        node_label="test",
    )
    states = model.sample_parameters(num_samples=1)

    # Force the selected_object to be 0 for the test.
    states.set("test", "selected_object", 0)
    assert model.minwave(states) == 100.0
    assert model.maxwave(states) == 400.0

    # Force the selected_object to be 1 for the test.
    states.set("test", "selected_object", 1)
    assert model.minwave(states) == 200.0
    assert model.maxwave(states) == 500.0

    # We fail if we do not pass in the states.
    with pytest.raises(ValueError):
        _ = model.minwave()
    with pytest.raises(ValueError):
        _ = model.maxwave()
