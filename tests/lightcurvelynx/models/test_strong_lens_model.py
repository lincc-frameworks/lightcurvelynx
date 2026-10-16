import numpy as np
from numpy.testing import assert_array_equal

from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.models.strong_lens_model import _LensImageExpansionNode

# Two systems with fixed width 4. System 0 has 3 active images, given out of arrival order
# and with a delay tie. System 1 has 2 active images, and its 0.0 padding is smaller than its
# active delays, so it must not be mistaken for the earliest image.
IMAGE_X = np.array([[0.5, -0.4, 0.1, np.nan], [1.0, -1.0, 0.0, 0.0]])
IMAGE_Y = np.array([[0.2, 0.3, -0.6, np.nan], [0.0, 0.5, 0.0, 0.0]])
MAGNIFICATIONS = np.array([[2.0, 4.0, 1.5, np.nan], [3.0, 0.5, 0.0, 0.0]])
TIME_DELAYS = np.array([[30.0, 10.0, 30.0, np.nan], [5.0, 12.0, 0.0, 0.0]])
NUM_IMAGES = np.array([3, 2])

# Expected per-image rows: arrival order (stable for ties), delays normalized to the earliest image.
EXPECTED_SYSTEM_ID = np.array([0, 0, 0, 1, 1])
EXPECTED_IMAGE_ID = np.array([0, 1, 2, 0, 1])
EXPECTED_IMAGE_X = np.array([-0.4, 0.5, 0.1, 1.0, -1.0])
EXPECTED_IMAGE_Y = np.array([0.3, 0.2, -0.6, 0.0, 0.5])
EXPECTED_MAGNIFICATION = np.array([4.0, 2.0, 1.5, 3.0, 0.5])
EXPECTED_TIME_DELAY = np.array([0.0, 20.0, 20.0, 0.0, 7.0])


def _given(values):
    """Return a stateless GivenValueList so repeated or batched sampling is reproducible."""
    return GivenValueList(values, stateful=False)


def _two_system_image_inputs():
    """Return the image-array setters for the two test systems."""
    return {
        "image_x": _given(IMAGE_X),
        "image_y": _given(IMAGE_Y),
        "macro_magnifications": _given(MAGNIFICATIONS),
        "time_delays": _given(TIME_DELAYS),
        "num_images": _given(NUM_IMAGES),
    }


def test_expansion_sorts_normalizes_and_drops_padding():
    """Test that each system expands into its active images in arrival order."""
    node = _LensImageExpansionNode(**_two_system_image_inputs(), node_label="images")
    state = node.sample_parameters(num_samples=2)

    assert state.num_samples == 5
    assert_array_equal(state["images"]["system_id"], EXPECTED_SYSTEM_ID)
    assert_array_equal(state["images"]["image_id"], EXPECTED_IMAGE_ID)
    assert_array_equal(state["images"]["image_x"], EXPECTED_IMAGE_X)
    assert_array_equal(state["images"]["image_y"], EXPECTED_IMAGE_Y)
    assert_array_equal(state["images"]["macro_magnification"], EXPECTED_MAGNIFICATION)
    assert_array_equal(state["images"]["time_delay"], EXPECTED_TIME_DELAY)


def test_expansion_repeats_per_system_values():
    """Test that values sampled before expansion are repeated once per image."""
    inputs = _two_system_image_inputs()
    state = _LensImageExpansionNode(**inputs, node_label="images").sample_parameters(num_samples=2)
    rows = state[inputs["image_x"].node_string]["function_node_result"]
    assert_array_equal(rows, IMAGE_X[EXPECTED_SYSTEM_ID])


def test_expansion_uses_full_width_without_num_images():
    """Test that all entries are active when num_images is not given."""
    node = _LensImageExpansionNode(
        image_x=[1.0, 2.0],
        image_y=[0.0, 0.0],
        macro_magnifications=[1.0, 2.0],
        time_delays=[3.0, 1.0],
        node_label="images",
    )
    state = node.sample_parameters(num_samples=1)
    assert state.num_samples == 2
    assert_array_equal(state["images"]["image_id"], [0, 1])
    assert_array_equal(state["images"]["image_x"], [2.0, 1.0])
    assert_array_equal(state["images"]["time_delay"], [0.0, 2.0])


def test_expansion_single_image_stores_scalars():
    """Test that a one-image system becomes a single row of scalar values."""
    node = _LensImageExpansionNode(
        image_x=[0.7, np.nan],
        image_y=[0.1, np.nan],
        macro_magnifications=[1.5, np.nan],
        time_delays=[4.0, np.nan],
        num_images=1,
        node_label="images",
    )
    state = node.sample_parameters(num_samples=1)
    assert state.num_samples == 1
    expected = {
        "system_id": 0,
        "image_id": 0,
        "image_x": 0.7,
        "image_y": 0.1,
        "macro_magnification": 1.5,
        "time_delay": 0.0,
    }
    for name, value in expected.items():
        assert np.isscalar(state["images"][name])
        assert state["images"][name] == value


def test_expansion_system_id_includes_sample_offset():
    """Test that system_id continues from the state's sample offset (batched simulation)."""
    node = _LensImageExpansionNode(
        image_x=[0.0, 1.0],
        image_y=[0.0, 1.0],
        macro_magnifications=[1.0, 1.0],
        time_delays=[0.0, 1.0],
        node_label="images",
    )
    state = node.sample_parameters(num_samples=2, sample_offset=10)
    assert_array_equal(state["images"]["system_id"], [10, 10, 11, 11])
    assert_array_equal(state["images"]["image_id"], [0, 1, 0, 1])


def test_expansion_accepts_integer_valued_float_counts():
    """Test that float-typed image counts such as 3.0 are used as counts."""
    inputs = _two_system_image_inputs() | {"num_images": _given(np.array([3.0, 2.0]))}
    state = _LensImageExpansionNode(**inputs, node_label="images").sample_parameters(num_samples=2)
    assert_array_equal(state["images"]["system_id"], EXPECTED_SYSTEM_ID)
    assert_array_equal(state["images"]["time_delay"], EXPECTED_TIME_DELAY)
