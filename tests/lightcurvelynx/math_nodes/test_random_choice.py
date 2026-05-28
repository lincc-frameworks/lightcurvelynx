import numpy as np
import pytest
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.random_choice import RandomChoiceNode


def test_random_choice_node():
    """Test that we can retrieve numbers from a RandomChoiceNode."""
    random_node = RandomChoiceNode([1, 3, 5, 7], node_label="random_node")

    # Check that we have sampled uniformly from the given options.
    state = random_node.sample_parameters(num_samples=5_000)

    # Check that the sampled indices match the expected distributions.
    indices = state["random_node"]["selected_index"]
    assert len(indices) == 5_000
    assert np.all((indices >= 0) & (indices < 4))
    assert len(indices[indices == 0]) > 1000
    assert len(indices[indices == 1]) > 1000
    assert len(indices[indices == 2]) > 1000
    assert len(indices[indices == 3]) > 1000

    # Check that we sampled the values match the expected distributions.
    values = state["random_node"]["function_node_result"]
    assert len(values) == 5_000
    assert np.all((values == 1) | (values == 3) | (values == 5) | (values == 7))
    assert np.all(values[indices == 0] == 1)
    assert np.all(values[indices == 1] == 3)
    assert np.all(values[indices == 2] == 5)
    assert np.all(values[indices == 3] == 7)

    # We fail if we try to create a RandomChoiceNode with an empty list.
    with pytest.raises(ValueError):
        _ = RandomChoiceNode([])

    # We fail if the number of weights doesn't match the number of values.
    with pytest.raises(ValueError):
        _ = RandomChoiceNode([1, 3, 5], weights=[0.5, 0.5])


def test_random_choice_node_chained():
    """Test that we can chain a RandomChoiceNode with another node."""
    input1 = NumpyRandomFunc("uniform", low=0, high=1, node_label="input1")
    input2 = NumpyRandomFunc("uniform", low=9, high=10, node_label="input2")
    choice_node = RandomChoiceNode([input1, input2], node_label="choice_node")

    # Check that we have sampled (approximately) uniformly from the given options.
    rng = np.random.default_rng(seed=1000)
    state = choice_node.sample_parameters(num_samples=5_000, rng_info=rng)

    # Check that the sampled indices match the expected distributions.
    indices = state["choice_node"]["selected_index"]
    assert len(indices) == 5_000
    assert np.all((indices == 0) | (indices == 1))
    assert len(indices[indices == 0]) > 2000
    assert len(indices[indices == 1]) > 2000

    values = state["choice_node"]["function_node_result"]
    assert np.all(values[indices == 0] < 1)
    assert np.all(values[indices == 1] >= 9)


def test_random_choice_node_weighted():
    """Test that we can weight the choices in a RandomChoiceNode."""
    input1 = NumpyRandomFunc("uniform", low=0, high=1, node_label="input1")
    input2 = NumpyRandomFunc("uniform", low=9, high=10, node_label="input2")
    input3 = NumpyRandomFunc("uniform", low=20, high=30, node_label="input3")
    choice_node = RandomChoiceNode(
        [input1, input2, input3],
        weights=[0.1, 0.8, 0.1],
        node_label="choice_node",
    )

    # Check that we have sampled (approximately) uniformly from the given options.
    rng = np.random.default_rng(seed=1001)
    state = choice_node.sample_parameters(num_samples=10_000, rng_info=rng)
    results = state["choice_node"]["function_node_result"]
    assert len(results) == 10_000
    assert len(results[results < 1]) > 500
    assert len(results[(results >= 9) & (results < 10)]) > 7000
    assert len(results[(results >= 20) & (results < 30)]) > 500


def test_random_choice_node_seeded():
    """Test that we can set the seed for a RandomChoiceNode."""
    random_node1 = RandomChoiceNode([1, 3, 5, 7], seed=1234, node_label="random_node1")
    state1 = random_node1.sample_parameters(num_samples=5_000)
    samples1 = state1["random_node1"]["function_node_result"]

    random_node2 = RandomChoiceNode([1, 3, 5, 7], seed=1234, node_label="random_node2")
    state2 = random_node2.sample_parameters(num_samples=5_000)
    samples2 = state2["random_node2"]["function_node_result"]

    assert np.allclose(samples1, samples2)

    random_node3 = RandomChoiceNode([1, 3, 5, 7], seed=5678, node_label="random_node3")
    state3 = random_node3.sample_parameters(num_samples=5_000)
    samples3 = state3["random_node3"]["function_node_result"]

    assert not np.allclose(samples1, samples3)


def test_random_choice_node_single_sample():
    """Test RandomChoiceNode when requesting a single sample."""
    random_node = RandomChoiceNode([10, 20, 30], seed=123, node_label="random_node")
    state = random_node.sample_parameters(num_samples=1)

    selected_index = state["random_node"]["selected_index"]
    result = state["random_node"]["function_node_result"]

    assert selected_index in [0, 1, 2]
    assert result in [10, 20, 30]
    assert result == [10, 20, 30][selected_index]


def test_random_choice_node_one_value():
    """Test that a single-value RandomChoiceNode always returns that value."""
    random_node = RandomChoiceNode([42], node_label="random_node")
    state = random_node.sample_parameters(num_samples=1_000)

    selected_indices = state["random_node"]["selected_index"]
    results = state["random_node"]["function_node_result"]

    assert np.all(selected_indices == 0)
    assert np.all(results == 42)


@pytest.mark.parametrize("bad_weights", [[0.0, 0.0, 0.0], [-1.0, 1.0, 1.0], [np.nan, 1.0, 1.0]])
def test_random_choice_node_invalid_weights(bad_weights):
    """Test failure for invalid choice probabilities."""
    with pytest.raises(ValueError):
        _ = RandomChoiceNode([1, 2, 3], weights=bad_weights)


def test_random_choice_node_uses_external_rng():
    """Test that an external RNG controls sampling regardless of node seed."""
    node1 = RandomChoiceNode([1, 3, 5, 7], seed=111, node_label="node1")
    node2 = RandomChoiceNode([1, 3, 5, 7], seed=999, node_label="node2")

    rng1 = np.random.default_rng(seed=2024)
    rng2 = np.random.default_rng(seed=2024)
    state1 = node1.sample_parameters(num_samples=500, rng_info=rng1)
    state2 = node2.sample_parameters(num_samples=500, rng_info=rng2)

    samples1 = state1["node1"]["function_node_result"]
    samples2 = state2["node2"]["function_node_result"]
    inds1 = state1["node1"]["selected_index"]
    inds2 = state2["node2"]["selected_index"]

    assert np.array_equal(inds1, inds2)
    assert np.array_equal(samples1, samples2)
