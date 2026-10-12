import numpy as np

from lightcurvelynx.math_nodes.single_value_node import SingleVariableNode


def test_single_variable_node():
    """Test that we can create and query a SingleVariableNode."""
    node = SingleVariableNode("A", 10.0)
    assert str(node) == "SingleVariableNode"
    assert node.has_valid_param("A")

    state = node.sample_parameters()
    assert node.get_param(state, "A") == 10


def test_single_variable_node_none():
    """Test that we can create and query a SingleVariableNode with
    a constant value of None."""
    node = SingleVariableNode("A", None)
    assert not node.has_valid_param("A")

    state = node.sample_parameters()
    assert node.get_param(state, "A") is None


def test_single_variable_node_multi_dim():
    """Test that we can create and query a SingleVariableNode with
    a multi-dimensional constant value."""
    # Test with a list.
    node = SingleVariableNode("A", [0, 1, 2])
    state = node.sample_parameters(num_samples=10)
    samples = node.get_param(state, "A")
    assert len(samples) == 10
    for sample in samples:
        assert np.all(sample == [0, 1, 2])

    # Test with a numpy array.
    node = SingleVariableNode("A", np.array([0, 1, 2]))
    state = node.sample_parameters(num_samples=10)
    samples = node.get_param(state, "A")
    assert len(samples) == 10
    for sample in samples:
        assert np.array_equal(sample, np.array([0, 1, 2]))
