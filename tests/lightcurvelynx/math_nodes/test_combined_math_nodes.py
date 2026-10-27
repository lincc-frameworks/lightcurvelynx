"""This file includes tests for combinations of math_nodes and settings that may not
be used in production, but help stress test the underlying machinery.
"""

import numpy as np

from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc


def test_math_with_numpy_random_inputs():
    """Test that math nodes work correctly with numpy random inputs."""
    a_sampler_node = NumpyRandomFunc("uniform", low=0.0, high=100.0, node_label="a")
    b_sampler_node = NumpyRandomFunc("uniform", low=100.0, high=200.0, node_label="b")
    basic_node = BasicMathNode("a + b", a=a_sampler_node, b=b_sampler_node, node_label="math")

    # Test basic sampling (no RNG given).
    state = basic_node.sample_parameters(num_samples=1_000)

    a_vals = state["math"]["a"]
    assert np.all((a_vals >= 0.0) & (a_vals <= 100.0))
    assert np.allclose(a_vals, state["a"]["function_node_result"])  # Output of a = input of math

    b_vals = state["math"]["b"]
    assert np.all((b_vals >= 100.0) & (b_vals <= 200.0))
    assert np.allclose(b_vals, state["b"]["function_node_result"])  # Output of b = input of math

    sum_vals = state["math"]["function_node_result"]
    assert np.all((sum_vals >= 100.0) & (sum_vals <= 300.0))
    assert np.allclose(sum_vals, a_vals + b_vals)

    # Check that if we pass a random number generator, it is passed to both sampler nodes.
    rng1 = np.random.default_rng(123456)
    state1 = basic_node.sample_parameters(num_samples=1_000, rng_info=rng1)
    rng2 = np.random.default_rng(123456)
    state2 = basic_node.sample_parameters(num_samples=1_000, rng_info=rng2)
    assert np.allclose(state1["math"]["function_node_result"], state2["math"]["function_node_result"])
    assert np.allclose(state1["a"]["function_node_result"], state2["a"]["function_node_result"])
    assert np.allclose(state1["b"]["function_node_result"], state2["b"]["function_node_result"])

    # The results should be different than what we got with the original RNG.
    assert not np.allclose(state1["math"]["function_node_result"], state["math"]["function_node_result"])
    assert not np.allclose(state1["a"]["function_node_result"], state["a"]["function_node_result"])
    assert not np.allclose(state1["b"]["function_node_result"], state["b"]["function_node_result"])

    # If we change the RNG, everything should change.
    rng3 = np.random.default_rng(654321)
    state3 = basic_node.sample_parameters(num_samples=1_000, rng_info=rng3)
    assert not np.allclose(state1["math"]["function_node_result"], state3["math"]["function_node_result"])
    assert not np.allclose(state1["a"]["function_node_result"], state3["a"]["function_node_result"])
    assert not np.allclose(state1["b"]["function_node_result"], state3["b"]["function_node_result"])
