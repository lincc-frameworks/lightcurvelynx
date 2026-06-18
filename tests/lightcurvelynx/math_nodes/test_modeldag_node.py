import numpy as np
import pytest
from lightcurvelynx.math_nodes.modeldag_node import ModelDAGNode


def test_model_dag_simple():
    """Test that we can generate numbers from a simple ModelDAGNode."""

    def _function1(val1, val2):
        return val1 + val2

    def _function2(value):
        return value**2

    # Define a simple modeldag.
    modeldag = pytest.importorskip("modeldag")
    model_dict = {
        # a, b, and c are generated from uniform distributions with different ranges.
        "a": {"func": np.random.uniform, "kwargs": {"low": 0, "high": 5}},
        "b": {"func": np.random.uniform, "kwargs": {"low": 0, "high": 10}},
        "c": {"func": np.random.uniform, "kwargs": {"low": -2, "high": 2}},
        # d is the sum of a and b.
        "d": {"func": _function1, "kwargs": {"val1": "@a", "val2": "@b"}},
        # e is the square of c.
        "e": {"func": _function2, "kwargs": {"value": "@c"}},
        # f is the sum of d and e.
        "f": {"func": _function1, "kwargs": {"val1": "@d", "val2": "@e"}},
    }
    model = modeldag.ModelDAG(model_dict)

    # We can create and sample from a ModelDAGNode with this model.
    node = ModelDAGNode(model, node_label="test")
    params = node.sample_parameters(num_samples=10, rng_info=np.random.default_rng(seed=100))
    assert params["test"]["a"].shape == (10,)
    assert params["test"]["b"].shape == (10,)
    assert params["test"]["c"].shape == (10,)
    assert np.allclose(params["test"]["d"], params["test"]["a"] + params["test"]["b"])
    assert np.allclose(params["test"]["e"], params["test"]["c"] ** 2)
    assert np.allclose(params["test"]["f"], params["test"]["d"] + params["test"]["e"])

    # We can create a ModelDAGNode using dictionary.
    node2 = ModelDAGNode(model_dict, node_label="test2")
    params2 = node2.sample_parameters(num_samples=10, rng_info=np.random.default_rng(seed=100))
    assert np.allclose(params["test"]["a"], params2["test2"]["a"])
    assert np.allclose(params["test"]["b"], params2["test2"]["b"])
    assert np.allclose(params["test"]["c"], params2["test2"]["c"])
    assert np.allclose(params["test"]["d"], params2["test2"]["d"])
    assert np.allclose(params["test"]["e"], params2["test2"]["e"])
    assert np.allclose(params["test"]["f"], params2["test2"]["f"])

    # If we pass a different random number generator, we should get different samples.
    params3 = node.sample_parameters(num_samples=10, rng_info=np.random.default_rng(seed=101))
    assert not np.allclose(params["test"]["a"], params3["test"]["a"])
    assert not np.allclose(params["test"]["b"], params3["test"]["b"])
    assert not np.allclose(params["test"]["c"], params3["test"]["c"])
    assert not np.allclose(params["test"]["d"], params3["test"]["d"])
    assert not np.allclose(params["test"]["e"], params3["test"]["e"])
    assert not np.allclose(params["test"]["f"], params3["test"]["f"])
