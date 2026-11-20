import numpy as np
import pytest

from lightcurvelynx.math_nodes.caskade_node import (
    CaskadeFunctionNode,
    CaskadeParameterNode,
)


def test_caskade_parameter_node():
    """Test that we can use the CaskadeParameterNode to interface with Caskade nodes."""
    ck = pytest.importorskip("caskade")
    ck.backend.backend = "numpy"

    class _TestCaskadeModule(ck.Module):
        def __init__(self, name, x=None, y=None, z=None):
            super().__init__(name)
            self.x = ck.Param("x", x)
            self.y = ck.Param("y", y)
            self.z = ck.Param("z", z, shape=(3,))
            self.w = ck.Param("w", self.my_test_fn, dynamic=True)

        @ck.forward
        def my_test_fn(self, x, y):
            return x + y

        @ck.forward
        def my_test_fn2(self, x, y):
            return x * y

    # Instantiate the test module and the CaskadeParameterNode
    test_module = _TestCaskadeModule("test", x=1.0, y=2.0, z=np.array([1.0, 2.0, 3.0]))
    node = CaskadeParameterNode(test_module, node_label="test")

    # A single sample should match the initial values
    single_sample = node.sample_parameters(num_samples=1)
    assert single_sample["test"]["x"] == 1.0
    assert single_sample["test"]["y"] == 2.0
    assert single_sample["test"]["w"] == 3.0
    print(single_sample["test"]["z"])
    print(np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(single_sample["test"]["z"], np.array([1.0, 2.0, 3.0]))

    # Sample parameters and check their shapes and values.
    params = node.sample_parameters(num_samples=10)
    assert params["test"]["x"].shape == (10,)
    assert params["test"]["y"].shape == (10,)
    assert params["test"]["z"].shape == (10, 3)
    assert params["test"]["w"].shape == (10,)
    assert np.all(params["test"]["x"] == 1.0)
    assert np.all(params["test"]["y"] == 2.0)
    assert np.all(params["test"]["z"] == np.array([1.0, 2.0, 3.0]))
    assert np.all(params["test"]["w"] == 3.0)

    # If we change one value, it changes other values (specifically w).
    test_module.x = 5.0
    updated_sample = node.sample_parameters(num_samples=2)
    assert np.array_equal(updated_sample["test"]["x"], np.array([5.0, 5.0]))
    assert np.array_equal(updated_sample["test"]["w"], np.array([7.0, 7.0]))
    assert np.array_equal(updated_sample["test"]["y"], np.array([2.0, 2.0]))


def test_caskade_parameter_node_batched():
    """Test that we can use the CaskadeParameterNode to interface with batched input."""
    ck = pytest.importorskip("caskade")
    ck.backend.backend = "numpy"

    class _TestCaskadeModule(ck.Module):
        def __init__(self, name, x=None, y=None, z=None):
            super().__init__(name)
            self.x = ck.Param("x", x, batch_shape=(3,))
            self.y = ck.Param("y", y, batch_shape=(3,))
            self.z = ck.Param("z", z, shape=(3,), batch_shape=(3,))

    # Instantiate the batched test module and the CaskadeParameterNode
    z_vals = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    test_module = _TestCaskadeModule(
        "test",
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([4.0, 5.0, 6.0]),
        z=z_vals,
    )
    node = CaskadeParameterNode(test_module, node_label="test")

    # We can sample with num_samples < 3.
    sample = node.sample_parameters(num_samples=2)
    assert np.array_equal(sample["test"]["x"], np.array([1.0, 2.0]))
    assert np.array_equal(sample["test"]["y"], np.array([4.0, 5.0]))
    assert np.array_equal(sample["test"]["z"], z_vals[:2])

    # We can sample with num_samplers == 3. Note the sampler is NOT stateful.
    sample = node.sample_parameters(num_samples=3)
    assert np.array_equal(sample["test"]["x"], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(sample["test"]["y"], np.array([4.0, 5.0, 6.0]))
    assert np.array_equal(sample["test"]["z"], z_vals)

    # We can correctly handle sample offsets.
    sample = node.sample_parameters(num_samples=2, sample_offset=1)
    assert np.array_equal(sample["test"]["x"], np.array([2.0, 3.0]))
    assert np.array_equal(sample["test"]["y"], np.array([5.0, 6.0]))
    assert np.array_equal(sample["test"]["z"], z_vals[1:3])

    # We run into an error if our slice exceeds the batch size.
    with pytest.raises(ValueError):
        _ = node.sample_parameters(num_samples=4)

    with pytest.raises(ValueError):
        _ = node.sample_parameters(num_samples=1, sample_offset=3)


def test_caskade_function_node():
    """Test that we can access the functions of a Caskade module."""
    ck = pytest.importorskip("caskade")
    ck.backend.backend = "numpy"

    class _TestCaskadeModule(ck.Module):
        def __init__(self, name, x=None, y=None):
            super().__init__(name)
            self.x = ck.Param("x", x)
            self.y = ck.Param("y", y)

        @ck.forward
        def my_test_fn(self, x, y):
            return x + y

    # Create a _TestCaskadeModule and wrap its function directly in a FunctionNode.
    test_module = _TestCaskadeModule("test", x=1.0, y=2.0)
    node = CaskadeFunctionNode(test_module, "my_test_fn", node_label="test")

    sample = node.sample_parameters(num_samples=1)
    assert sample["test"]["my_test_fn"] == 3.0

    # If we change the values, the sampled result should reflect the new values.
    test_module.x = -2.0
    sample = node.sample_parameters(num_samples=1)
    assert sample["test"]["my_test_fn"] == 0.0
