import math

import jax
import numpy as np
import pytest
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.single_value_node import SingleVariableNode


def test_basic_math_node():
    """Test that we can perform computations via a BasicMathNode."""
    node_a = SingleVariableNode("a", 10.0)
    node_b = SingleVariableNode("b", -5.0)
    node = BasicMathNode("a + b", a=node_a.a, b=node_b.b, node_label="test", backend="math")
    assert node.eval(a=10.0, b=-5.0) == pytest.approx(5.0)

    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Try with a math function.
    node_c = SingleVariableNode("c", 1000.0)
    node = BasicMathNode("a + b - log10(c)", a=10.0, b=5.0, c=node_c.c, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 12.0

    # Try with a second math function.
    node = BasicMathNode(
        "sqrt(a) + b - log10(c)", a=16.0, b=4.0, c=node_c.c, node_label="test", backend="math"
    )
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Test that we can reproduce the power function.
    node_d = SingleVariableNode("d", 5.0)
    node = BasicMathNode("a ** b", a=node_d.d, b=2.5, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == pytest.approx(math.pow(5.0, 2.5))

    # We fail with an invalid backend.
    with pytest.raises(ValueError):
        _ = BasicMathNode("a + b", a=10.0, b=-5.0, node_label="test", backend="invalid")


def test_basic_math_node_list_functions():
    """Test that we can list the functions available for a BasicMathNode."""
    funcs = BasicMathNode.list_functions()
    assert len(funcs) > 20


def test_basic_math_node_special_cases():
    """Test that we can handle some of the special cases for a BasicMathNode."""
    node_a = SingleVariableNode("a", 180.0)
    node = BasicMathNode("sin(deg2rad(x) + pi / 2.0)", x=node_a.a, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == pytest.approx(-1.0)


def test_basic_math_node_multi_dim():
    """Test that we can perform multidimensional computations via a BasicMathNode using numpy."""
    node_a = SingleVariableNode("a", np.array([10.0, 20.0]))
    node_b = SingleVariableNode("b", np.array([-5.0, 5.0]))
    node = BasicMathNode("a + b", a=node_a.a, b=node_b.b, node_label="test", backend="numpy")
    state = node.sample_parameters()
    assert np.array_equal(state["test"]["function_node_result"], [5.0, 25.0])

    state = node.sample_parameters(num_samples=20)
    results = state["test"]["function_node_result"]
    assert results.shape == (20, 2)
    for row in range(20):
        assert np.array_equal(results[row], [5.0, 25.0])


def test_basic_math_node_fail():
    """Test that we perform the needed checks for a math node."""
    # Imports not allowed
    with pytest.raises(ValueError):
        _ = BasicMathNode("import os")

    # Ifs not allowed (won't work with JAX)
    with pytest.raises(ValueError):
        _ = BasicMathNode("x if 1.0 else 1.0", x=2.0)

    # We only allow functions on the allow list.
    with pytest.raises(ValueError):
        _ = BasicMathNode("fake_delete_everything_no_confirm('./')")
    with pytest.raises(ValueError):
        _ = BasicMathNode("median(10, 20)")

    # All variables must be defined.
    with pytest.raises(ValueError):
        _ = BasicMathNode("x + y", x=1.0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_basic_math_node_error():
    """Test that we augment the error with information about the expression and parameters."""
    node = BasicMathNode("y / x", x=0.0, y=1.0)
    try:
        node.sample_parameters()
    except ZeroDivisionError as err:
        assert str(err) == "Error during math operation 'y / x' with args={'x': 0.0, 'y': 1.0}"

    node = BasicMathNode("sqrt(x)", x=-10.0)
    try:
        node.sample_parameters()
    except ValueError as err:
        assert str(err) == "Error during math operation 'np.sqrt(x)' with args={'x': -10.0}"


def test_basic_math_node_numpy():
    """Test that we can perform computations via a BasicMathNode."""
    node_a = SingleVariableNode("a", 10.0)
    node_b = SingleVariableNode("b", -5.0)
    node = BasicMathNode("a + b", a=node_a.a, b=node_b.b, node_label="test", backend="numpy")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Try with a math function.
    node_c = SingleVariableNode("c", 1000.0)
    node = BasicMathNode("a + b - log10(c)", a=10.0, b=5.0, c=node_c.c, node_label="test", backend="numpy")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 12.0

    # Try with a second math function.
    node = BasicMathNode(
        "sqrt(a) + b - log10(c)", a=16.0, b=4.0, c=node_c.c, node_label="test", backend="numpy"
    )
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Test that we can reproduce the power function.
    node_d = SingleVariableNode("d", 5.0)
    node = BasicMathNode("a ** b", a=node_d.d, b=2.5, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == pytest.approx(math.pow(5.0, 2.5))


def test_basic_math_node_jax():
    """Test that we can perform computations via a BasicMathNode."""
    node_a = SingleVariableNode("a", 10.0)
    node_b = SingleVariableNode("b", -5.0)
    node = BasicMathNode("a + b", a=node_a.a, b=node_b.b, node_label="test", backend="jax")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Try with a math function.
    node_c = SingleVariableNode("c", 1000.0)
    node = BasicMathNode("a + b - log10(c)", a=10.0, b=5.0, c=node_c.c, node_label="test", backend="jax")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 12.0

    # Try with a second math function.
    node = BasicMathNode(
        "sqrt(a) + b - log10(c)", a=16.0, b=4.0, c=node_c.c, node_label="test", backend="jax"
    )
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == 5.0

    # Test that we can reproduce the power function.
    node_d = SingleVariableNode("d", 5.0)
    node = BasicMathNode("a ** b", a=node_d.d, b=2.5, node_label="test", backend="math")
    state = node.sample_parameters()
    assert state["test"]["function_node_result"] == pytest.approx(math.pow(5.0, 2.5))


def test_basic_math_node_autodiff_jax():
    """Test that we can do auto-differentiation with JAX."""
    node_a = SingleVariableNode("a", 16.0, node_label="a_node")
    node_b = SingleVariableNode("b", 1000.0, node_label="b_node")

    # Create a basic math function and create tghe pytree.
    node = BasicMathNode(
        "sqrt(a) + 1.0 - log10(b)", a=node_a.a, b=node_b.b, node_label="diff_test", backend="jax"
    )
    state = node.sample_parameters()
    pytree = node.build_pytree(state)

    gr_func = jax.value_and_grad(node.generate)
    values, gradients = gr_func(pytree)
    assert values == 2.0
    assert gradients["a_node"]["a"] > 0.0
    assert gradients["b_node"]["b"] < 0.0
