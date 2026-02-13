import numpy as np
import pytest
from lightcurvelynx.math_nodes.np_random import NumpyMultivariateNormalFunc, NumpyRandomFunc


def test_numpy_random_uniform():
    """Test that we can generate numbers from a uniform distribution."""
    np_node = NumpyRandomFunc("uniform", seed=100)

    values = np.array([np_node.generate() for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 1.0)
    assert np.all(values >= 0.0)
    assert np.abs(np.mean(values) - 0.5) < 0.01

    # If we reuse the seed, we get the same numbers.
    np_node2 = NumpyRandomFunc("uniform", seed=100)
    values2 = np.array([np_node2.generate() for _ in range(10_000)])
    assert np.allclose(values, values2)

    # If we use a different seed, we get different numbers
    np_node2 = NumpyRandomFunc("uniform", seed=101)
    values2 = np.array([np_node2.generate() for _ in range(10_000)])
    assert not np.allclose(values, values2)

    # But we can override the seed and get the same results again.
    np_node2.set_seed(100)
    values2 = np.array([np_node2.generate() for _ in range(10_000)])
    assert np.allclose(values, values2)

    # We can change the range.
    np_node3 = NumpyRandomFunc("uniform", low=10.0, high=20.0, seed=100)
    values = np.array([np_node3.generate() for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 10.0)
    assert np.abs(np.mean(values) - 15.0) < 0.5

    # We fail with invalid sizes.
    with pytest.raises(ValueError):
        NumpyRandomFunc("uniform", size=(10, -1))
    with pytest.raises(ValueError):
        NumpyRandomFunc("uniform", size=())


def test_numpy_random_uniform_multi_samples():
    """Test that we can generate many numbers at once from a uniform distribution."""
    np_node = NumpyRandomFunc("uniform", seed=100)
    state = np_node.sample_parameters(num_samples=10_000)
    samples = np_node.get_param(state, "function_node_result")
    assert len(samples) == 10_000
    assert len(np.unique(samples)) > 1_000
    assert np.abs(np.mean(samples) - 0.5) < 0.01


def test_numpy_random_uniform_multi_dim():
    """Test that we can generate multi-dimensional vectors from a uniform distribution."""
    # Sample size 2 arrays
    np_node = NumpyRandomFunc("uniform", seed=100, size=2)
    state = np_node.sample_parameters(num_samples=10)
    samples = np_node.get_param(state, "function_node_result")
    assert samples.shape == (10, 2)
    assert len(np.unique(samples.flatten())) == 20

    # Sample size (2, 3) arrays.
    np_node = NumpyRandomFunc("uniform", seed=100, size=(2, 3))
    state = np_node.sample_parameters(num_samples=10)
    samples = np_node.get_param(state, "function_node_result")
    assert samples.shape == (10, 2, 3)
    assert len(np.unique(samples.flatten())) == 60

    # Sample size (2, 3) arrays with parameters. For a given element of each sample,
    # the values should be between the corresponding low and high values.
    num_samples = 100
    low_vals = np.array([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0]])
    high_vals = low_vals + 5.0
    np_node = NumpyRandomFunc("uniform", low=low_vals, high=high_vals, seed=100, size=(2, 3))
    state = np_node.sample_parameters(num_samples=num_samples)
    samples = np_node.get_param(state, "function_node_result")
    assert samples.shape == (100, 2, 3)
    assert len(np.unique(samples.flatten())) == 600
    assert np.all(samples[:, 0, 0] >= 0.0) and np.all(samples[:, 0, 0] <= 5.0)
    assert np.all(samples[:, 0, 1] >= 10.0) and np.all(samples[:, 0, 1] <= 15.0)
    assert np.all(samples[:, 0, 2] >= 20.0) and np.all(samples[:, 0, 2] <= 25.0)
    assert np.all(samples[:, 1, 0] >= 30.0) and np.all(samples[:, 1, 0] <= 35.0)
    assert np.all(samples[:, 1, 1] >= 40.0) and np.all(samples[:, 1, 1] <= 45.0)
    assert np.all(samples[:, 1, 2] >= 50.0) and np.all(samples[:, 1, 2] <= 55.0)

    # We can mix multi-dimensional parameters with single value parameters.
    num_samples = 100
    low_vals = np.array([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0]])
    np_node = NumpyRandomFunc("uniform", low=low_vals, high=100.0, seed=100, size=(2, 3))
    state = np_node.sample_parameters(num_samples=num_samples)
    samples = np_node.get_param(state, "function_node_result")
    assert samples.shape == (100, 2, 3)
    assert len(np.unique(samples.flatten())) == 600
    assert np.all(samples[:, 0, 0] >= 0.0) and np.all(samples[:, 0, 0] <= 100.0)
    assert np.all(samples[:, 0, 1] >= 10.0) and np.all(samples[:, 0, 1] <= 100.0)
    assert np.all(samples[:, 0, 2] >= 20.0) and np.all(samples[:, 0, 2] <= 100.0)
    assert np.all(samples[:, 1, 0] >= 30.0) and np.all(samples[:, 1, 0] <= 100.0)
    assert np.all(samples[:, 1, 1] >= 40.0) and np.all(samples[:, 1, 1] <= 100.0)
    assert np.all(samples[:, 1, 2] >= 50.0) and np.all(samples[:, 1, 2] <= 100.0)

    # If we do not specify a size and use a single sample, we get a float.
    np_node = NumpyRandomFunc("uniform", seed=100)
    state = np_node.sample_parameters(num_samples=1)
    assert np.isscalar(np_node.get_param(state, "function_node_result"))


def test_numpy_random_normal():
    """Test that we can generate numbers from a normal distribution."""
    np_node = NumpyRandomFunc("normal", loc=100.0, scale=10.0, seed=100, node_label="normal1")

    values = np.array([np_node.generate() for _ in range(10_000)])
    assert np.abs(np.mean(values) - 100.0) < 0.5
    assert np.abs(np.std(values) - 10.0) < 0.5

    # If we reuse the seed, we get the same number.
    np_node2 = NumpyRandomFunc("normal", loc=100.0, scale=10.0, seed=100)
    values2 = np.array([np_node2.generate() for _ in range(10_000)])
    assert np.allclose(values, values2)

    # Check that we can get a dependency graph that correctly includes this node.
    dep_graph = np_node.build_dependency_graph()
    assert dep_graph.all_nodes == {"normal1"}
    assert dep_graph.incoming["normal1.function_node_result"] == set(["normal1.loc", "normal1.scale"])
    assert dep_graph.outgoing["normal1.loc"] == set(["normal1.function_node_result"])
    assert dep_graph.outgoing["normal1.scale"] == set(["normal1.function_node_result"])


def test_numpy_random_integers():
    """Test that we can generate numbers from an integer distribution."""
    np_node = NumpyRandomFunc("integers", low=500, high=100_000, seed=100)

    values = np.array([np_node.generate() for _ in range(1_000)])
    assert len(np.unique(values)) > 950  # Allow a few lucky duplicates.
    assert np.all(values >= 500)
    assert np.all(values < 100_000)


def test_numpy_random_given_rng():
    """Test that we can generate numbers from a uniform distribution."""
    np_node1 = NumpyRandomFunc("uniform", seed=100, node_label="node1")
    np_node2 = NumpyRandomFunc("uniform", seed=100, node_label="node2")

    # The first value generated is the same because they are using the node's seed.
    assert np_node1.generate() == pytest.approx(np_node2.generate())

    # But if we use a given random number generators, we get different values.
    value1 = np_node1.generate(rng_info=np.random.default_rng(1))
    value2 = np_node2.generate(rng_info=np.random.default_rng(2))
    assert value1 != pytest.approx(value2)


def test_numpy_multivariate_normal():
    """Test that we can generate numbers from a multi-variate normal distribution."""
    mean = [0.0, 10.0]
    cov = [[1.0, 0.5], [0.5, 2.0]]

    # We cannot sample from a multi-variate normal distribution with NumpyRandomFunc.
    with pytest.raises(ValueError):
        _ = NumpyRandomFunc("multivariate_normal", mean=mean, cov=cov, seed=100, size=2)

    np_node = NumpyMultivariateNormalFunc(mean=mean, cov=cov, node_label="xy")
    state = np_node.sample_parameters(num_samples=10_000)

    values = state["xy"]["function_node_result"]
    assert values.shape == (10_000, 2)
    assert np.abs(np.mean(values[:, 0]) - mean[0]) < 0.1
    assert np.abs(np.mean(values[:, 1]) - mean[1]) < 0.1
    assert np.abs(np.cov(values.T) - cov).max() < 0.1

    # We can set the seed to get deterministic results.
    np_node.set_seed(100)
    state1 = np_node.sample_parameters(num_samples=10_000)
    results1 = state1["xy"]["function_node_result"]

    np_node2 = NumpyMultivariateNormalFunc(mean=mean, cov=cov, node_label="xy", seed=100)
    state2 = np_node2.sample_parameters(num_samples=10_000)
    results2 = state2["xy"]["function_node_result"]

    assert np.allclose(results1, results2)


def test_numpy_choice_fails():
    """Test that we cannot use NumpyRandomFunc with a choice distribution."""
    with pytest.raises(ValueError):
        NumpyRandomFunc("choice", a=5)


def test_numpy_random_invalid_func():
    """Test that we cannot use an invalid function."""
    with pytest.raises(ValueError):
        NumpyRandomFunc("invalid_func")
