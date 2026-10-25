import numpy as np
import pytest

from lightcurvelynx.math_nodes.np_random import NumpyMultivariateNormalFunc, NumpyRandomFunc


def test_numpy_random_uniform():
    """Test that we can generate numbers from a uniform distribution."""
    np_node = NumpyRandomFunc("uniform")

    rng1 = np.random.default_rng(100)
    values = np.array([np_node.generate(rng_info=rng1) for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 1.0)
    assert np.all(values >= 0.0)
    assert np.abs(np.mean(values) - 0.5) < 0.01

    # If we reuse the seed for the random number generator, we get the same numbers.
    np_node2 = NumpyRandomFunc("uniform")
    rng2 = np.random.default_rng(100)
    values2 = np.array([np_node2.generate(rng_info=rng2) for _ in range(10_000)])
    assert np.allclose(values, values2)

    # If we use a different seed for the random number generator, we get different numbers
    np_node2 = NumpyRandomFunc("uniform")
    rng2 = np.random.default_rng(101)
    values2 = np.array([np_node2.generate(rng_info=rng2) for _ in range(10_000)])
    assert not np.allclose(values, values2)

    # If we don't pass a random number generator, we still get different numbers.
    np_node2b = NumpyRandomFunc("uniform")
    values2 = np.array([np_node2b.generate() for _ in range(10_000)])
    assert not np.allclose(values, values2)

    # We can change the range.
    np_node3 = NumpyRandomFunc("uniform", low=10.0, high=20.0)
    rng3 = np.random.default_rng(105)
    values = np.array([np_node3.generate(rng_info=rng3) for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 10.0)
    assert np.abs(np.mean(values) - 15.0) < 0.5

    # We can specify the range with lists of values.
    np_node4 = NumpyRandomFunc("uniform", low=[0.0, 10.0], high=[1.0, 20.0], size=2)
    rng4 = np.random.default_rng(106)
    values = np.array([np_node4.generate(rng_info=rng4) for _ in range(1000)])
    assert values.shape == (1000, 1, 2)
    assert np.all(values[:, 0, 0] <= 1.0)
    assert np.all(values[:, 0, 0] >= 0.0)
    assert np.all(values[:, 0, 1] <= 20.0)
    assert np.all(values[:, 0, 1] >= 10.0)

    # We fail with invalid sizes.
    with pytest.raises(ValueError):
        NumpyRandomFunc("uniform", size=(10, -1))
    with pytest.raises(ValueError):
        NumpyRandomFunc("uniform", size=())


def test_numpy_random_func_invalid_kwargs():
    """Test that invalid kwargs raise errors."""
    with pytest.raises(ValueError):
        NumpyRandomFunc("uniform", seed=123)
    with pytest.raises(ValueError):
        NumpyRandomFunc("uniform", fixed_params={"low": 0.0})


def test_numpy_random_uniform_multi_samples():
    """Test that we can generate many numbers at once from a uniform distribution."""
    rng = np.random.default_rng(101010)  # For reproducible tests

    np_node = NumpyRandomFunc("uniform")
    state = np_node.sample_parameters(num_samples=10_000, rng_info=rng)
    samples = np_node.get_param(state, "function_node_result")
    assert len(samples) == 10_000
    assert len(np.unique(samples)) > 1_000
    assert np.abs(np.mean(samples) - 0.5) < 0.01


def test_numpy_random_uniform_multi_dim():
    """Test that we can generate multi-dimensional vectors from a uniform distribution."""
    test_rng = np.random.default_rng(10000)  # For reproducible tests

    # Sample size 2 arrays
    np_node = NumpyRandomFunc("uniform", size=2)
    state = np_node.sample_parameters(num_samples=100, rng_info=test_rng)
    samples = np_node.get_param(state, "function_node_result")
    assert samples.shape == (100, 2)
    assert len(np.unique(samples.flatten())) == 200
    assert np.all(samples >= 0.0) and np.all(samples <= 1.0)

    # Test that if we use the same seed for the random number generator, we
    # get the same results.
    test_rng2 = np.random.default_rng(10000)
    np_node2 = NumpyRandomFunc("uniform", size=2)
    state2 = np_node2.sample_parameters(num_samples=100, rng_info=test_rng2)
    samples2 = np_node2.get_param(state2, "function_node_result")
    assert np.allclose(samples, samples2)

    # Sample size (2, 3) arrays.
    np_node = NumpyRandomFunc("uniform", size=(2, 3))
    state = np_node.sample_parameters(num_samples=10, rng_info=test_rng)
    samples = np_node.get_param(state, "function_node_result")
    assert samples.shape == (10, 2, 3)
    assert np.all(samples >= 0.0) and np.all(samples <= 1.0)
    assert len(np.unique(samples.flatten())) == 60

    # Sample size (2, 3) arrays with parameters. For a given element of each sample,
    # the values should be between the corresponding low and high values.
    num_samples = 100
    low_vals = np.array([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0]])
    high_vals = low_vals + 5.0
    np_node = NumpyRandomFunc("uniform", low=low_vals, high=high_vals, size=(2, 3))
    state = np_node.sample_parameters(num_samples=num_samples, rng_info=test_rng)
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
    np_node = NumpyRandomFunc("uniform", low=low_vals, high=100.0, size=(2, 3))
    state = np_node.sample_parameters(num_samples=num_samples, rng_info=test_rng)
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
    np_node = NumpyRandomFunc("uniform")
    state = np_node.sample_parameters(num_samples=1, rng_info=test_rng)
    assert np.isscalar(np_node.get_param(state, "function_node_result"))


def test_numpy_random_normal():
    """Test that we can generate numbers from a normal distribution."""
    np_node = NumpyRandomFunc("normal", loc=100.0, scale=10.0, node_label="normal1")

    rng1 = np.random.default_rng(10005)
    values = np.array([np_node.generate(rng_info=rng1) for _ in range(10_000)])
    assert np.abs(np.mean(values) - 100.0) < 0.5
    assert np.abs(np.std(values) - 10.0) < 0.5

    # If we reuse the seed, we get the same number.
    np_node2 = NumpyRandomFunc("normal", loc=100.0, scale=10.0)
    rng2 = np.random.default_rng(10005)
    values2 = np.array([np_node2.generate(rng_info=rng2) for _ in range(10_000)])
    assert np.allclose(values, values2)

    # Check that we can get a dependency graph that correctly includes this node.
    dep_graph = np_node.build_dependency_graph()
    assert dep_graph.all_nodes == {"normal1"}
    assert dep_graph.incoming["normal1.function_node_result"] == set(["normal1.loc", "normal1.scale"])
    assert dep_graph.outgoing["normal1.loc"] == set(["normal1.function_node_result"])
    assert dep_graph.outgoing["normal1.scale"] == set(["normal1.function_node_result"])


def test_numpy_random_integers():
    """Test that we can generate numbers from an integer distribution."""
    np_node = NumpyRandomFunc("integers", low=500, high=100_000)
    test_rng = np.random.default_rng(100)

    values = np.array([np_node.generate(rng_info=test_rng) for _ in range(1_000)])
    assert len(np.unique(values)) > 950  # Allow a few lucky duplicates.
    assert np.all(values >= 500)
    assert np.all(values < 100_000)

    # Try with a different range and use the sample_parameters method to generate states.
    np_node2 = NumpyRandomFunc("integers", low=0, high=500, node_label="int2")
    states = np_node2.sample_parameters(num_samples=100, rng_info=test_rng)
    values2 = states["int2"]["function_node_result"]
    assert len(np.unique(values2)) > 90  # Allow a few lucky duplicates.
    assert np.all(values2 >= 0)
    assert np.all(values2 < 500)


def test_numpy_multivariate_normal():
    """Test that we can generate numbers from a multi-variate normal distribution."""
    mean = [0.0, 10.0]
    cov = [[1.0, 0.5], [0.5, 2.0]]

    # We cannot sample from a multi-variate normal distribution with NumpyRandomFunc.
    with pytest.raises(ValueError):
        _ = NumpyRandomFunc("multivariate_normal", mean=mean, cov=cov, size=2)

    np_node = NumpyMultivariateNormalFunc(mean=mean, cov=cov, node_label="xy")
    rng = np.random.default_rng(100)
    state = np_node.sample_parameters(num_samples=10_000, rng_info=rng)

    values = state["xy"]["function_node_result"]
    assert values.shape == (10_000, 2)
    assert np.abs(np.mean(values[:, 0]) - mean[0]) < 0.1
    assert np.abs(np.mean(values[:, 1]) - mean[1]) < 0.1
    assert np.abs(np.cov(values.T) - cov).max() < 0.1

    # We can set the seed to get deterministic results.
    rng = np.random.default_rng(100)
    state1 = np_node.sample_parameters(num_samples=10_000, rng_info=rng)
    results1 = state1["xy"]["function_node_result"]

    # Using the same seed should produce the same results.
    np_node2 = NumpyMultivariateNormalFunc(mean=mean, cov=cov, node_label="xy")
    rng = np.random.default_rng(100)
    state2 = np_node2.sample_parameters(num_samples=10_000, rng_info=rng)
    results2 = state2["xy"]["function_node_result"]
    assert np.allclose(results1, results2)

    # Using a different seeded RNG should produce different results.
    rng = np.random.default_rng(101)
    state3 = np_node.sample_parameters(num_samples=10_000, rng_info=rng)
    results3 = state3["xy"]["function_node_result"]
    assert not np.allclose(results1, results3)


def test_numpy_multivariate_normal_invalid_kwargs():
    """Test that invalid kwargs raise errors."""
    mean = [0.0, 10.0]
    cov = [[1.0, 0.5], [0.5, 2.0]]

    with pytest.raises(ValueError):
        NumpyMultivariateNormalFunc(mean=mean, cov=cov, seed=123)
    with pytest.raises(ValueError):
        NumpyMultivariateNormalFunc(mean=mean, cov=cov, fixed_params={"mean": mean})


def test_numpy_choice_fails():
    """Test that we cannot use NumpyRandomFunc with a choice distribution."""
    with pytest.raises(ValueError):
        NumpyRandomFunc("choice", a=5)


def test_numpy_random_invalid_func():
    """Test that we cannot use an invalid function."""
    with pytest.raises(ValueError):
        NumpyRandomFunc("invalid_func")
