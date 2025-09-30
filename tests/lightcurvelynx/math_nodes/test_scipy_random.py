import numpy as np
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.scipy_random import (
    NumericalInversePolynomialFunc,
    SampleLogPDF,
    SamplePDF,
    ScipyRandomDist,
)


class FlatDist:
    """A flat (uniform) distribution used for testing.

    Attributes
    ----------
    min_val : `float`
        The minimum value of the distribution.
    max_val : `float`
        The maximum value of the distribution.
    width : `float`
        The width of the distribution.
    """

    def __init__(self, min_val, max_val):
        if min_val >= max_val:
            raise ValueError("Invalid flat_dist bounds.")
        self.min_val = min_val
        self.max_val = max_val
        self.width = self.max_val - self.min_val

    def pdf(self, xval):
        """The pdf of the distribution.

        Parameters
        ----------
        xval: `float`
            The x value to evaluate.

        Returns
        -------
        result : `float`
            The value of the pdf function at the given points.
        """
        if xval <= self.min_val:
            return 0.0
        elif xval >= self.max_val:
            return 0.0
        else:
            return 1.0 / self.width


def test_numerical_inverse_polynomial_func_object():
    """Test that we can generate numbers from a uniform distribution."""
    dist = FlatDist(min_val=0.25, max_val=0.75)
    scipy_node = NumericalInversePolynomialFunc(dist, seed=100)

    # Test that we get a uniform distribution.
    num_bins = 10
    num_samples = 10_000
    counts = [0] * num_bins
    all_values = [0] * num_samples
    for i in range(10_000):
        value = scipy_node.generate()
        assert 0.25 <= value <= 0.75

        all_values[i] = value
        bin = int((value - 0.25) * (num_bins / 0.5))
        counts[bin] += 1

    for idx in range(num_bins):
        assert np.abs(counts[idx] - 10_000 / num_bins) < 200.0

    # If we reuse the same seed, we get the same values.
    dist2 = FlatDist(min_val=0.25, max_val=0.75)
    scipy_node2 = NumericalInversePolynomialFunc(dist2, seed=100)
    all_values2 = [0] * num_samples
    for i in range(10_000):
        all_values2[i] = scipy_node2.generate()
    assert np.allclose(all_values, all_values2)

    # If we use a different seed, we get different values.
    dist3 = FlatDist(min_val=0.25, max_val=0.75)
    scipy_node3 = NumericalInversePolynomialFunc(dist3, seed=101)
    all_values3 = [0] * num_samples
    for i in range(10_000):
        all_values3[i] = scipy_node3.generate()
    assert not np.allclose(all_values, all_values3)

    # We can batch generate results.
    state = scipy_node.sample_parameters(num_samples=500)
    samples = scipy_node.get_param(state, "function_node_result")
    assert len(samples) == 500
    assert len(np.unique(samples)) > 50
    assert np.abs(np.mean(samples) - 0.5) < 0.01


def test_numerical_inverse_polynomial_func_object_seed():
    """Test that we can generate numbers from a uniform distribution with a given seed."""
    dist = FlatDist(min_val=0.25, max_val=0.75)
    scipy_node = NumericalInversePolynomialFunc(dist)

    # Sample without a given seed.
    state1 = scipy_node.sample_parameters(num_samples=10)
    values1 = scipy_node.get_param(state1, "function_node_result")

    # Re-sample with a given seed=100.
    state2 = scipy_node.sample_parameters(num_samples=10, rng_info=np.random.default_rng(seed=100))
    values2 = scipy_node.get_param(state2, "function_node_result")

    # Re-sample again with a given seed=100.
    state3 = scipy_node.sample_parameters(num_samples=10, rng_info=np.random.default_rng(seed=100))
    values3 = scipy_node.get_param(state3, "function_node_result")

    assert np.allclose(values2, values3)
    assert not np.allclose(values1, values2)
    assert not np.allclose(values1, values3)


def test_numerical_inverse_polynomial_func_class():
    """Test that we can generate numbers from a uniform distribution."""
    scipy_node = NumericalInversePolynomialFunc(
        FlatDist,
        min_val=NumpyRandomFunc("uniform", low=0.0, high=0.4, seed=101),
        max_val=NumpyRandomFunc("uniform", low=0.6, high=1.0, seed=102),
        seed=100,
    )

    # Check that when we sample both the values and the distribution's parameters.
    num_samples = 1000
    states = scipy_node.sample_parameters(num_samples=num_samples)
    values = scipy_node.get_param(states, "function_node_result")
    min_vals = scipy_node.get_param(states, "min_val")
    max_vals = scipy_node.get_param(states, "max_val")
    assert len(np.unique(values)) == num_samples
    assert len(np.unique(min_vals)) == num_samples
    assert len(np.unique(max_vals)) == num_samples

    # Test that the generated values are consistent with the distributions.
    assert np.all(values >= min_vals)
    assert np.all(values <= max_vals)


def test_numerical_sample_pdf():
    """Test that we can create a SamplePDF node from a function."""

    def _triangle_pdf(x):
        if x < 0.0:
            return 0.0
        if x > 2.0:
            return 0.0
        return 1.0 - 0.5 * x

    scipy_node = SamplePDF(_triangle_pdf, seed=100)

    # Test that we get distribution that ramps down as x increases.
    num_samples = 50_000
    counts = np.zeros(10)
    for _ in range(num_samples):
        value = scipy_node.generate()
        assert 0.0 <= value <= 2.0

        bin = int(10 * value / 2.0)
        counts[bin] += 1

    mid_heights = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])
    expected = mid_heights * 0.2 * num_samples
    for idx in range(10):
        assert np.abs(counts[idx] - expected[idx]) < 200.0


def test_numerical_sample_logpdf():
    """Test that we can create a SampleLogPDF node from a function."""

    def _triangle_logpdf(x):
        if x < 0.0:
            return -np.inf
        if x > 2.0:
            return -np.inf
        return np.log(1.0 - 0.5 * x)

    scipy_node = SampleLogPDF(_triangle_logpdf, seed=100)

    # Test that we get distribution that ramps down as x increases.
    num_samples = 50_000
    counts = np.zeros(10)
    for _ in range(num_samples):
        value = scipy_node.generate()
        assert 0.0 <= value <= 2.0

        bin = int(10 * value / 2.0)
        counts[bin] += 1

    mid_heights = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])
    expected = mid_heights * 0.2 * num_samples
    for idx in range(10):
        assert np.abs(counts[idx] - expected[idx]) < 200.0


def test_scipy_random_dist():
    """Test that we can generate numbers from a uniform distribution."""
    scipy_node = ScipyRandomDist("uniform", seed=100, node_label="uniform1")

    samples = scipy_node.sample_parameters(num_samples=10_000)
    values = scipy_node.get_param(samples, "function_node_result")
    assert len(values) == 10_000
    assert len(np.unique(values)) > 10
    assert np.all(values <= 1.0)
    assert np.all(values >= 0.0)
    assert np.abs(np.mean(values) - 0.5) < 0.01

    # If we reuse the seed, we get the same numbers.
    scipy_node2 = ScipyRandomDist("uniform", seed=100)
    samples2 = scipy_node2.sample_parameters(num_samples=10_000)
    values2 = scipy_node2.get_param(samples2, "function_node_result")
    assert np.allclose(values, values2)

    # If we use a different seed, we get different numbers
    scipy_node2 = ScipyRandomDist("uniform", seed=101)
    samples2 = scipy_node2.sample_parameters(num_samples=10_000)
    values2 = scipy_node2.get_param(samples2, "function_node_result")
    assert not np.allclose(values, values2)

    # But we can override the seed and get the same results again.
    scipy_node2.set_seed(100)
    samples2 = scipy_node2.sample_parameters(num_samples=10_000)
    values2 = scipy_node2.get_param(samples2, "function_node_result")
    assert np.allclose(values, values2)


def test_numpy_random_uniform_diff_distributions():
    """Test that we can generate multi-dimensional vectors from a uniform distribution."""
    # We can use different hyperparameters to create different distributions.
    # This generates different uniform distributions such that distribution i
    # is uniform between 10*i and 10*i+5.
    low_val = np.arange(0, 5000, 10)
    scipy_node = ScipyRandomDist(
        "uniform",
        seed=100,
        loc=GivenValueList(low_val),
        scale=GivenValueList(np.full_like(low_val, 5.0)),
    )
    state = scipy_node.sample_parameters(num_samples=50)
    samples = scipy_node.get_param(state, "function_node_result")
    assert len(samples) == 50
    for i in range(50):
        assert samples[i] >= 10 * i and samples[i] <= 10 * i + 5.0
