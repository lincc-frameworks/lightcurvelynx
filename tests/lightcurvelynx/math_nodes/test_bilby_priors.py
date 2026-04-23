import numpy as np
import pytest
from lightcurvelynx.math_nodes.bilby_priors import BilbyPriorNode


def test_bilby_priors_simple():
    """Test that we can generate numbers from a simple Bilby prior."""
    bilby = pytest.importorskip("bilby.core.prior")
    priors = dict(
        a=bilby.Uniform(0, 5, "a"),
        b=bilby.Uniform(0, 10, "b"),
        c=bilby.Cosine(-2, 2, "c"),
    )
    prior_dict = bilby.PriorDict(priors)
    node = BilbyPriorNode(prior=prior_dict, seed=100, node_label="sampler")
    assert "a" in node.outputs
    assert "b" in node.outputs
    assert "c" in node.outputs

    params = node.sample_parameters(num_samples=10_000)

    # "a" should be 10,000 samples from a uniform distribution between 0 and 5.
    assert params["sampler"]["a"].shape == (10_000,)
    assert np.all(params["sampler"]["a"] >= 0)
    assert np.all(params["sampler"]["a"] <= 5)
    for bin in np.linspace(0, 4, 5):
        in_bin = np.sum((params["sampler"]["a"] >= bin) & (params["sampler"]["a"] < bin + 1))
        assert 1_500 < in_bin < 2_500

    # "b" should be 10,000 samples from a uniform distribution between 0 and 10.
    assert params["sampler"]["b"].shape == (10_000,)
    assert np.all(params["sampler"]["b"] >= 0)
    assert np.all(params["sampler"]["b"] <= 10)
    for bin in np.linspace(0, 9, 10):
        in_bin = np.sum((params["sampler"]["b"] >= bin) & (params["sampler"]["b"] < bin + 1))
        assert 500 < in_bin < 1_500

    # "c" should be 10,000 samples from a cosine distribution between -2 and 2.
    assert params["sampler"]["c"].shape == (10_000,)
    assert np.all(params["sampler"]["c"] >= -2)
    assert np.all(params["sampler"]["c"] <= 2)


def test_bilby_priors_single():
    """Test that we can generate numbers from a simple Bilby prior with a single sample."""
    bilby = pytest.importorskip("bilby.core.prior")
    priors = dict(
        a=bilby.Uniform(0, 5, "a"),
        b=bilby.Uniform(0, 10, "b"),
        c=bilby.Cosine(-2, 2, "c"),
    )
    # Pass the dictionary in directly to the BilbyPriorNode.
    node = BilbyPriorNode(prior=priors, seed=100, node_label="sampler")

    params = node.sample_parameters(num_samples=1)

    # "a" should be a single sample from a uniform distribution between 0 and 5.
    assert np.isscalar(params["sampler"]["a"])
    assert params["sampler"]["a"] >= 0
    assert params["sampler"]["a"] <= 5

    # "b" should be a single sample from a uniform distribution between 0 and 10.
    assert np.isscalar(params["sampler"]["b"])
    assert params["sampler"]["b"] >= 0
    assert params["sampler"]["b"] <= 10

    # "c" should be a single sample from a cosine distribution between -2 and 2.
    assert np.isscalar(params["sampler"]["c"])
    assert params["sampler"]["c"] >= -2
    assert params["sampler"]["c"] <= 2


def test_bilby_priors_errors():
    """Test that BilbyPriorNode raises errors when given an empty prior model."""
    pytest.importorskip("bilby.core.prior")
    with pytest.raises(ValueError):
        BilbyPriorNode(prior=dict(), seed=100, node_label="sampler")
