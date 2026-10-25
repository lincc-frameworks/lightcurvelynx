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
    node = BilbyPriorNode(prior=prior_dict, node_label="sampler")
    assert "a" in node.outputs
    assert "b" in node.outputs
    assert "c" in node.outputs

    rng = np.random.default_rng(100)
    params = node.sample_parameters(num_samples=10_000, rng_info=rng)

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

    # If we use the same random number generator, we should get the same distribution.
    rng2 = np.random.default_rng(100)
    params2 = node.sample_parameters(num_samples=10_000, rng_info=rng2)
    assert np.allclose(params["sampler"]["a"], params2["sampler"]["a"])
    assert np.allclose(params["sampler"]["b"], params2["sampler"]["b"])
    assert np.allclose(params["sampler"]["c"], params2["sampler"]["c"])

    # If we use a different random number generator, we should get a different distribution.
    rng3 = np.random.default_rng(101)
    params3 = node.sample_parameters(num_samples=10_000, rng_info=rng3)
    assert not np.allclose(params["sampler"]["a"], params3["sampler"]["a"])
    assert not np.allclose(params["sampler"]["b"], params3["sampler"]["b"])
    assert not np.allclose(params["sampler"]["c"], params3["sampler"]["c"])

    # Test that we fail construction if we set the seed parameter during construction.
    with pytest.raises(ValueError):
        BilbyPriorNode(prior=priors, node_label="sampler", seed=100)


def test_bilby_priors_single():
    """Test that we can generate numbers from a simple Bilby prior with a single sample."""
    bilby = pytest.importorskip("bilby.core.prior")
    priors = dict(
        a=bilby.Uniform(0, 5, "a"),
        b=bilby.Uniform(0, 10, "b"),
        c=bilby.Cosine(-2, 2, "c"),
    )
    # Pass the dictionary in directly to the BilbyPriorNode.
    node = BilbyPriorNode(prior=priors, node_label="sampler")

    rng = np.random.default_rng(100)
    params = node.sample_parameters(num_samples=1, rng_info=rng)

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

    # Test that using a different random number generator gives a different sample.
    rng2 = np.random.default_rng(101)
    params2 = node.sample_parameters(num_samples=1, rng_info=rng2)
    assert not np.allclose(params["sampler"]["a"], params2["sampler"]["a"])
    assert not np.allclose(params["sampler"]["b"], params2["sampler"]["b"])
    assert not np.allclose(params["sampler"]["c"], params2["sampler"]["c"])

    # Test that using the same random number generator gives the same sample.
    rng3 = np.random.default_rng(100)
    params3 = node.sample_parameters(num_samples=1, rng_info=rng3)
    assert np.allclose(params["sampler"]["a"], params3["sampler"]["a"])
    assert np.allclose(params["sampler"]["b"], params3["sampler"]["b"])
    assert np.allclose(params["sampler"]["c"], params3["sampler"]["c"])

    # Without providing a random number generator, we should get a different sample.
    params4 = node.sample_parameters(num_samples=1)
    assert not np.allclose(params["sampler"]["a"], params4["sampler"]["a"])
    assert not np.allclose(params["sampler"]["b"], params4["sampler"]["b"])
    assert not np.allclose(params["sampler"]["c"], params4["sampler"]["c"])


def test_bilby_priors_errors():
    """Test that BilbyPriorNode raises errors when given an empty prior model."""
    pytest.importorskip("bilby.core.prior")
    with pytest.raises(ValueError):
        BilbyPriorNode(prior=dict(), node_label="sampler")
