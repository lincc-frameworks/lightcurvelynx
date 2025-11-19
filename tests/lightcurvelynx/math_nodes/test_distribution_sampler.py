import numpy as np
import pytest
from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.math_nodes.distribution_sampler import DistributionSampler


def test_distribution_sampler():
    """Test that we can generate numbers from a given distribution."""
    # Create a bimodal distribution with peaks at 2.0 and 6.0.
    x_value = np.arange(10)
    pdf_values = np.array([0.0, 0.05, 0.4, 0.05, 0.0, 0.05, 0.4, 0.05, 0.0, 0.0])
    sampler_node = DistributionSampler(
        x_value,
        pdf_values,
        seed=100,
        node_label="sampler",
        outputs=["sample"],
    )

    # Test that we generate samples from the given distribution. We extract the samples
    # from the state to check that they are correctly saved.
    num_samples = 10_000
    state = GraphState(num_samples=num_samples)
    _ = sampler_node.compute(state)
    samples = state["sampler"]["sample"]

    # Check that the samples are from the correct range of x.
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 9.0)

    # Check that most of the samples are near the two peaks (within 1.2) and
    # that there are few samples at other points.
    assert np.sum(np.abs(samples - 2.0) <= 1.2) > 0.4 * num_samples
    assert np.sum(np.abs(samples - 6.0) <= 1.2) > 0.4 * num_samples
    assert np.sum(np.abs(samples - 0.0) <= 0.5) < 0.2 * num_samples
    assert np.sum(np.abs(samples - 9.0) <= 0.5) < 0.05 * num_samples
    assert np.sum(np.abs(samples - 4.0) <= 0.5) < 0.05 * num_samples

    # If we reset the seed, we should get the same samples.
    sampler_node.set_seed(100)
    _ = sampler_node.compute(state)
    samples2 = state["sampler"]["sample"]
    assert np.all(samples == samples2)

    # Creation fails if x and y have different lengths.
    with pytest.raises(ValueError):
        _ = DistributionSampler(x_value, pdf_values[:-1])

    # Creation fails if y has negative values.
    with pytest.raises(ValueError):
        _ = DistributionSampler(x_value, pdf_values - 0.1)
