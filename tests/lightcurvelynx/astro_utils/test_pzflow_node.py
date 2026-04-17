import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from citation_compass import find_in_citations
from lightcurvelynx.astro_utils.pzflow_node import PZFlowNode
from lightcurvelynx.base_models import FunctionNode, ParameterizedNode
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc


class SumNode(ParameterizedNode):
    """A test class for the ParameterizedNode.

    Parameters
    ----------
    value1 : `float`
        The first value.
    value2 : `float`
        The second value.
    value_sum : `float`
        The sum of the two values.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, value1, value2, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("value1", value1, **kwargs)
        self.add_parameter("value2", value2, **kwargs)
        self.add_parameter(
            "value_sum",
            FunctionNode(self._sum, value1=self.value1, value2=self.value2),
            **kwargs,
        )

    def _sum(self, value1, value2):
        return value1 + value2


def test_pzflow_node_sample():
    """Test that we can sample numbers from a PZFlowNode."""
    pzflow = pytest.importorskip("pzflow")
    flow = pzflow.Flow(("x", "y"), pzflow.bijectors.Reverse())
    pz_node = PZFlowNode(flow, node_label="pznode")

    # Sample a bunch of parameters.
    state = pz_node.sample_parameters(num_samples=100)
    assert len(state["pznode"]) == 2
    assert len(state["pznode"]["x"]) == 100
    assert len(state["pznode"]["y"]) == 100

    # If we pass a seeded random number generator, we control the randomness.
    state1 = pz_node.sample_parameters(num_samples=50, rng_info=np.random.default_rng(seed=1))
    state2 = pz_node.sample_parameters(num_samples=50, rng_info=np.random.default_rng(seed=1))
    assert np.allclose(state1["pznode"]["x"], state2["pznode"]["x"])
    assert np.allclose(state1["pznode"]["y"], state2["pznode"]["y"])


def test_pzflow_node_chained():
    """Test that we can sample numbers from a PZFlowNode through another node."""
    pzflow = pytest.importorskip("pzflow")
    flow = pzflow.Flow(("a", "b"), pzflow.bijectors.Reverse())
    pz_node = PZFlowNode(flow, node_label="pznode")
    sum_node = SumNode(value1=pz_node.a, value2=pz_node.b, node_label="sum")

    # Sample a bunch of parameters.
    state = sum_node.sample_parameters(num_samples=25)
    assert len(state["sum"]["value_sum"]) == 25
    assert np.allclose(state["pznode"]["a"], state["sum"]["value1"])
    assert np.allclose(state["pznode"]["b"], state["sum"]["value2"])

    expected_sum = state["pznode"]["a"] + state["pznode"]["b"]
    assert np.allclose(state["sum"]["value_sum"], expected_sum)

    # Make sure that we only call the pzflow's sample() once during sampling of the node
    # (so all the outputs are consistent with each other). In other words, we do
    # not call compute once for each output variable.
    def _mock_sample(self=None, nsamples=1, conditions=None, save_conditions=True, seed=None):
        return pd.DataFrame({"a": [1], "b": [2]})

    with patch("pzflow.flow.Flow.sample", side_effect=_mock_sample) as mocked_sample:
        state = sum_node.sample_parameters(num_samples=1)
        mocked_sample.assert_called_once()


def test_pzflow_node_from_file(test_flow_filename):
    """Test that we can load and query a test flow."""
    pytest.importorskip("pzflow")
    pz_node = PZFlowNode.from_file(test_flow_filename, node_label="loaded_node")

    # Sample the pair of parameters defined by this flow (redshift and hostmass).
    state = pz_node.sample_parameters(num_samples=100)
    assert len(state["loaded_node"]) == 2
    assert len(state["loaded_node"]["redshift"]) == 100
    assert len(state["loaded_node"]["brightness"]) == 100


def test_conditional_pzflow_node(test_conditional_flow_filename):
    """Test that we can load and query a conditional flow."""
    pytest.importorskip("pzflow")
    redshift_node = NumpyRandomFunc(
        "uniform",
        low=0.05,
        high=0.3,
        node_label="redshift_node",
    )

    pz_node = PZFlowNode.from_file(
        test_conditional_flow_filename,
        node_label="loaded_node",
        redshift=redshift_node,
    )

    # Sample the pair of parameters defined by this flow (redshift and hostmass).
    state = pz_node.sample_parameters(num_samples=100)
    assert len(state["loaded_node"]) == 2
    assert len(state["loaded_node"]["redshift"]) == 100
    assert len(state["loaded_node"]["brightness"]) == 100


def test_invalid_conditional_pzflow_node(test_conditional_flow_filename):
    """Test that we raise an error if we try to create a pzflow node from
    a conditional flow, but do not provide the input parameters."""
    pytest.importorskip("pzflow")
    with pytest.raises(ValueError):
        # redshift is missing.
        _ = PZFlowNode.from_file(test_conditional_flow_filename, node_label="loaded_node")


def test_pzflow_node_citation(test_flow_filename):
    """Test that we can recover the citations for pzflow."""
    pytest.importorskip("pzflow")
    _ = PZFlowNode.from_file(test_flow_filename, node_label="loaded_node")
    citations = find_in_citations("PZFlowNode")
    for citation in citations:
        assert "Crenshaw et. al. 2024" in citation


def test_pzflow_node_pickle():
    """Test that we can pickle and unpickle a PZFlowNode."""
    pzflow = pytest.importorskip("pzflow")
    flow = pzflow.Flow(("x", "y"), pzflow.bijectors.Reverse())
    pz_node = PZFlowNode(flow, node_label="pznode")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_pzflow_node.pkl"
        assert not file_path.exists()

        with open(file_path, "wb") as f:
            pickle.dump(pz_node, f)
        assert file_path.exists()

        with open(file_path, "rb") as f:
            loaded_pz_node = pickle.load(f)
        assert loaded_pz_node is not None

        # Sample a parameters.
        state = loaded_pz_node.sample_parameters(num_samples=10)
        assert len(state["pznode"]) == 2
        assert len(state["pznode"]["x"]) == 10
        assert len(state["pznode"]["y"]) == 10
