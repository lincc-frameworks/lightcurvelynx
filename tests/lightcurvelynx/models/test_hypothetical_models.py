"""This file includes tests for combinations of models and settings that will not be used
in production, but help stress test the underlying machinery and make sure that the models
are working as expected in a variety of scenarios.
"""

import numpy as np
from lightcurvelynx.base_models import FunctionNode, StateExpansionNode
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.basic_models import ConstantSEDModel


class _DuplicatingObjectModel(ConstantSEDModel):
    """A model that creates two nearby paired objects - with random offsets in position
    and brightness. This is meant to test the ExpansionNode and the ability to change the
    number of samples during the sampling process.

    Parameters
    ----------
    brightness : parameter
        The inherent brightness.
    number_of_duplicates : parameter
        The number of duplicates to create for each sample.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        ra=None,  # We include the values we are going to vary.
        dec=None,  # We include the values we are going to vary.
        brightness=None,  # We include the values we are going to vary.
        number_of_duplicates=None,
        **kwargs,
    ):
        # First, we call the SEDModel constructor first to set up the base parameters.
        # We use None for the parameters we will override later.
        super().__init__(ra=None, dec=None, brightness=None, **kwargs)

        # Second, we need to create 'base' versions of the parameter that will vary
        # per duplicate. This allows us to save the original value and sample offsets from it.
        self.add_parameter("base_brightness", brightness, description="Base brightness")
        self.add_parameter("base_ra", ra, description="Base RA")
        self.add_parameter("base_dec", dec, description="Base Dec")

        # Third, we add an expansion parameter that will create the duplicates.
        # This needs to be done AFTER the base parameters are set, because this triggers
        # the actual expansion of the samples.
        self.add_parameter(
            "number_of_duplicates",
            StateExpansionNode(repeats=number_of_duplicates).repeats,
            description="The number of duplicates to create for each sample.",
        )

        # We delete the original RA, dec, and brightness so we can override them.
        # This should only be done with EXTREME caution as it can cause difficult to debug
        # issues in the order that parameters are sampled and used.
        self.remove_parameter("ra")
        self.remove_parameter("dec")
        self.remove_parameter("brightness")

        # Now we create new versions of the RA, dec, and brightness parameters that will
        # be sampled from the base values with a small random offset.
        self.add_parameter(
            "ra",
            NumpyRandomFunc("normal", loc=self.base_ra, scale=0.01),
            description="Base RA with small random offset",
        )
        self.add_parameter(
            "dec",
            NumpyRandomFunc("normal", loc=self.base_dec, scale=0.01),
            description="Base Dec with small random offset",
        )
        self.add_parameter(
            "brightness",
            NumpyRandomFunc("normal", loc=self.base_brightness, scale=0.01),
            description="Base Brightness with small random offset",
        )


def test_duplicate_object_model():
    """Test that we can create a _DuplicateObjectModel."""
    model1 = _DuplicatingObjectModel(
        ra=GivenValueList([0.0, 45.0, 90.0]),
        dec=GivenValueList([0.0, -10.0, 10.0]),
        brightness=10.0,
        number_of_duplicates=GivenValueList([2, 3, 1]),
        node_label="test_model",
    )
    state = model1.sample_parameters(num_samples=3)
    assert state.num_samples == 6

    # We capture the base values for the original samples, and check that
    # they are repeated correctly.
    assert np.array_equal(state["test_model"]["base_ra"], [0.0, 0.0, 45.0, 45.0, 45.0, 90.0])
    assert np.array_equal(state["test_model"]["base_dec"], [0.0, 0.0, -10.0, -10.0, -10.0, 10.0])
    assert np.array_equal(state["test_model"]["base_brightness"], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    # The RA, dec, and brightness values should be close to the base values,
    # but with some random offset.
    assert not np.array_equal(state["test_model"]["ra"], state["test_model"]["base_ra"])
    assert np.allclose(state["test_model"]["ra"], state["test_model"]["base_ra"], atol=0.2)
    assert not np.array_equal(state["test_model"]["dec"], state["test_model"]["base_dec"])
    assert not np.array_equal(state["test_model"]["brightness"], state["test_model"]["base_brightness"])
    assert np.allclose(state["test_model"]["dec"], state["test_model"]["base_dec"], atol=0.2)
    assert np.allclose(state["test_model"]["brightness"], state["test_model"]["base_brightness"], atol=0.2)


class _GenerateOffsets(FunctionNode):
    """A function node that takes some parameters and generates offsets of the core physical model
    parameters for each duplicate. This is similar to what we would expect a strong lensing model
    to do (although that would be based on physics instead of random offsets).

    Parameters
    ----------
    splits : parameter
        A list of how many duplicates to create for each sample.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, splits, **kwargs):
        super().__init__(
            func=self._non_func,  # We will override compute() so the function doesn't matter.
            splits=splits,
            outputs=["subparameters"],
            **kwargs,
        )

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Overriden compute function that takes the splits parameter uses it to create
        a list of subparameterr dictionaries."""
        args = self._build_inputs(graph_state, **kwargs)

        # This is a linear function that creates a list of subparameter dictionaries based on the splits.
        # We could replace this with something systematic (like a strong lensing model).
        results = [None] * graph_state.num_samples
        for i in range(graph_state.num_samples):
            split = args["splits"][i]
            results[i] = {
                "dRA": np.random.normal(loc=0.0, scale=0.01, size=split),
                "dDec": np.random.normal(loc=0.0, scale=0.01, size=split),
                "dT0": np.arange(split) * 0.1,
            }
        self._save_results(results, graph_state)


def test_generate_offsets():
    """Test that we can create a _GenerateOffsets node and that it creates the expected output."""
    node = _GenerateOffsets(splits=GivenValueList([2, 3, 1]), node_label="test_offsets")
    state = node.sample_parameters(num_samples=3)
    assert state.num_samples == 3
    assert len(state["test_offsets"]["subparameters"]) == 3
    assert np.allclose(state["test_offsets"]["subparameters"][0]["dRA"], np.zeros(2), atol=0.1)
    assert np.allclose(state["test_offsets"]["subparameters"][0]["dDec"], np.zeros(2), atol=0.1)
    assert np.allclose(state["test_offsets"]["subparameters"][0]["dT0"], np.arange(2) * 0.1)
    assert np.allclose(state["test_offsets"]["subparameters"][1]["dRA"], np.zeros(3), atol=0.1)
    assert np.allclose(state["test_offsets"]["subparameters"][1]["dDec"], np.zeros(3), atol=0.1)
    assert np.allclose(state["test_offsets"]["subparameters"][1]["dT0"], np.arange(3) * 0.1)
    assert np.allclose(state["test_offsets"]["subparameters"][2]["dRA"], np.zeros(1), atol=0.1)
    assert np.allclose(state["test_offsets"]["subparameters"][2]["dDec"], np.zeros(1), atol=0.1)
    assert np.allclose(state["test_offsets"]["subparameters"][2]["dT0"], np.arange(1) * 0.1)


class _ApplyOffsetsNode(FunctionNode):
    """A function node that takes the subparameters generated by _GenerateOffsets and applies them
    to the base parameters to create new parameters for each duplicate.

    Parameters
    ----------
    base_ra : parameter
        The base RA for each sample.
    base_dec : parameter
        The base Dec for each sample.
    base_t0 : parameter
        The base T0 for each sample.
    subparameters : parameter
        The subparameters generated by _GenerateOffsets, which include the offsets to apply
        to the base parameters.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, base_ra, base_dec, base_t0, offsets, **kwargs):
        super().__init__(
            func=self._non_func,  # We will override compute() so the function doesn't matter.
            base_ra=base_ra,
            base_dec=base_dec,
            base_t0=base_t0,
            offsets=offsets,
            outputs=["ra", "dec", "t0"],
            **kwargs,
        )

        # Add the relevant information from the StateExpansionNode.
        self._expansion_node = StateExpansionNode(subparameters=offsets)
        self.add_parameter("org_inds", self._expansion_node.org_inds)
        self.add_parameter("sub_inds", self._expansion_node.sub_inds)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Overriden compute function that takes the offsets and applies them to the base parameters."""
        params = self.get_local_params(graph_state)
        added_params = self._expansion_node.get_local_params(graph_state)

        # At this point the graph state has already been expanded by the StateExpansionNode and
        # the delta columns reside in that node's parameter columns.
        ra = params["base_ra"] + added_params["dRA"]
        dec = params["base_dec"] + added_params["dDec"]
        t0 = params["base_t0"] + added_params["dT0"]

        self._save_results((ra, dec, t0), graph_state)


def test_apply_offsets():
    """Test that we can create an ApplyOffsets node and that it creates the expected output."""
    apply_node = _ApplyOffsetsNode(
        base_ra=GivenValueList([0.0, 45.0, 90.0, 120.0]),
        base_dec=GivenValueList([0.0, -10.0, 10.0, 20.0]),
        base_t0=GivenValueList([0.0, 1.0, 2.0, 3.0]),
        offsets=_GenerateOffsets(splits=GivenValueList([2, 3, 1, 2])).subparameters,
        node_label="test",
    )

    # Sample the first 3 cases.
    state = apply_node.sample_parameters(num_samples=3)
    assert state.num_samples == 6

    # The values should be identical to the given and the final values should be close, but with
    # a random offset.
    assert np.allclose(state["test"]["base_ra"], [0.0, 0.0, 45.0, 45.0, 45.0, 90.0], atol=1e-8)
    assert not np.allclose(state["test"]["ra"], [0.0, 0.0, 45.0, 45.0, 45.0, 90.0], atol=1e-8)
    assert np.allclose(state["test"]["ra"], [0.0, 0.0, 45.0, 45.0, 45.0, 90.0], atol=0.1)

    assert np.allclose(state["test"]["base_dec"], [0.0, 0.0, -10.0, -10.0, -10.0, 10.0], atol=1e-8)
    assert not np.allclose(state["test"]["dec"], [0.0, 0.0, -10.0, -10.0, -10.0, 10.0], atol=1e-8)
    assert np.allclose(state["test"]["dec"], [0.0, 0.0, -10.0, -10.0, -10.0, 10.0], atol=0.1)

    assert np.allclose(state["test"]["base_t0"], [0.0, 0.0, 1.0, 1.0, 1.0, 2.0], atol=1e-8)
    assert np.allclose(state["test"]["t0"], [0.0, 0.1, 1.0, 1.1, 1.2, 2.0], atol=0.1)
