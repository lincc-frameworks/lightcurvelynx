"""This file includes tests for combinations of models and settings that will not be used
in production, but help stress test the underlying machinery and make sure that the models
are working as expected in a variety of scenarios.
"""

import numpy as np
from lightcurvelynx.base_models import StateExpansionNode
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
