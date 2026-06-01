"""This file includes tests for combinations of models and settings that will not be used
in production, but help stress test the underlying machinery and make sure that the models
are working as expected in a variety of scenarios.
"""

import numpy as np
from lightcurvelynx.base_models import StateExpansionNode
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.physical_model import SEDModel


class _DuplicatingObjectModel(SEDModel):
    """A model that creates two nearby paired objects - one with the given brightness
    and one with half the brightness. This is meant to test the ExpansionNode and the
    ability to change the number of samples during the sampling process.

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
        super().__init__(**kwargs)

        # Second, we need to create 'base' versions of the parameter that will vary
        # per duplicate. This allows us to save the original value and sample offsets from it.
        self.add_parameter("base_brightness", brightness, description="Base brightness")
        self.add_parameter("base_ra", ra, description="Base RA")
        self.add_parameter("base_dec", dec, description="Base Dec")

        # Third, we add an expansion parameter that will create the duplicates.
        # This needs to be done AFTER the base parameters are set.
        self.add_parameter(
            "number_of_duplicates",
            StateExpansionNode(repeats=number_of_duplicates),
            description="The number of duplicates to create for each sample.",
        )

        # Finally, we override the parameters that will vary per duplicate.
        self.set_parameter(
            "ra",
            NumpyRandomFunc("normal", loc=self.base_ra, scale=0.01),
            description="Base RA with small random offset",
        )
        self.set_parameter(
            "dec",
            NumpyRandomFunc("normal", loc=self.base_dec, scale=0.01),
            description="Base Dec with small random offset",
        )
        self.add_parameter(
            "brightness",
            NumpyRandomFunc("normal", loc=self.base_brightness, scale=0.01),
            description="Base Brightness with small random offset",
        )

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)
        return np.full((len(times), len(wavelengths)), params["brightness"])


def test_duplicate_object_model():
    """Test that we can create a _DuplicateObjectModel."""
    model1 = _DuplicatingObjectModel(
        ra=GivenValueList([0.0, 45.0, 90.0]),
        dec=GivenValueList([0.0, -10.0, 10.0]),
        brightness=10.0,
        number_of_duplicates=2,
        node_label="test_model",
    )
    state = model1.sample_parameters(num_samples=3)
    assert state.num_samples == 6
