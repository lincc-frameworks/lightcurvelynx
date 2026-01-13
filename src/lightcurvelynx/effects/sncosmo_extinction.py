"""A wrapper for applying extinction functions using the sncosmo extinction library.

Citation:
    https://github.com/sncosmo/extinction
    Barbary et. al. 2017
    doi: 10.5281/zenodo.804966
"""

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.effects.effect_model import EffectModel


class SncosmoExtinctionEffect(EffectModel, CiteClass):
    """A general dust extinction effect model.

    References
    ----------
    https://github.com/sncosmo/extinction
    Barbary et. al. 2017
    doi: 10.5281/zenodo.804966

    Attributes
    ----------
    extinction_model : function or str
        The extinction object from the sncosmo extinction library or its name.
        If a string is provided, the code will find a matching extinction
        function in the extinction package and use that.
    a_v : parameter
        The setter for the extinction parameter A(V).
    r_v : parameter
        The setter for the extinction parameter R(V).
    frame : str
        The frame for extinction. 'rest' or 'observer'.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        extinction_model=None,
        a_v=None,
        r_v=None,
        frame=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_effect_parameter("a_v", a_v)
        self.add_effect_parameter("r_v", r_v)

        if frame == "observer":
            self.rest_frame = False
        elif frame == "rest":
            self.rest_frame = True
        else:
            raise ValueError("frame must be 'observer' or 'rest'.")

        if isinstance(extinction_model, str):
            self._model_name = extinction_model
            extinction_model = self.load_extinction_model(extinction_model, **kwargs)
        else:
            # Save the function name so we can pickle it later.
            self._model_name = extinction_model.__name__
        self.extinction_model = extinction_model

    def __getstate__(self):
        """We override the default pickling behavior to handle the extinction model, since
        it may not be picklable.
        """
        if self._model_name is None:
            raise ValueError(
                "Extinction model must be specified as a string (of the model name) in order to "
                "to be pickled and used with distributed computation."
            )

        # Return the state without the extinction model, since it may not be picklable.
        state = self.__dict__.copy()
        del state["extinction_model"]
        return state

    def __setstate__(self, state):
        """We override the default unpickling behavior to handle the extinction model, since
        it may not be picklable.
        """
        self.__dict__.update(state)
        self.extinction_model = self.load_extinction_model(self._model_name)

    @staticmethod
    def load_extinction_model(name, **kwargs):
        """Load the extinction model from the extinction package
        (https://github.com/sncosmo/extinction)

        Parameters
        ----------
        name : str
            The name of the extinction model to use.
        **kwargs : dict
            Any additional keyword arguments needed to create that argument.

        Returns
        -------
        ext_obj
            A extinction function.
        """
        try:
            import extinction  # noqa: F401
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The extinction package is needed to use the ExtinctionEffect. Please install it via"
                "`pip install extinction` or `conda install conda-forge::extinction`."
            ) from err

        # We scan all of the submodules in the extinction package,
        # looking for a matching name.
        if name not in dir(extinction):
            raise KeyError(f"Invalid extinction model '{name}'")
        ext_fn = getattr(extinction, name)
        return ext_fn

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        a_v=None,
        r_v=None,
        **kwargs,
    ):
        """Apply the extinction effect to the flux density.

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD). Not used for this effect.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms). Not used for this effect.
        a_v : float, optional
            The extinction parameter A(V). Raises an error if None is provided.
        r_v : float, optional
            The extinction parameter R(V). Raises an error if None is provided
            and the function requires it.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        if a_v is None:
            raise ValueError("a_v must be provided")
        if r_v is None and self._model_name != "fm07":
            raise ValueError(f"r_v must be provided for model {self._model_name}")
        if wavelengths is None:
            raise ValueError("wavelengths must be provided")

        # Use the function to compute the extinction magnitudes. Note that the fm07 model
        # requires only a_v (r_v is fixed at 3.1).
        if self._model_name == "fm07":
            ext_mags = self.extinction_model(wavelengths, a_v=a_v, unit="aa")
        else:
            ext_mags = self.extinction_model(wavelengths, a_v=a_v, r_v=r_v, unit="aa")

        # Apply the extinction to the flux density.
        ext_factor = 10 ** (-0.4 * ext_mags)  # Convert mags to flux factor
        return flux_density * ext_factor[np.newaxis, :]  # Broadcast to all times
