"""A wrapper for applying extinction functions using the dust_extinction library.

Citation:
Gordon 2024, JOSS, 9(100), 7023.
https://github.com/karllark/dust_extinction
"""

import importlib
from pkgutil import iter_modules
from typing import Literal

import astropy.units as u
from citation_compass import CiteClass

from lightcurvelynx.effects.effect_model import EffectModel


class ExtinctionEffect(EffectModel, CiteClass):
    """A general dust extinction effect model.

    References
    ----------
    Gordon 2024, JOSS, 9(100), 7023.
    https://github.com/karllark/dust_extinction

    Attributes
    ----------
    extinction_model : function or str
        The extinction object from the dust_extinction library or its name.
        If a string is provided, the code will find a matching extinction
        function in the dust_extinction package and use that.
    ebv : parameter
        The setter (function) for the extinction parameter E(B-V).
    frame : str
        The frame for extinction. 'rest' or 'observer'.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(
        self, extinction_model=None, ebv=None, frame: Literal["observer"] | Literal["rest"] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.add_effect_parameter("ebv", ebv)

        if frame == "observer":
            self.rest_frame = False
        elif frame == "rest":
            self.rest_frame = True
        else:
            raise ValueError("frame must be 'observer' or 'rest'.")

        if isinstance(extinction_model, str):
            self._model_name = extinction_model
            extinction_model = ExtinctionEffect.load_extinction_model(extinction_model, **kwargs)
        else:
            self._model_name = None
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
        self.extinction_model = ExtinctionEffect.load_extinction_model(self._model_name)

    @staticmethod
    def list_extinction_models():
        """List the extinction models from the dust_extinction package
        (https://github.com/karllark/dust_extinction)

        Returns
        -------
        list of str
            A list of the names of the extinction models.
        """
        model_names = []

        try:
            import dust_extinction  # noqa: F401
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The dust_extinction package is needed to use the ExtinctionEffect. Please install it via"
                "`pip install dust_extinction` or `conda install conda-forge::dust_extinction`."
            ) from err

        # We scan all of the submodules in the dust_extinction package,
        # looking for classes with extinguish() functions.
        for submodule in iter_modules(dust_extinction.__path__):
            ext_module = importlib.import_module(f"dust_extinction.{submodule.name}")
            for entry_name in dir(ext_module):
                entry_obj = getattr(ext_module, entry_name)
                if hasattr(entry_obj, "extinguish"):
                    model_names.append(entry_name)
        return model_names

    @staticmethod
    def load_extinction_model(name, **kwargs):
        """Load the extinction model from the dust_extinction package
        (https://github.com/karllark/dust_extinction)

        Parameters
        ----------
        name : str
            The name of the extinction model to use.
        **kwargs : dict
            Any additional keyword arguments needed to create that argument.

        Returns
        -------
        ext_obj
            A extinction object.
        """
        try:
            import dust_extinction  # noqa: F401
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The dust_extinction package is needed to use the ExtinctionEffect. Please install it via"
                "`pip install dust_extinction` or `conda install conda-forge::dust_extinction`."
            ) from err

        # We scan all of the submodules in the dust_extinction package,
        # looking for a matching name.
        for submodule in iter_modules(dust_extinction.__path__):
            ext_module = importlib.import_module(f"dust_extinction.{submodule.name}")
            if ext_module is not None and name in dir(ext_module):
                ext_class = getattr(ext_module, name)
                return ext_class(**kwargs)
        raise KeyError(f"Invalid dust extinction model '{name}'")

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        ebv=None,
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
        ebv : float, optional
            The extinction parameter E(B-V). Raises an error if None is provided.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        if ebv is None:
            raise ValueError("ebv must be provided")
        if wavelengths is None:
            raise ValueError("wavelengths must be provided")

        return flux_density * self.extinction_model.extinguish(wavelengths * u.angstrom, Ebv=ebv)
