"""A general extinction effect that wraps multiple backend libraries including:
* dust_extinction (https://github.com/karllark/dust_extinction)
* sncosmo extinction (https://github.com/sncosmo/extinction)
"""

import importlib
from pkgutil import iter_modules

import astropy.units as u
import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.effects.effect_model import EffectModel


class _DustExtinctionWrapper(CiteClass):
    """A wrapper for the dust_extinction package.

    References
    ----------
    Gordon 2024, JOSS, 9(100), 7023.
    https://github.com/karllark/dust_extinction

    Attributes
    ----------
    ext_obj : object
        The extinction object from the dust_extinction library. This will
        have r_v set during creation.

    Parameters
    ----------
    model_name : str
        The name of the extinction object in the dust_extinction library.
    r_v : float, optional
        The extinction parameter R(V). Optional for some models.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, model_name=None, r_v=None, **kwargs):
        try:
            import dust_extinction  # noqa: F401
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The dust_extinction package is needed to use the DustExtinctionEffect. Please install it via"
                "`pip install dust_extinction` or `conda install conda-forge::dust_extinction`."
            ) from err

        # We scan all of the submodules in the dust_extinction package,
        # looking for a matching name.
        self.ext_obj = None
        for submodule in iter_modules(dust_extinction.__path__):
            ext_module = importlib.import_module(f"dust_extinction.{submodule.name}")
            if ext_module is not None and model_name in dir(ext_module):
                ext_class = getattr(ext_module, model_name)

                # dust_extinction models optionally take Rv as a parameter during creation.
                self.ext_obj = ext_class(Rv=r_v, **kwargs)
                break

        if self.ext_obj is None:
            supported_models = self.list_extinction_models()
            raise KeyError(
                f"Invalid dust extinction model '{model_name}'. Supported models are: {supported_models}. "
            )

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
                "The dust_extinction package is needed to use the DustExtinctionEffect. Please install it via"
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

    def apply(self, flux_density, times=None, wavelengths=None, ebv=None, **kwargs):
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

        # The extinction factor computed by dust_extinction is a multiplicative
        # factor to reduce the flux (<= 1 for all wavelengths).
        ext_factor = self.ext_obj.extinguish(wavelengths * u.angstrom, Ebv=ebv)
        return flux_density * ext_factor


class _SncosmoExtinctionWrapper(CiteClass):
    """A wrapper for the (sncosmo) extinction package.

    References
    ----------
    https://github.com/sncosmo/extinction
    Barbary et. al. 2017
    doi: 10.5281/zenodo.804966

    Attributes
    ----------
    model_name : str
        The model_name of the extinction model to use.
    extinction_fn : function
        The extinction function from the backend library.

    Parameters
    ----------
    model_name : str
        The name of the extinction object in the dust_extinction library.
    r_v : float, optional
        The extinction parameter R(V). Optional for some models.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, model_name=None, r_v=None, **kwargs):
        try:
            import extinction  # noqa: F401
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "The extinction package is needed to use the ExtinctionEffect. Please install it via"
                "`pip install extinction` or `conda install conda-forge::extinction`."
            ) from err

        # We scan all of the submodules in the extinction package,
        # looking for a matching name.
        if model_name not in dir(extinction):
            raise KeyError(f"Invalid extinction model '{model_name}' from extinction package.")
        self.extinction_fn = getattr(extinction, model_name)
        self.model_name = model_name

        # Check that we either have r_v set or that the model does not need it.
        if model_name == "fm07":
            if r_v is not None and np.abs(r_v - 3.1) > 1e-8:
                raise ValueError("The fm07 extinction model requires r_v to be fixed at 3.1.")
            self.r_v = 3.1
        elif isinstance(r_v, float):
            self.r_v = r_v
        else:  # pragma: no cover
            raise ValueError(f"The {model_name} extinction model requires a floating point r_v parameter.")

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

        # Compute A(V) from E(B-V) and R(V)
        a_v = ebv * self.r_v

        # Use the function to compute the extinction magnitudes. Note that the fm07 model
        # requires only a_v (r_v is fixed at 3.1).
        if self.model_name == "fm07":
            ext_mags = self.extinction_fn(wavelengths, a_v=a_v, unit="aa")
        else:
            ext_mags = self.extinction_fn(wavelengths, a_v=a_v, r_v=self.r_v, unit="aa")

        # Apply the extinction to the flux density.
        ext_factor = 10 ** (-0.4 * ext_mags)  # Convert mags to flux factor
        return flux_density * ext_factor[np.newaxis, :]  # Broadcast to all times


class ExtinctionEffect(EffectModel):
    """A general extinction effect model that supports multiple backend libraries.

    Attributes
    ----------
    model_name : str
        The model_name of the extinction model to use.
    ebv : parameter
        The setter (function) for the extinction parameter E(B-V).
    frame : str
        The frame for extinction. 'rest' or 'observer'.
    r_v : float, optional
        The value for the extinction parameter R(V) if needed by the backend model.
    backend : str
        The backend extinction library to use. One of 'dust_extinction' or 'extinction'.
        Default is 'dust_extinction'.

    Parameters
    ----------
    extinction_model : str
        The name of the extinction model to use.
    ebv : parameter
        The setter (function) for the extinction parameter E(B-V).
    r_v : float, optional
        The value for the extinction parameter R(V) if needed by the backend model.
    frame : str
        The frame for extinction. 'rest' or 'observer'.
    backend : str
        The backend extinction library to use. One of 'dust_extinction' or 'extinction'.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    _BACKEND_TO_WRAPPER = {
        "dust_extinction": _DustExtinctionWrapper,
        "extinction": _SncosmoExtinctionWrapper,
    }

    def __init__(
        self,
        extinction_model=None,
        *,
        ebv=None,
        frame=None,
        r_v=None,
        backend=None,
        **kwargs,
    ):
        self.model_name = extinction_model
        self.backend = backend
        self.r_v = r_v

        # Set up the effect parameters (ebv).
        super().__init__(**kwargs)
        self.add_effect_parameter("ebv", ebv)

        # Set the frame.
        if frame == "observer":
            self.rest_frame = False
        elif frame == "rest":
            self.rest_frame = True
        else:
            raise ValueError("frame must be 'observer' or 'rest'.")

        # Check the backend is one of the supported ones and is installed.
        if self.backend not in self._BACKEND_TO_WRAPPER:
            raise ValueError(f"backend must be one of {list(self._BACKEND_TO_WRAPPER.keys())}")
        if importlib.util.find_spec(self.backend) is None:  # pragma: no cover
            raise ImportError(
                f"The specified backend '{self.backend}' library is not installed. Please install it via"
                f"`pip install {self.backend}` or `conda install conda-forge::{self.backend}`."
            )

        # Load the correct extinction model.
        self._extinction_wrapper = self._BACKEND_TO_WRAPPER[self.backend](self.model_name, r_v=self.r_v)

    def __repr__(self):
        return f"ExtinctionEffect({self.backend}.{self.model_name}, r_v={self.r_v})"

    def __getstate__(self):
        """We override the default pickling behavior to handle the extinction model, since
        it may not be picklable.
        """
        if self.model_name is None or self.backend is None:  # pragma: no cover
            raise ValueError(
                "Both the model name and backend must be specified (as strings) in order to "
                "to be pickled and used with distributed computation."
            )

        # Return the state without the extinction model, since it may not be picklable.
        state = self.__dict__.copy()
        del state["_extinction_wrapper"]
        return state

    def __setstate__(self, state):
        """We override the default unpickling behavior to handle the extinction model, since
        it may not be picklable.
        """
        self.__dict__.update(state)
        if self.model_name is None or self.backend is None:  # pragma: no cover
            raise ValueError(
                "Both the model name and backend must be specified (as strings) in order to "
                "to be pickled and used with distributed computation."
            )

        self._extinction_wrapper = self._BACKEND_TO_WRAPPER[self.backend](self.model_name, r_v=self.r_v)

    def apply(self, flux_density, times=None, wavelengths=None, ebv=None, **kwargs):
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
        return self._extinction_wrapper.apply(
            flux_density,
            times=times,
            wavelengths=wavelengths,
            ebv=ebv,
            **kwargs,
        )
