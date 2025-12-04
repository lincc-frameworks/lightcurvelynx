"""An effect that adds microlensing magnification."""

import numpy as np
from citation_compass import CiteClass

from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.math_nodes.given_sampler import BinarySampler


class Microlensing(EffectModel, CiteClass):
    """A simple microlensing effect that can be applied to basic models. For more
    complex models, such as those blended by the microlensing event, it is recommended
    to create a new physical model.

    This model is a pure Paczynski/point-source point-lens (PSPL) model, without any
    additional effects. They can be added later. At the moment no values are parametrized,
    but they can be in the future.

    Values that should be parametrized are:
    * lens_mass (M_L)
    * lens_distance (D_L)
    * source_distance (D_S)
    * relative_proper_motion (mu_rel, mu_LS)
    * source_radius (R_S)

    References
    ----------
    * V. Bozza, MNRAS 408 (2010) 2188: general algorithm for binary lensing;
    * V. Bozza, E. Bachelet, F. Bartolic, T. Heintz, A. Hoag, M. Hundertmark, MNRAS 479 (2018) 5157:
    BinaryMag2 function, Extended-Source-Point-Lens methods;
    * V. Bozza, E. Khalouei and E. Bachelet, MNRAS 505 (2021) 126: astrometry, generalized limb darkening,
    Keplerian orbital motion;
    * V. Bozza, v. Saggese, G. Covone, P. Rota & J. Zhang, A&A 694 (2025) 219: multiple lenses.

    Attributes
    ----------
    VBM : VBMicrolensing
        The microlensing model.

    Parameters
    ----------
    microlensing_t0 : parameter
        The time of the microlensing peak, in days.
    u_0 : parameter
        Impact parameter, distance from the source at time
        of peak in Einstein radii.
    t_E : parameter
        Einstein timescale, in days.
    probability : float, optional
        The probability of the microlensing event occurring. Default: 1.0
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, microlensing_t0, u_0, t_E, probability=1.0, **kwargs):
        super().__init__(**kwargs, rest_frame=False)
        self.add_effect_parameter("microlensing_t0", microlensing_t0)
        self.add_effect_parameter("u_0", u_0)
        self.add_effect_parameter("t_E", t_E)

        # Add a parameter that indicates whether or not we apply microlensing that
        # is drawn from a distribution with the given probability.
        if probability < 1.0:
            self.add_effect_parameter("apply_microlensing", BinarySampler(probability))
        else:
            # Don't bother to add another node if we will always apply it.
            self.add_effect_parameter("apply_microlensing", True)

        # Create the microlensing model once.
        try:
            import VBMicrolensing
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "VBMicrolensing package is not installed be default. To use the microlensing effect, "
                "please install it. For example, you can install it with `pip install VBMicrolensing`."
            ) from err
        self.VBM = VBMicrolensing.VBMicrolensing()

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        *,
        apply_microlensing=None,
        microlensing_t0=None,
        u_0=None,
        t_E=None,
        **kwargs,
    ):
        """Apply the effect to observations (flux_density values).

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD).
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms). Not used for this effect.
        apply_microlensing : bool
            Whether or not to apply microlensing.
        microlensing_t0 : float
            The time of the microlensing peak, in days.
        u_0 : float
            Impact parameter, distance from the source at time
            of peak in Einstein radii.
        t_E : float
            Einstein timescale, in days.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        # Skip the microlensing application if needed.
        if not apply_microlensing:
            return flux_density

        if microlensing_t0 is None:  # pragma: no cover
            raise ValueError("microlensing_t0 must be provided")
        if u_0 is None:  # pragma: no cover
            raise ValueError("u_0 must be provided")
        if t_E is None:  # pragma: no cover
            raise ValueError("t_E must be provided")

        # Array of parameters
        # Note that VBMicrolensing requires some parameters
        # in log space, as per example:
        # https://github.com/valboz/VBMicrolensing/blob/main/docs/python/LightCurves.md
        pr = [np.log(u_0), np.log(t_E), microlensing_t0]

        # Calculates the PSPL magnification at different times with parameters in pr
        # Attention: VBMicrolensing assumes that times is an np.array or a list.
        vbm_results = self.VBM.PSPLLightCurve(pr, times)

        # array of magnifications at each time in time_stamp
        magnifications = np.asarray(vbm_results[0])

        flux_density = flux_density * magnifications[:, np.newaxis]
        return flux_density

    def apply_bandflux(
        self,
        bandfluxes,
        *,
        apply_microlensing=None,
        microlensing_t0=None,
        u_0=None,
        t_E=None,
        times=None,
        filters=None,
        **kwargs,
    ):
        """Apply the effect to band fluxes.

        Parameters
        ----------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD).
        filters : numpy.ndarray, optional
            A length N array of filters. If not provided, the effect is applied to all
            band fluxes.
        apply_microlensing : bool
            Whether or not to apply microlensing.
        microlensing_t0 : float
            The time of the microlensing peak, in days.
        u_0 : float
            Impact parameter, distance from the source at time
            of peak in Einstein radii.
        t_E : float
            Einstein timescale, in days.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes after the effect is applied (in nJy).
        """
        # If we do not apply microlensing, we can skip the entire computation.
        if not apply_microlensing:
            return bandfluxes

        if microlensing_t0 is None:  # pragma: no cover
            raise ValueError("microlensing_t0 must be provided")
        if u_0 is None:  # pragma: no cover
            raise ValueError("u_0 must be provided")
        if t_E is None:  # pragma: no cover
            raise ValueError("t_E must be provided")

        # Array of parameters
        # Note that VBMicrolensing requires some parameters
        # in log space, as per example:
        # https://github.com/valboz/VBMicrolensing/blob/main/docs/python/LightCurves.md
        pr = [np.log(u_0), np.log(t_E), microlensing_t0]

        # Calculates the PSPL magnification at different times with parameters in pr
        # Attention: VBMicrolensing assumes that times is an np.array or a list.
        vbm_results = self.VBM.PSPLLightCurve(pr, times)

        # array of magnifications at each time in time_stamp
        bandfluxes = bandfluxes * np.asarray(vbm_results[0])

        return bandfluxes
