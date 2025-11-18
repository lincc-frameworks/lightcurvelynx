from abc import ABC, abstractmethod

import numpy as np

from lightcurvelynx.astro_utils.black_body import black_body_luminosity_density_per_solid
from lightcurvelynx.consts import ANGSTROM_TO_CM, CGS_FNU_UNIT_TO_NJY, PARSEC_TO_CM
from lightcurvelynx.models.periodic_model import PeriodicModel


class PeriodicVariableStar(PeriodicModel, ABC):
    """A model for a periodic variable star.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * period - The period of the source, in days. [from PeriodicModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]

    Attributes
    ----------
    period : float
        The period of the source, in days.
    t0 : float
        The t0 of the zero phase, date. Could be date of the minimum or maximum light
        or any other reference time point.
    distance : float
        The distance to the source, in pc.
    """

    def __init__(self, period, **kwargs):
        super().__init__(period, **kwargs)
        if not self.has_valid_param("distance"):
            raise ValueError("Distance parameter is required for PeriodicVariableStar")

    def _evaluate_phases(self, phases, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object, as a function of phase.

        Parameters
        ----------
        phases : numpy.ndarray
            A length T array of phases, in the range [0, 1].
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths, in angstroms.
        graph_state : dict, optional
            A given setting of all the parameters and their values.
        **kwargs : dict, optional
              Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values, in nJy.
        """
        distance_cm = self.get_param(graph_state, "distance") * PARSEC_TO_CM
        wavelengths_cm = wavelengths * ANGSTROM_TO_CM
        dl_dnu_domega_cgs = self._dl_dnu_domega_phases(phases, wavelengths_cm, graph_state, **kwargs)
        return dl_dnu_domega_cgs / np.square(distance_cm) * CGS_FNU_UNIT_TO_NJY

    @abstractmethod
    def _dl_dnu_domega_phases(self, phases, wavelengths_cm, graph_state, **kwargs):
        r"""Draw effect-free luminosity density per unit solid angle, as a function of phase.

        It is dL / d \nu / d \Omega, so the units are erg / s / Hz / sr.
        Sometimes it is called "observed luminosity density".
        L_nu = \int this_value d \Omega.
        For isotropic source L_nu = 4 \pi \int this_value d \Omega.
        Bolometric luminosity is \int this_value d \nu d \Omega.
        Flux density (static Universe) F_nu = this_value / distance^2.

        Parameters
        ----------
        phases : numpy.ndarray
            A length T array of phases, in the range [0, 1].
        wavelengths_cm : numpy.ndarray, optional
            A length N array of wavelengths in cm.
        graph_state : dict, optional
            A given setting of all the parameters and their values.
        **kwargs : dict, optional
              Any additional keyword arguments.

        Returns
        -------
        luminosity_density_per_solid_angle : numpy.ndarray
            A length T x N matrix of luminosity density per unit solid angle values.
            Units are CGS, erg/s/Hz/steradian.
        """
        raise NotImplementedError()  # pragma: no cover


class EclipsingBinaryStar(PeriodicVariableStar):
    """A toy model for a detached eclipsing binary star.

    It is assumed that the stars are spherical, SED is black-body,
    and the orbits are circular. t0 is the epoch of the primary minimum.
    No limb darkening, reflection, or other effects are included.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * inclination - The inclination of the orbit, in degrees.
      * major_semiaxis - The major semiaxis of the orbit, in AU.
      * period - The period of the source, in days. [from PeriodicModel]
      * primary_radius - The radius of the primary star, in solar radii.
      * primary_temperature - The effective temperature of the primary star, in kelvins.
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * secondary_radius - The radius of the secondary star, in solar radii.
      * secondary_temperature - The effective temperature of the secondary star, in kelvins.
      * t0 - The t0 of the zero phase, date. [from PeriodicModel]

    Attributes
    ----------
    period : float
        The period of the source, in days.
    major_semiaxis : float
        The major semiaxis of the orbit, in AU.
    inclination : float
        The inclination of the orbit, in degrees.
    primary_radius : float
        The radius of the primary star, in solar radii.
    secondary_radius : float
        The radius of the secondary star, in solar radii.
    primary_temperature : float
        The effective temperature of the primary star, in kelvins.
    secondary_temperature : float
        The effective temperature of the secondary star, in kelvins.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("major_semiaxis", description="The major semiaxis of the orbit, in AU.", **kwargs)
        self.add_parameter("inclination", description="The inclination of the orbit, in degrees.", **kwargs)
        self.add_parameter(
            "primary_radius", description="The radius of the primary star, in solar radii.", **kwargs
        )
        self.add_parameter(
            "secondary_radius", description="The radius of the secondary star, in solar radii.", **kwargs
        )
        self.add_parameter(
            "primary_temperature",
            description="The effective temperature of the primary star, in kelvins.",
            **kwargs,
        )
        self.add_parameter(
            "secondary_temperature",
            description="The effective temperature of the secondary star, in kelvins.",
            **kwargs,
        )

    def _dl_dnu_domega_phases(self, phases, wavelengths_cm, graph_state, **kwargs):
        """Draw effect-free luminosity density for this object, as a function of phase.

        Parameters
        ----------
        phases : numpy.ndarray
            A length T array of phases, in the range [0, 1].
        wavelengths_cm : numpy.ndarray, optional
            A length N array of wavelengths, in cm.
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
              Any additional keyword arguments.

        Returns
        -------
        luminosity_density : numpy.ndarray
            A length T x N matrix of luminosity density values.
            Output is in CGS units of erg/s/Hz/steradian.
        """
        phases = phases[:, None]

        # Extract the parameters we need.
        params = self.get_local_params(graph_state)
        primary_temperature = params["primary_temperature"]
        primary_radius = params["primary_radius"]
        secondary_temperature = params["secondary_temperature"]
        secondary_radius = params["secondary_radius"]
        major_semiaxis = params["major_semiaxis"]
        inclination = params["inclination"]

        primary_lum = black_body_luminosity_density_per_solid(
            primary_temperature, primary_radius, wavelengths_cm
        )
        secondary_lum = black_body_luminosity_density_per_solid(
            secondary_temperature, secondary_radius, wavelengths_cm
        )

        # Distance between the centers of the stars on the plane of the sky
        star_centers_distance = major_semiaxis * self._norm_star_center_distance(phases, inclination)

        overlap_area = self._circle_overlap_area(star_centers_distance, primary_radius, secondary_radius)

        # For phases around the main minimum (0) primary star is eclipsed
        secondary_star_closer = np.logical_or(phases < 0.25, phases > 0.75)
        primary_star_overlap_ratio = np.where(
            secondary_star_closer, overlap_area / (np.pi * primary_radius**2), 0.0
        )
        secondary_star_overlap_ratio = np.where(
            ~secondary_star_closer, overlap_area / (np.pi * secondary_radius**2), 0.0
        )

        primary_lum_eclipsed = primary_lum * (1.0 - primary_star_overlap_ratio)
        secondary_lum_eclipsed = secondary_lum * (1.0 - secondary_star_overlap_ratio)

        # Just in case, we clip negative values to zero
        total_lum = np.where(primary_lum_eclipsed >= 0, primary_lum_eclipsed, 0) + np.where(
            secondary_lum_eclipsed >= 0, secondary_lum_eclipsed, 0
        )

        return total_lum

    @staticmethod
    def _norm_star_center_distance(phase_fraction, inclination_degree):
        """Calculate the distance between the centers of the stars on the observer's plane.

        Parameters
        ----------
        phase_fraction : np.ndarray
            The phase of the orbit, in the range [0, 1].
        inclination_degree : float
            The inclination of the orbit, in degrees.

        Returns
        -------
        distance : np.ndarray
            The distance between the centers of the stars on the plane of the sky,
            normalized by the major semiaxis.
        """
        phase_radians = 2 * np.pi * phase_fraction
        inclination_radians = np.radians(inclination_degree)
        return np.hypot(np.cos(inclination_radians) * np.cos(phase_radians), np.sin(phase_radians))

    @staticmethod
    def _circle_overlap_area(d, r1, r2):
        """Calculate the area of overlap between two circles.

        Parameters
        ----------
        d : np.ndarray
            The distance between the centers of the circles.
        r1 : float
            The radius of the first circle.
        r2 : float
            The radius of the second circle.

        Returns
        -------
        overlap_area : np.ndarray
            The area of overlap between the two circles.
        """

        # We consider four points here: the centers of the circles and the points where the circles intersect
        # The intersection region is a union of two segments of the circles, while their intersection is
        # a quadrilateral. So the area of the intersection is the sum of the areas of the two segments minus
        # the area of the quadrilateral. The intersection region is symmetric over the line connecting
        # the centers of the circles, so we can consider only one half of it, and multiply the result by 2.
        # In this half region intersection is a triangle with sides d, r1, and r2.

        r1_sq = np.square(r1)
        r2_sq = np.square(r2)
        d_sq = np.square(d)

        # We mute warnings for division by zero and invalid values in arccos, as we handle them manually
        with np.errstate(divide="ignore", invalid="ignore"):
            # Angles between lines connecting the centers of the circles and the points of intersection.
            # These are halves of the intersection-center-intersection angles.
            alpha1 = np.arccos((d_sq + r1_sq - r2_sq) / (2 * r1 * d))
            alpha2 = np.arccos((d_sq + r2_sq - r1_sq) / (2 * r2 * d))

            # Area of circular segments, it is 1/2 r^2 angle, angle = 2 alpha, so area = r^2 alpha
            area1 = r1_sq * alpha1
            area2 = r2_sq * alpha2

            # Area of the intersection quadrilateral, twice triangular area which is 1/2 r d sin(alpha).
            # Should be symmetric over 1-2 index swap.
            triangle_area = r1 * d * np.sin(alpha1)

        area = area1 + area2 - triangle_area

        # Just in case, we clip negative values to zero
        area = np.where(area >= 0.0, area, 0.0)

        # Account for the total overlap
        area = np.where(d <= np.abs(r1 - r2), np.pi * np.square(np.minimum(r1, r2)), area)
        # Account for the no overlap
        area = np.where(d >= r1 + r2, 0, area)

        return area
