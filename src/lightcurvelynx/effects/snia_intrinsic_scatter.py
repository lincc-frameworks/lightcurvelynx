"""Intrinsic scatter for SN Ia"""

import numpy as np
import sncosmo
from citation_compass import cite_inline

from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc

# C11 model constants from Table 14.2, Chotard PhD thesis (2011) via SNANA sntools_genSmear.c.
# 6 bands: v(2500), U(3560), B(4390), V(5490), R(6545), I(8045) Angstroms.
# The v band (first row/column) is uncorrelated with others (OPT_farUV=0 default).
_C11_COVARIANCE = np.array(
    [
        [+1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        [0.000000, +1.000000, -0.118516, -0.768635, -0.908202, -0.219447],
        [0.000000, -0.118516, +1.000000, +0.570333, -0.238470, -0.888611],
        [0.000000, -0.768635, +0.570333, +1.000000, +0.530320, -0.399538],
        [0.000000, -0.908202, -0.238470, +0.530320, +1.000000, +0.490134],
        [0.000000, -0.219447, -0.888611, -0.399538, +0.490134, +1.000000],
    ]
)
_C11_DIAG = np.array([0.5900, 0.06001, 0.040034, 0.050014, 0.040017, 0.080007])
_C11_KNOT_WAVELENGTHS = np.array([2500.0, 3560.0, 4390.0, 5490.0, 6545.0, 8045.0])
_C11_COV_SCALE = 1.3

# Precomputed covariance matrix at the 6 knot wavelengths
_C11_COV_KNOTS = _C11_COVARIANCE * np.outer(_C11_DIAG, _C11_DIAG) * _C11_COV_SCALE

_DEFAULT_COH_SIGMA = 0.1
_DEFAULT_COH_SIGMA_G10 = 0.0
_DEFAULT_COH_SIGMA_C11 = 0.0


class SNIaIntrinsicScatter(EffectModel):
    """An effect model for intrinsic scatter in SN Ia.

    Attributes
    ----------
    modelpars : dict
        A dictionary of model parameters, which must include the key "modelname" with value
        "COH", "G10", or "C11". Additional parameters may be included depending on the modelname.
    interp_method : str
        The interpolation method to use for the G10 and C11 models.
        Must be one of "sine", "linear", "pchip", or "cubic". Default is "sine".
    """

    def __init__(self, modelpars, interp_method="sine", **kwargs):
        super().__init__(**kwargs)
        self.modelpars = modelpars
        if interp_method not in ("sine", "linear", "pchip", "cubic"):
            raise ValueError(
                f"interp_method must be 'sine', 'linear', 'pchip', or 'cubic', got '{interp_method}'"
            )
        self.rest_frame = True
        self.interp_method = interp_method
        self.add_effect_parameter(
            "snia_scatter_seed",
            NumpyRandomFunc("integers", low=0, high=2**32 - 1),
        )

    def _get_g10_color_dispersion(self, sourcename="salt3"):
        """Returns a callable color dispersion function loaded from the sncosmo source.

        Parameters
        ----------
        sourcename : str
            The name of the sncosmo source to use for the color dispersion function. Default is "salt3".

        """
        source = sncosmo.get_source(sourcename)
        return source._colordisp

    def _get_g10_node_wavelengths(self, modelpars):
        """Return node wavelengths for G10 scatter interpolation.

        Parameters
        ----------
        modelpars : dict
            The model parameters dictionary, which may contain the key "interp_wave_interval".
            ``interp_wave_interval="C11"`` (default) uses the 6 C11 knot wavelengths.
            ``interp_wave_interval=<float>`` generates evenly spaced nodes from
            2000 to 9200 Å at that spacing (in Angstroms).
        """
        interval = modelpars.get("interp_wave_interval", "C11")
        if interval == "C11":
            return _C11_KNOT_WAVELENGTHS
        return np.arange(2000.0, 9200.0 + float(interval) / 2, float(interval))

    def _interp(self, node_waves, node_values, wavelengths):
        """Dispatch to the selected interpolation method.
        Parameters
        ----------
        node_waves : np.ndarray
            Wavelengths of the nodes, shape (K,), must be sorted ascending.
        node_values : np.ndarray
            Scatter values at each node, shape (K,).
        wavelengths : np.ndarray
            Target wavelength grid, shape (N,).
        Returns
        -------
        np.ndarray
            Interpolated scatter values, shape (N,).
        """
        if self.interp_method == "sine":
            return self._sine_interp(node_waves, node_values, wavelengths)
        if self.interp_method == "pchip":
            return self._pchip_interp(node_waves, node_values, wavelengths)
        if self.interp_method == "cubic":
            return self._cubic_interp(node_waves, node_values, wavelengths)
        if self.interp_method == "linear":
            return self._linear_interp(node_waves, node_values, wavelengths)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interp_method}")

    def _linear_interp(self, node_waves, node_values, wavelengths):
        """Linear interpolation between nodes, clamped to edge values outside the range.
        Parameters
        ----------
        node_waves : np.ndarray
            Wavelengths of the nodes, shape (K,), must be sorted ascending.
        node_values : np.ndarray
            Scatter values at each node, shape (K,).
        wavelengths : np.ndarray
            Target wavelength grid, shape (N,).
        Returns
        -------
        np.ndarray
            Interpolated scatter values, shape (N,).
        """
        return np.interp(wavelengths, node_waves, node_values)

    def _pchip_interp(self, node_waves, node_values, wavelengths):
        """PCHIP interpolation — monotone-preserving cubic Hermite, no overshoot.
        Parameters
        ----------
        node_waves : np.ndarray
            Wavelengths of the nodes, shape (K,), must be sorted ascending.
        node_values : np.ndarray
            Scatter values at each node, shape (K,).
        wavelengths : np.ndarray
            Target wavelength grid, shape (N,).
        Returns
        -------
        np.ndarray
            Interpolated scatter values, shape (N,).
        """
        from scipy.interpolate import PchipInterpolator

        lam = np.clip(wavelengths, node_waves[0], node_waves[-1])
        return PchipInterpolator(node_waves, node_values)(lam)

    def _cubic_interp(self, node_waves, node_values, wavelengths):
        """Cubic spline interpolation — C² smooth, may overshoot between nodes.
        Parameters
        ----------
        node_waves : np.ndarray
            Wavelengths of the nodes, shape (K,), must be sorted ascending.
        node_values : np.ndarray
            Scatter values at each node, shape (K,).
        wavelengths : np.ndarray
            Target wavelength grid, shape (N,).
        Returns
        -------
        np.ndarray
            Interpolated scatter values, shape (N,).
        """
        from scipy.interpolate import CubicSpline

        lam = np.clip(wavelengths, node_waves[0], node_waves[-1])
        return CubicSpline(node_waves, node_values)(lam)

    def _sine_interp(self, node_waves, node_values, wavelengths):
        """Sine-interpolate node values onto a wavelength grid (SNANA's interp_SINFUN).

        Between adjacent nodes k and k+1:
            scatter(λ) = z_k cos²(θ/2) + z_{k+1} sin²(θ/2),  θ = π(λ−λ_k)/(λ_{k+1}−λ_k)

        Outside the node range the edge values are used.

        Parameters
        ----------
        node_waves : np.ndarray
            Wavelengths of the nodes, shape (K,), must be sorted ascending.
        node_values : np.ndarray
            Scatter values at each node, shape (K,).
        wavelengths : np.ndarray
            Target wavelength grid, shape (N,).

        Returns
        -------
        np.ndarray
            Interpolated scatter values, shape (N,).
        """
        out = np.empty(len(wavelengths))
        out[wavelengths <= node_waves[0]] = node_values[0]
        out[wavelengths >= node_waves[-1]] = node_values[-1]

        mask = (wavelengths > node_waves[0]) & (wavelengths < node_waves[-1])
        lam = wavelengths[mask]
        k = np.clip(np.searchsorted(node_waves, lam, side="right") - 1, 0, len(node_waves) - 2)
        theta = np.pi * (lam - node_waves[k]) / (node_waves[k + 1] - node_waves[k])
        out[mask] = node_values[k] * np.cos(theta / 2) ** 2 + node_values[k + 1] * np.sin(theta / 2) ** 2
        return out

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        **kwargs,
    ):
        """Apply the effect to observations (flux_density values).

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD). Not used for this effect.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        **kwargs : `dict`, optional
           Any additional keyword arguments. Pass ``modelpars`` to override
           the instance-level modelpars for this call.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        modelpars = {**self.modelpars, **kwargs.get("modelpars", {})}
        rng = np.random.default_rng(kwargs.get("snia_scatter_seed"))

        if modelpars["modelname"] == "COH":
            # Coherent scatter: one magnitude shift for all wavelengths and independent of time.
            sigma = modelpars.get("sigma", _DEFAULT_COH_SIGMA)
            scatter = rng.normal(0, sigma)
            return flux_density * np.power(10, -0.4 * scatter)

        if modelpars["modelname"] == "G10":
            # Chromatic scatter: draw at node wavelengths and interpolate using SALT-like color dispersion.
            # plus a coherent component. Both drawn once per SN, broadcast over epochs.
            g10_colordisp = self._get_g10_color_dispersion(modelpars.get("sourcename", "salt3"))
            node_waves = self._get_g10_node_wavelengths(modelpars)
            node_sigma = g10_colordisp(node_waves)
            node_draws = rng.normal(0, node_sigma)
            scatter_chrom = self._interp(node_waves, node_draws, wavelengths)
            coh_sigma = modelpars.get("coh_sigma", _DEFAULT_COH_SIGMA_G10)
            scatter = rng.normal(0, coh_sigma) + scatter_chrom  # shape (N,)
            # Cite G10 source
            cite_inline("Guy et al. (2010)", "https://doi.org/10.1051/0004-6361/201014468")
            cite_inline("Kenworthy et al. (2021)", "https://doi.org/10.3847/1538-4357/ac30d8")

            return flux_density * np.power(10, -0.4 * scatter[np.newaxis, :])

        if modelpars["modelname"] == "C11":
            # Correlated chromatic scatter: draw 6 correlated values from the C11 covariance
            # at the knot wavelengths via Cholesky, then sine-interpolate. Same as G10 but
            # with inter-band correlations from Chotard et al. (2011).
            # coh_sigma adds a gray floor; C11 models only chromatic scatter so a coherent
            # component must be added separately (matching SNANA's COH+C11 combination).
            node_draws = rng.multivariate_normal(np.zeros(6), _C11_COV_KNOTS)
            scatter_chrom = self._interp(_C11_KNOT_WAVELENGTHS, node_draws, wavelengths)
            coh_sigma = modelpars.get("coh_sigma", _DEFAULT_COH_SIGMA_C11)
            scatter = rng.normal(0, coh_sigma) + scatter_chrom
            # Cite C11 source
            cite_inline(
                "Chotard et al. (2011)", "PhD thesis, University Claude Bernard Lyon, 1, Lyon, France"
            )
            return flux_density * np.power(10, -0.4 * scatter[np.newaxis, :])

        raise ValueError(f"Unknown intrinsic scatter model: {modelpars['modelname']}")
