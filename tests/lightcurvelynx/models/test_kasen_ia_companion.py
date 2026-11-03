import zipfile

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from lightcurvelynx.models import kasen_ia_companion
from lightcurvelynx.models.kasen_ia_companion import (
    KASEN_COMPANIONS,
    KasenIaCompanionModel,
    load_kasen_companion_grid,
)

# A small fake grid in the format of the prestocolor per-angle spectrum files.
_FAKE_TIMES = np.array([0.05, 0.15, 0.25, 0.35])
_FAKE_WAVES = np.array([51.0, 151.0, 251.0, 351.0, 451.0, 551.0])
# Unsorted, as in the original angle list.
_FAKE_ANGLES = np.array([167.1616, 91.4326, 12.8386])


def _fake_file_values(companion_idx):
    """The fake luminosity column (in units of 1e36 erg/s/A) as a (T, W, A) array."""
    rng = np.random.default_rng(companion_idx)
    values = rng.uniform(1.0, 1000.0, (len(_FAKE_TIMES), len(_FAKE_WAVES), len(_FAKE_ANGLES)))
    # Round to five significant digits, as the "%.4e" format in the files.
    return np.array([[[float(f"{v:.4e}") for v in row] for row in block] for block in values])


@pytest.fixture
def kasen_data_dir(tmp_path):
    """Create a fake prestocolor archive in a temporary data directory."""
    prefix = "prestocolor-0123abcd/FastTrWP/models/companions"
    with zipfile.ZipFile(tmp_path / "prestocolor.zip", "w") as zf:
        for c_idx, companion in enumerate(KASEN_COMPANIONS):
            values = _fake_file_values(c_idx)
            for t_idx, time in enumerate(_FAKE_TIMES):
                for a_idx, angle in enumerate(_FAKE_ANGLES):
                    lines = [
                        f"{wave:.4f} {value:.4e}"
                        for wave, value in zip(_FAKE_WAVES, values[t_idx, :, a_idx], strict=True)
                    ]
                    name = f"{prefix}/companion_{companion}/optical_t{time:06.2f}_I3.spec.{angle:.4f}"
                    zf.writestr(name, "\n".join(lines) + "\n")
            # A file with a different naming scheme that must be ignored.
            zf.writestr(f"{prefix}/companion_{companion}/optical_t0-1_I3.spec.12", "garbage\n")
    return tmp_path


def test_load_kasen_companion_grid(tmp_path, kasen_data_dir, monkeypatch):
    """Test that we download the archive once, convert it, and remove it."""
    calls = []

    def _fake_download(data_path, data_url, force_download=False):
        calls.append(data_url)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        (kasen_data_dir / "prestocolor.zip").rename(data_path)
        return True

    monkeypatch.setattr(kasen_ia_companion, "download_data_file_if_needed", _fake_download)
    data_dir = tmp_path / "cache"
    for c_idx, companion in enumerate(KASEN_COMPANIONS):
        grid = load_kasen_companion_grid(companion, data_dir=data_dir)
        np.testing.assert_allclose(grid["times"], _FAKE_TIMES)
        np.testing.assert_allclose(grid["wavelengths"], _FAKE_WAVES)
        np.testing.assert_allclose(grid["angles"], np.sort(_FAKE_ANGLES))

        # The luminosity is stored as (A, T, W) in units of 1e36 erg/s/A.
        expected = np.moveaxis(_fake_file_values(c_idx)[:, :, np.argsort(_FAKE_ANGLES)], -1, 0)
        assert grid["luminosity"].dtype == np.float32
        np.testing.assert_allclose(grid["luminosity"], expected, rtol=1e-7)

    assert calls == [kasen_ia_companion._PRESTOCOLOR_ZIP_URL]
    assert not (data_dir / "prestocolor.zip").exists()

    with pytest.raises(ValueError):
        _ = load_kasen_companion_grid("a1e10", data_dir=data_dir)


def test_kasen_matches_regular_grid_interpolator(kasen_data_dir):
    """Test that the interpolation is identical to the notebook's RegularGridInterpolator."""
    sort_idx = np.argsort(_FAKE_ANGLES)
    reference = RegularGridInterpolator(
        (_FAKE_TIMES, _FAKE_WAVES, _FAKE_ANGLES[sort_idx]),
        _fake_file_values(0)[:, :, sort_idx] * 1e36,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    model = KasenIaCompanionModel("a5e11", viewing_angle=90.0, t0=0.0, data_dir=kasen_data_dir)

    # Include the grid points and points outside of the grid in every dimension.
    rng = np.random.default_rng(100)
    times = np.concatenate([rng.uniform(-0.1, 0.5, 20), _FAKE_TIMES])
    waves = np.concatenate([rng.uniform(0.0, 600.0, 20), _FAKE_WAVES])
    tt, ww = np.meshgrid(times, waves, indexing="ij")
    for angle in np.concatenate([rng.uniform(5.0, 175.0, 10), _FAKE_ANGLES]):
        expected = reference((tt, ww, np.full_like(tt, angle)))
        # The tolerance accounts for the float32 storage of the grid.
        np.testing.assert_allclose(model.compute_luminosity(times, waves, angle), expected, rtol=1e-7)


def test_kasen_compute_sed_units(kasen_data_dir):
    """Test that the SED is the luminosity converted to nJy at the given distance."""
    t0 = 60000.0
    distance = 1.0e6  # pc
    model = KasenIaCompanionModel(
        "a2e12", viewing_angle=50.0, t0=t0, distance=distance, data_dir=kasen_data_dir
    )
    times = t0 + np.array([0.1, 0.2, 0.3])
    waves = np.array([100.0, 300.0, 500.0])
    sed = model.evaluate_sed(times, waves)

    lum = model.compute_luminosity(times - t0, waves, 50.0) * u.erg / u.s / u.AA
    flam = lum / (4.0 * np.pi * (distance * u.pc) ** 2)
    expected = (flam * (waves * u.AA) ** 2 / const.c).to_value(u.nJy)
    np.testing.assert_allclose(sed, expected, rtol=1e-12)


def test_kasen_invalid(kasen_data_dir):
    """Test that we raise errors for invalid inputs."""
    with pytest.raises(ValueError):
        _ = KasenIaCompanionModel("a2e13", viewing_angle=90.0, distance=10.0, data_dir=kasen_data_dir)

    # A model without a distance cannot compute fluxes.
    model = KasenIaCompanionModel("a2e13", viewing_angle=90.0, t0=0.0, data_dir=kasen_data_dir)
    with pytest.raises(ValueError):
        _ = model.evaluate_sed(np.array([0.2]), np.array([300.0]))
