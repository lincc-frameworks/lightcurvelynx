import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from lightcurvelynx.obstable.argus_obstable import ArgusHealpixObsTable


def _ra_dec_to_healpix(ra, dec, nside=32):
    """Convert RA and Dec to HEALPix indices."""
    hpx = HEALPix(nside=nside, order="nested", frame="icrs")
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return hpx.skycoord_to_healpix(coords)


def test_create_argus_obstable():
    """Create a minimal ArgusHealpixObsTable object and perform basic queries."""
    ra = np.array([15.0, 30.0, 15.0, 0.0, 15.0, 30.0])
    dec = np.array([10.0, -5.0, 10.0, 5.0, 10.0, -5.0])
    healpix = _ra_dec_to_healpix(ra, dec)
    values = {
        "epoch": 59000.0 + np.arange(len(ra)),
        "ra": ra,
        "dec": dec,
        "zp": np.ones_like(ra),
        "nside": np.full_like(ra, 32),
    }
    pdf = pd.DataFrame(values, index=healpix)

    # We fail if we are missing a healpix column or index.
    with pytest.raises(ValueError):
        _ = ArgusHealpixObsTable(pdf)

    pdf.index.name = "healpix"
    ops_data = ArgusHealpixObsTable(pdf)
    assert len(ops_data) == 6
    assert ops_data.nside == 32
    assert ops_data.depth == 5

    # We have all the attributes set at their default values.
    assert ops_data.survey_values["pixel_scale"] == 1.0
    assert ops_data.survey_values["radius"] == 52.0
    assert ops_data.survey_values["read_noise"] == 1.4
    assert ops_data.survey_values["survey_name"] == "Argus"

    # Check that we can extract the time bounds.
    t_min, t_max = ops_data.time_bounds()
    assert t_min == 59000.0
    assert t_max == 59005.0

    # We can query which columns the LSSTObsTable has by new or old name.
    assert "ra" in ops_data
    assert "dec" in ops_data
    assert "time" in ops_data
    assert "zp" in ops_data
    assert "healpix" in ops_data  # We move the index to a column
    assert "filter" in ops_data  # We add a default filter column

    # We can access columns directly as though it was a table.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["epoch"])

    # Check that we can determine which times a point is seen.
    assert np.array_equal(ops_data.range_search(15.0, 10.0), [0, 2, 4])
    assert np.array_equal(ops_data.range_search(30.0, -5.0), [1, 5])
    assert np.array_equal(ops_data.range_search(0.0, 5.0), [3])

    # Far away points are not seen.
    assert np.array_equal(ops_data.range_search(100.0, 80.0), [])
    assert np.array_equal(ops_data.range_search(15.0, -0.0), [])

    # Very close points fall in the same bucket, since they fall within an observed healpix pixel.
    assert np.array_equal(ops_data.range_search(15.00001, 9.99999), [0, 2, 4])
    assert np.array_equal(ops_data.range_search(29.999, -4.999), [1, 5])

    # We can filter the range search by time.
    assert np.array_equal(ops_data.range_search(15.0, 10.0, t_min=59001.0), [2, 4])
    assert np.array_equal(ops_data.range_search(15.0, 10.0, t_max=59001.0), [0])
    assert np.array_equal(ops_data.range_search(15.0, 10.0, t_min=59001.0, t_max=59003.0), [2])

    # We can do a vectorized range search.
    matches = ops_data.range_search(np.array([15.0, 30.0]), np.array([10.0, -5.0]))
    assert len(matches) == 2
    assert np.array_equal(matches[0], [0, 2, 4])
    assert np.array_equal(matches[1], [1, 5])

    # Check we can build a MOC.
    assert ops_data.build_moc() is not None


def test_create_argus_obstable_from_dict():
    """Create a ArgusHealpixObsTable object from a dictionary instead of a DataFrame."""
    ra = np.array([15.0, 30.0, 15.0, 0.0, 15.0, 30.0])
    dec = np.array([10.0, -5.0, 10.0, 5.0, 10.0, -5.0])
    healpix = _ra_dec_to_healpix(ra, dec)
    values = {
        "epoch": 59000.0 + np.arange(len(ra)),
        "ra": ra,
        "dec": dec,
        "zp": np.ones_like(ra),
        "healpix": healpix,
        "nside": np.full_like(ra, 32),
    }

    table = ArgusHealpixObsTable(values)
    assert len(table) == 6


def test_create_argus_obstable_alternate():
    """Create a ArgusHealpixObsTable object with healpix as a column instead of index."""
    ra = np.array([15.0, 30.0, 15.0, 0.0, 15.0, 30.0])
    dec = np.array([10.0, -5.0, 10.0, 5.0, 10.0, -5.0])
    healpix = _ra_dec_to_healpix(ra, dec)
    values = {
        "epoch": 59000.0 + np.arange(len(ra)),
        "ra": ra,
        "dec": dec,
        "zp": np.ones_like(ra),
        "healpix": healpix,  # We move the index to a column
    }
    pdf = pd.DataFrame(values)

    # We fail without nside information.
    with pytest.raises(ValueError):
        _ = ArgusHealpixObsTable(pdf)

    table = ArgusHealpixObsTable(pdf, nside=32)
    assert len(table) == 6

    # We fail if we have inconsistent nside information.
    pdf["nside"] = np.array([32, 32, 16, 32, 32, 32])
    with pytest.raises(ValueError):
        _ = ArgusHealpixObsTable(pdf)


def test_argus_obstable_footprint():
    """We cannot use a footprint for the argus obstable since it is already healpix."""
    ra = np.array([15.0, 30.0, 15.0, 0.0, 15.0, 30.0])
    dec = np.array([10.0, -5.0, 10.0, 5.0, 10.0, -5.0])
    healpix = _ra_dec_to_healpix(ra, dec)
    values = {
        "epoch": 59000.0 + np.arange(len(ra)),
        "ra": ra,
        "dec": dec,
        "zp": np.ones_like(ra),
        "healpix": healpix,  # We move the index to a column
    }
    pdf = pd.DataFrame(values)
    table = ArgusHealpixObsTable(pdf, nside=32)
    assert not table.uses_footprint()

    # We fail if we even call set_detector_footprint.
    with pytest.raises(NotImplementedError):
        table.set_detector_footprint(None)


def test_argus_obstable_noise():
    """Test that we can automatically compute the zero point for a ArgusHealpixObsTable
    and use that to compute the bandflux_error
    """
    ra = np.array([15.0, 30.0, 15.0, 0.0, 15.0, 30.0])
    dec = np.array([10.0, -5.0, 10.0, 5.0, 10.0, -5.0])
    healpix = _ra_dec_to_healpix(ra, dec)
    values = {
        "epoch": 59000.0 + np.arange(len(ra)),
        "ra": ra,
        "dec": dec,
        "healpix": healpix,
        # To derive zeropoint we need dark_electrons, exptime, seeing
        "dark_electrons": np.full_like(ra, 0.405696),
        "exptime": np.full_like(ra, 60.0),
        "limmag": np.full_like(ra, 20.569349),
        "seeing": np.full_like(ra, 0.804956),
        "sky_electrons": np.full_like(ra, 243.357877),
    }
    pdf = pd.DataFrame(values)
    table = ArgusHealpixObsTable(pdf, nside=32)

    assert "zp" in table
    assert np.all(table["zp"] > 15.0)

    # Check that we can compute the bandflux error for a given bandflux.
    bandfluxes = np.array([1000.0, 500.0, 200.0])
    index = np.array([0, 2, 4])
    bandflux_err = table.bandflux_error_point_source(bandflux=bandfluxes, index=index)
    assert len(bandflux_err) == len(bandfluxes)
    assert np.all(bandflux_err > 0.0)
