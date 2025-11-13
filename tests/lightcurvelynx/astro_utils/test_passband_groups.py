import matplotlib

matplotlib.use("Agg")  # Suppress the plots during testing

from unittest.mock import patch

import numpy as np
import pytest
from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.models.sed_template_model import SEDTemplateModel
from sncosmo import Bandpass


def create_lsst_passband_group(passbands_dir, delta_wave=5.0, trim_quantile=None):
    """Helper function to create a PassbandGroup object for LSST passbands, using transmission tables
    included in the test data directory."""
    return PassbandGroup.from_preset(
        preset="LSST",
        table_dir=passbands_dir,
        delta_wave=delta_wave,
        trim_quantile=trim_quantile,
    )


def create_toy_passband_group(path, delta_wave=5.0, trim_quantile=None):
    """Helper function to create a toy PassbandGroup object for testing.

    Notable features of this toy passband group:
    - Three passbands: A, B, C
    - Each transmission table includes only three rows
    - Passbands A and B have overlapping wavelength ranges
    - Passbands A and C are defined with a 100 unit grid step, while passband B is defined with a 50 unit step
    """
    # Initialize requirements for PassbandGroup object
    survey = "TOY"
    filter_names = ["a", "b", "c"]
    table_dir = path / survey
    table_dir.mkdir()

    transmission_tables = {
        "a": "100 0.5\n200 0.75\n300 0.25\n",
        "b": "250 0.25\n300 0.5\n350 0.75\n",
        "c": "400 0.75\n500 0.25\n600 0.5\n",
    }
    for filter_name in filter_names:
        with open(table_dir / f"{filter_name}.dat", "w") as f:
            f.write(transmission_tables[filter_name])

    # Load the PassbandGroup object
    passbands = []
    for filter_name in filter_names:
        passbands.append(
            {
                "survey": survey,
                "filter_name": filter_name,
                "table_path": table_dir / f"{filter_name}.dat",
                "delta_wave": delta_wave,
                "trim_quantile": trim_quantile,
            }
        )
    return PassbandGroup(passbands)


def test_passband_group_access(tmp_path):
    """Test that we can create a passband group and access the individual passbands."""
    table_vals = np.array([[100, 0.5], [200, 0.75], [300, 0.25]])
    pb_list = [
        Passband(table_vals, "survey1", "a", trim_quantile=None),
        Passband(table_vals, "survey1", "b", trim_quantile=None),
        Passband(table_vals, "survey1", "c", trim_quantile=None),
        Passband(table_vals, "survey2", "c", trim_quantile=None),
        Passband(table_vals, "survey2", "d", trim_quantile=None),
    ]

    pb_group = PassbandGroup(given_passbands=pb_list)
    assert len(pb_group) == 5
    assert "survey1_a" in pb_group
    assert "survey1_b" in pb_group
    assert "survey1_c" in pb_group
    assert "survey2_c" in pb_group
    assert "survey2_d" in pb_group

    # We can access the passbands by their full name.
    assert pb_group["survey1_a"].filter_name == "a"
    assert pb_group["survey1_b"].filter_name == "b"
    assert pb_group["survey1_c"].filter_name == "c"
    assert pb_group["survey2_c"].filter_name == "c"
    assert pb_group["survey2_d"].filter_name == "d"

    # Test that we can access a list of filters.
    assert pb_group.filters == ["a", "b", "c", "d"]

    # If the group only has a single passband for a given filter,
    # We can use the filter name to access it.
    assert pb_group["a"].full_name == "survey1_a"
    assert pb_group["b"].full_name == "survey1_b"
    assert pb_group["d"].full_name == "survey2_d"

    # But we get an error if we try to look up by a filter name that occurs twice.
    with pytest.raises(KeyError):
        _ = pb_group["c"]

    # The contains functionality will work with all filter names.
    assert "a" in pb_group
    assert "b" in pb_group
    assert "c" in pb_group
    assert "d" in pb_group

    # Check that we can iterate through the passbands.
    for idx, pb in enumerate(pb_group):
        assert pb == pb_list[idx]

    # Check that we can filter a list of filter_names by whether they occur in the passband group.
    filters = ["a", "b", "e", "f", "c", "1", "2", "d", "a", "a"]
    expected = [True, True, False, False, True, False, False, True, True, True]
    assert np.array_equal(pb_group.mask_by_filter(filters), expected)


def test_passband_group_create(tmp_path):
    """Test that we can create passband groups with different inputs."""
    table_vals = np.array([[100, 0.5], [200, 0.75], [300, 0.25]])
    pb1 = Passband(table_vals, "survey1", "a", trim_quantile=None)

    # We create it directly from a passband.
    pb_group = PassbandGroup(given_passbands=pb1)
    assert len(pb_group) == 1
    assert "survey1_a" in pb_group
    assert "survey1_b" not in pb_group

    # We throw an error if we try to create a PassbandGroup with no passbands.
    with pytest.raises(ValueError):
        _ = PassbandGroup(given_passbands=None)
    with pytest.raises(ValueError):
        _ = PassbandGroup(given_passbands=[])


def test_passband_group_process_transmission_tables(tmp_path):
    """Test that we can create reprocess the passbands in a group."""
    table_vals = np.array([[100, 0.5], [200, 0.75], [300, 0.25]])
    pb1 = Passband(table_vals, "survey1", "a", delta_wave=5.0, trim_quantile=None)
    pb2 = Passband(table_vals, "survey1", "b", delta_wave=5.0, trim_quantile=None)
    pb_group = PassbandGroup(given_passbands=[pb1, pb2])

    # The wavelengths are sampled every 5 AA.
    assert np.allclose(pb_group["survey1_a"].waves, np.arange(100, 301, 5))
    assert np.allclose(pb_group["survey1_b"].waves, np.arange(100, 301, 5))

    # Reprocess with a different delta_wave. Check they are now sampled every 10 AA.
    pb_group.process_transmission_tables(delta_wave=10.0, trim_quantile=None)
    assert np.allclose(pb_group["survey1_a"].waves, np.arange(100, 301, 10))
    assert np.allclose(pb_group["survey1_b"].waves, np.arange(100, 301, 10))


def test_passband_group_init(tmp_path, passbands_dir):
    """Test the initialization of the Passband class, and implicitly, _load_preset."""
    # Test that we cannot create an empty PassbandGroup object
    with pytest.raises(TypeError):
        _ = PassbandGroup()

    # Test that the PassbandGroup class raises an error for an unknown preset
    try:
        _ = PassbandGroup.from_preset(preset="Unknown")
    except ValueError as e:
        assert str(e) == "Unknown passband preset: unknown"
    else:
        raise AssertionError("PassbandGroup should raise an error for an unknown preset")

    # Test that the PassbandGroup class can be initialized with a preset
    lsst_passband_group = create_lsst_passband_group(passbands_dir)
    assert len(lsst_passband_group.passbands) == 6
    assert len(lsst_passband_group) == 6
    assert "LSST_u" in lsst_passband_group
    assert "LSST_g" in lsst_passband_group
    assert "LSST_r" in lsst_passband_group
    assert "LSST_i" in lsst_passband_group
    assert "LSST_z" in lsst_passband_group
    assert "LSST_y" in lsst_passband_group
    assert "LSST_purple" not in lsst_passband_group

    # Test that we fail if we use the wrong units for the LSST preset
    with pytest.raises(ValueError):
        _ = PassbandGroup.from_preset(
            preset="LSST",
            table_dir=passbands_dir,
            delta_wave=5.0,
            trim_quantile=None,
            units="A",
        )

    # We can access passbands using the [] notation.
    assert lsst_passband_group["LSST_u"].filter_name == "u"
    with pytest.raises(KeyError):
        _ = lsst_passband_group["LSST_purple"]

    # Test that the PassbandGroup class can be initialized with a dict of passband parameters
    lsst_gri_passband_parameters = [
        {"survey": "LSST", "filter_name": "g", "table_path": f"{passbands_dir}/LSST/g.dat"},
        {"survey": "LSST", "filter_name": "r", "table_path": f"{passbands_dir}/LSST/r.dat"},
        {"survey": "LSST", "filter_name": "i", "table_path": f"{passbands_dir}/LSST/i.dat"},
    ]
    lsst_gri_passband_group = PassbandGroup(given_passbands=lsst_gri_passband_parameters)
    assert len(lsst_gri_passband_group) == 3
    assert "LSST_g" in lsst_gri_passband_group
    assert "LSST_r" in lsst_gri_passband_group
    assert "LSST_i" in lsst_gri_passband_group

    # Test our toy passband group, which makes a PassbandGroup using a custom passband parameters dictionary
    toy_passband_group = create_toy_passband_group(tmp_path)
    assert len(toy_passband_group) == 3
    assert "TOY_a" in toy_passband_group
    assert "TOY_b" in toy_passband_group
    assert "TOY_c" in toy_passband_group
    np.testing.assert_allclose(
        toy_passband_group.waves,
        np.unique(np.concatenate([np.arange(100, 301, 5), np.arange(250, 351, 5), np.arange(400, 601, 5)])),
    )

    # Test that we can retrieve bounds for this passband group.
    min_w, max_w = toy_passband_group.wave_bounds()
    assert min_w == 100.0
    assert max_w == 600.0

    # Test that the PassbandGroup class raises an error for an unknown preset
    try:
        _ = PassbandGroup.from_preset(preset="Unknown")
    except ValueError as e:
        assert str(e) == "Unknown passband preset: unknown"
    else:
        raise AssertionError("PassbandGroup should raise an error for an unknown preset")


def test_passband_group_ztf_preset(passbands_dir):
    """Test that we can load a PassbandGroup using the ZTF preset."""
    pbg = PassbandGroup.from_preset(
        preset="ZTF",
        table_dir=passbands_dir,
        delta_wave=5.0,
        trim_quantile=None,
    )

    # Test that all of the filters are there
    assert len(pbg.passbands) == 3
    assert len(pbg) == 3
    assert "g" in pbg
    assert "r" in pbg
    assert "i" in pbg


def test_passband_group_roman_preset(passbands_dir):
    """Test that we can load a PassbandGroup using the Roman preset."""
    # Load the cached (older and simplified) Roman passband files from the test data directory.
    pbg = PassbandGroup.from_preset(
        preset="Roman",
        table_dir=passbands_dir,
        delta_wave=5.0,
        trim_quantile=None,
    )

    # Test that all of the filters are there
    assert len(pbg.passbands) == 8
    assert len(pbg) == 8
    assert "F062" in pbg
    assert "F087" in pbg
    assert "F106" in pbg
    assert "F129" in pbg
    assert "F158" in pbg
    assert "F184" in pbg
    assert "F146" in pbg
    assert "F213" in pbg

    # We can also load a subset of the filters.
    pbg_subset = PassbandGroup.from_preset(
        preset="Roman",
        table_dir=passbands_dir,
        delta_wave=5.0,
        trim_quantile=None,
        filters=["F062", "F106", "F158"],
    )

    # Test that the subset contains the correct filters.
    assert len(pbg_subset.passbands) == 3
    assert "F062" in pbg_subset
    assert "F106" in pbg_subset
    assert "F158" in pbg_subset
    assert "F087" not in pbg_subset
    assert "F129" not in pbg_subset
    assert "F184" not in pbg_subset
    assert "F146" not in pbg_subset
    assert "F213" not in pbg_subset

    # The load fails if we given the wrong units.
    with pytest.raises(ValueError):
        _ = PassbandGroup.from_preset(
            preset="Roman",
            table_dir=passbands_dir,
            delta_wave=5.0,
            trim_quantile=None,
            units="A",
            filters=["F062", "F106", "F158", "F999"],
        )


def test_passband_group_from_dir(tmp_path):
    """Test that we can load a PassbandGroup from a directory."""
    survey = "FAKE"

    # Create a new survey directory and fill it with filter files.
    table_dir = tmp_path / survey
    table_dir.mkdir()
    transmission_tables = {
        "a": "100 0.5\n150 0.75\n200 0.25\n",
        "b": "200 0.25\n250 0.5\n300 0.75\n",
        "c": "300 0.75\n350 0.25\n400 0.5\n",
        "d": "400 0.75\n450 0.25\n500 0.5\n",
        "e": "500 0.75\n550 0.25\n600 0.5\n",
    }
    for filter_name in transmission_tables:
        with open(table_dir / f"{filter_name}.dat", "w") as f:
            f.write(transmission_tables[filter_name])

    # Load four of the filters from the PassbandGroup
    load_filters = ["a", "b", "c", "e"]
    pb_group = PassbandGroup.from_dir(table_dir, filters=load_filters)
    assert len(pb_group) == len(load_filters)

    for filter in load_filters:
        assert filter in pb_group

        wave_start = int(transmission_tables[filter].split()[0])
        assert wave_start == pb_group[filter].transmission_table[0][0]

    # Check that we throw an error if we try to access an invalid directory.
    with pytest.raises(ValueError):
        _ = PassbandGroup.from_dir("./no_such_directory", filters=["a", "b"])

    # Check that we throw an error if we try to load a filter that does not exist.
    with pytest.raises(ValueError):
        _ = PassbandGroup.from_dir(table_dir, filters=["a", "b", "z"])


def test_passband_group_from_list(tmp_path):
    """Test that we can create a PassbandGroup from a pre-specified list."""
    pb_list = [
        Passband(
            np.array([[100, 0.5], [200, 0.75], [300, 0.25]]),
            "my_survey",
            "a",
            trim_quantile=None,
        ),
        Passband(
            np.array([[250, 0.25], [300, 0.5], [350, 0.75]]),
            "my_survey",
            "b",
            trim_quantile=None,
        ),
        Passband(
            np.array([[400, 0.75], [500, 0.25], [600, 0.5]]),
            "my_survey",
            "c",
            trim_quantile=None,
        ),
    ]
    test_passband_group = PassbandGroup(given_passbands=pb_list)
    assert len(test_passband_group) == 3
    assert "my_survey_a" in test_passband_group
    assert "my_survey_b" in test_passband_group
    assert "my_survey_c" in test_passband_group

    all_waves = np.concatenate([np.arange(100, 301, 5), np.arange(250, 351, 5), np.arange(400, 601, 5)])
    assert np.allclose(test_passband_group.waves, np.unique(all_waves))

    # Test that we can manually add another passband to the group.
    pb4 = Passband(
        np.array([[800, 0.75], [850, 0.25], [900, 0.5]]),
        "my_survey2",
        "a",
        trim_quantile=None,
    )
    test_passband_group.add_passband(pb4)

    assert len(test_passband_group) == 4
    assert "my_survey_a" in test_passband_group
    assert "my_survey_b" in test_passband_group
    assert "my_survey_c" in test_passband_group
    assert "my_survey2_a" in test_passband_group

    all_waves = np.concatenate([all_waves, np.arange(800, 901, 5)])
    assert np.allclose(test_passband_group.waves, np.unique(all_waves))


def test_passband_load_subset_passbands(tmp_path):
    """Test that we can load a subset of filters."""
    pb_list = [
        Passband(
            np.array([[100, 0.5], [200, 0.75], [300, 0.25]]),
            "my_survey",
            "a",
            trim_quantile=None,
        ),
        Passband(
            np.array([[250, 0.25], [300, 0.5], [350, 0.75]]),
            "my_survey",
            "b",
            trim_quantile=None,
        ),
        Passband(
            np.array([[400, 0.75], [500, 0.25], [600, 0.5]]),
            "my_survey",
            "c",
            trim_quantile=None,
        ),
        Passband(
            np.array([[800, 0.75], [850, 0.25], [900, 0.5]]),
            "my_survey",
            "d",
            trim_quantile=None,
        ),
    ]

    # Load one filter by full name and the other by filter name.
    test_passband_group = PassbandGroup(given_passbands=pb_list, filters=["my_survey_a", "c"])
    assert len(test_passband_group) == 2

    assert np.allclose(
        test_passband_group.waves,
        np.unique(np.concatenate([np.arange(100, 301, 5), np.arange(400, 601, 5)])),
    )

    # We run into an error if we try to load a filter that does not exist.
    with pytest.raises(ValueError):
        _ = PassbandGroup(given_passbands=pb_list, filters=["my_survey_a", "z"])


def test_passband_ztf_preset():
    """Test that we can load the ZTF passbands."""

    def _mock_get_bandpass(name):
        """Return a predefined Bandpass object instead of downloading the transmission table."""
        return Bandpass(np.array([6000, 6005, 6010]), np.array([0.5, 0.6, 0.7]))

    # Mock the get_bandpass portion of the download method
    with patch("sncosmo.get_bandpass", side_effect=_mock_get_bandpass):
        group = PassbandGroup.from_preset(preset="ZTF")
        assert len(group) == 3
        assert "ZTF_g" in group
        assert "ZTF_r" in group
        assert "ZTF_i" in group

    # Try the load with a subset of filters.
    with patch("sncosmo.get_bandpass", side_effect=_mock_get_bandpass):
        group = PassbandGroup.from_preset(preset="ZTF", filters=["g", "i"])
        assert len(group) == 2
        assert "ZTF_g" in group
        assert "ZTF_i" in group
        assert "ZTF_r" not in group


def test_passband_group_from_svo():
    """Test that we can load a PassbandGroup from the SVO database."""

    # Mock the Passband.from_svo method to return a predefined Passband object.
    # This function is already tested in test_passband.py.
    def _mock_from_svo(full_filter_name, *args, **kwargs):
        """Return a predefined Passband object instead of downloading the transmission table."""
        survey, filter = full_filter_name.split(".")
        return Passband(
            np.array([[6000, 0.5], [6005, 0.6], [6010, 0.7]]),
            survey=survey,
            filter_name=filter,
        )

    # Mock the get_bandpass portion of the download method
    with patch("lightcurvelynx.astro_utils.passbands.Passband.from_svo", side_effect=_mock_from_svo):
        group = PassbandGroup.from_svo(
            ["SLOAN/SDSS.u", "SLOAN/SDSS.g", "SLOAN/SDSS.r", "SLOAN/SDSS.i", "SLOAN/SDSS.z"]
        )
        assert len(group) == 5
        assert "SLOAN/SDSS_u" in group
        assert "SLOAN/SDSS_g" in group
        assert "SLOAN/SDSS_r" in group
        assert "SLOAN/SDSS_i" in group
        assert "SLOAN/SDSS_z" in group


def test_passband_invalid_preset():
    """Test that we throw an error when given an invalid preset name."""
    with pytest.raises(ValueError):
        _ = PassbandGroup.from_preset(preset="Invalid")


def test_passband_unique_waves():
    """Test that if we create two passbands with very similar wavelengths, they get merged."""
    pb_list = [
        Passband(
            np.array([[100, 0.5], [250, 0.25]]),
            "my_survey",
            "a",
            trim_quantile=None,
        ),
        Passband(
            np.array([[200.000001, 0.25], [300.000001, 0.5]]),
            "my_survey",
            "b",
            trim_quantile=None,
        ),
    ]
    test_passband_group = PassbandGroup(given_passbands=pb_list)
    assert np.allclose(
        test_passband_group.waves,
        np.concatenate([np.arange(100, 201, 5), np.arange(205.000001, 301, 5)]),
    )

    # Larger gaps won't register.
    pb_list = [
        Passband(
            np.array([[100, 0.5], [250, 0.25]]),
            "my_survey",
            "a",
            trim_quantile=None,
        ),
        Passband(
            np.array([[200.5, 0.25], [300.5, 0.5]]),
            "my_survey",
            "b",
            trim_quantile=None,
        ),
    ]
    test_passband_group = PassbandGroup(given_passbands=pb_list)
    assert np.allclose(
        test_passband_group.waves,
        np.unique(np.concatenate([np.arange(100, 251, 5), np.arange(200.5, 301, 5)])),
    )


def test_passband_group_str(passbands_dir, tmp_path):
    """Test the __str__ method of the PassbandGroup class."""
    toy_passband_group = create_toy_passband_group(tmp_path)
    assert str(toy_passband_group) == "PassbandGroup containing 3 passbands: TOY_a, TOY_b, TOY_c"

    lsst_passband_group = create_lsst_passband_group(passbands_dir)
    assert str(lsst_passband_group) == (
        "PassbandGroup containing 6 passbands: LSST_u, LSST_g, LSST_r, LSST_i, LSST_z, LSST_y"
    )


def test_passband_group_fluxes_to_bandfluxes(passbands_dir):
    """Test the fluxes_to_bandfluxes method of the PassbandGroup class."""
    # Test with simple passband group
    lsst_passband_group = create_lsst_passband_group(passbands_dir, delta_wave=20, trim_quantile=None)

    # Create some mock flux data with the same number of columns as wavelengths in our PassbandGroup
    flux = np.linspace(lsst_passband_group.waves * 10, 100, 5)
    bandfluxes = lsst_passband_group.fluxes_to_bandfluxes(flux)

    # Check structure of result is as expected (see Passband tests for value tests)
    assert len(bandfluxes) == 6
    for band_name in lsst_passband_group.passbands:
        assert band_name in bandfluxes
        assert bandfluxes[band_name].shape == (5,)

        # Check that we get the same result with fluxes_to_bandflux.
        bandflux = lsst_passband_group.fluxes_to_bandflux(flux, band_name)
        np.testing.assert_allclose(bandflux, bandfluxes[band_name])

    # Test that we fail with a bad flux matrix (not enough wavelengths).
    with pytest.raises(ValueError):
        _ = lsst_passband_group.fluxes_to_bandfluxes(np.array([[1, 2], [3, 4]]))

    # We also fail if we give it a bad filter name.
    with pytest.raises(ValueError):
        _ = lsst_passband_group.fluxes_to_bandflux(flux, "nonexistent_band")


def test_passband_group_wrapped_from_physical_source(passbands_dir, tmp_path):
    """Test evaluate_bandflux, SEDModel's wrapped version of PassbandGroup's fluxes_to_bandfluxes."""
    # Set up physical model
    sed_values = np.array(
        [
            [1.0, 10.0, 1.0],
            [1.0, 20.0, 5.0],
            [1.0, 30.0, 1.0],
            [2.0, 10.0, 5.0],
            [2.0, 20.0, 10.0],
            [2.0, 30.0, 5.0],
            [3.0, 10.0, 1.0],
            [3.0, 20.0, 5.0],
            [3.0, 30.0, 3.0],
        ]
    )
    model = SEDTemplateModel(sed_values, sed_data_t0=0.0, t0=0.0, interpolation_type="linear")
    state = model.sample_parameters()

    test_times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    # Test wrapper with a PassbandGroup as input (see Passband tests for single-band tests)
    # Using LSST passband group:
    lsst_passband_group = create_lsst_passband_group(passbands_dir, delta_wave=20, trim_quantile=None)
    n_lsst_bands = len(lsst_passband_group.passbands)
    n_times = len(test_times)

    fluxes_source_model = model.evaluate_bandfluxes(
        lsst_passband_group,
        times=np.repeat(test_times, n_lsst_bands),
        filters=np.tile(list(lsst_passband_group.passbands.keys()), n_times),
        state=state,
    )
    result_from_source_model = {
        filter_name: fluxes_source_model[i::n_lsst_bands]
        for i, filter_name in enumerate(lsst_passband_group.passbands)
    }
    evaluated_fluxes = model.evaluate_sed(test_times, lsst_passband_group.waves, state)
    result_from_passband_group = lsst_passband_group.fluxes_to_bandfluxes(evaluated_fluxes)

    # Check the two dicts are the same
    assert result_from_source_model.keys() == result_from_passband_group.keys()
    for key in result_from_source_model:
        np.testing.assert_allclose(result_from_source_model[key], result_from_passband_group[key])

    # Using toy passband group:
    toy_passband_group = create_toy_passband_group(tmp_path, delta_wave=20, trim_quantile=None)
    n_toy_bands = len(toy_passband_group.passbands)
    fluxes_source_model = model.evaluate_bandfluxes(
        toy_passband_group,
        times=np.repeat(test_times, n_toy_bands),
        filters=np.tile(list(toy_passband_group.passbands.keys()), n_times),
        state=state,
    )
    result_from_source_model = {
        filter_name: fluxes_source_model[i::n_toy_bands]
        for i, filter_name in enumerate(toy_passband_group.passbands)
    }
    evaluated_fluxes = model.evaluate_sed(test_times, toy_passband_group.waves, state)
    result_from_passband_group = toy_passband_group.fluxes_to_bandfluxes(evaluated_fluxes)

    # Check the two dicts are the same
    assert result_from_source_model.keys() == result_from_passband_group.keys()
    for key in result_from_source_model:
        np.testing.assert_allclose(result_from_source_model[key], result_from_passband_group[key])


def test_passband_group_calculate_in_band_wave_indices(passbands_dir, tmp_path):
    """Test the calculate_in_band_wave_indices method of the PassbandGroup class using both toy and LSST
    transmission tables."""
    # Using LSST passband group:
    lsst_passband_group = create_lsst_passband_group(passbands_dir, delta_wave=20, trim_quantile=None)

    # Make sure group.waves contains the union of all passband waves with no duplicates or other values
    np.testing.assert_allclose(
        lsst_passband_group.waves,
        np.unique(np.concatenate([passband.waves for passband in lsst_passband_group.passbands.values()])),
    )

    # Using toy passband group:
    toy_passband_group = create_toy_passband_group(tmp_path, delta_wave=20, trim_quantile=None)

    passband_A = toy_passband_group.passbands["TOY_a"]
    passband_B = toy_passband_group.passbands["TOY_b"]
    passband_C = toy_passband_group.passbands["TOY_c"]

    # Note that passband_A and passband_B have overlapping wavelength ranges
    # Where passband_A covers 100-300 and passband_B covers 250-350 (and passband_C covers 400-600)
    toy_a_inds = toy_passband_group._in_band_wave_indices["TOY_a"]
    np.testing.assert_allclose(toy_a_inds, np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13]))
    np.testing.assert_allclose(passband_A.waves, toy_passband_group.waves[toy_a_inds])

    toy_b_inds = toy_passband_group._in_band_wave_indices["TOY_b"]
    np.testing.assert_allclose(toy_b_inds, np.array([8, 10, 12, 14, 15, 16]))
    np.testing.assert_allclose(passband_B.waves, toy_passband_group.waves[toy_b_inds])

    toy_c_inds = toy_passband_group._in_band_wave_indices["TOY_c"]
    assert toy_c_inds == slice(17, 28)
    np.testing.assert_allclose(passband_C.waves, toy_passband_group.waves[toy_c_inds])


def test_write_and_read_passband_group(passbands_dir, tmp_path):
    """Test that we can write a PassbandGroup to a file and read it back in."""
    file_path = tmp_path / "test_passband_group.csv"

    # We can't read the file if it does not exist.
    with pytest.raises(FileNotFoundError):
        _ = PassbandGroup.from_preprocessed_file(file_path)

    pb_group = create_lsst_passband_group(passbands_dir)
    pb_group.to_file(file_path)
    assert file_path.is_file()

    # We can read the file back in.
    pb_group2 = PassbandGroup.from_preprocessed_file(file_path)
    assert pb_group2 is not None
    for key in pb_group.passbands:
        assert key in pb_group2.passbands
        assert pb_group.passbands[key] == pb_group2.passbands[key]

    # We get an error if we try to overwrite an existing file without permission.
    with pytest.raises(FileExistsError):
        pb_group.to_file(file_path, overwrite=False)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_passband_group_plot(tmp_path):
    """Test the plot method of the PassbandGroup class to make sure it doesn't crash."""
    table_vals = np.array([[100, 0.5], [200, 0.75], [300, 0.25]])
    pb1 = Passband(table_vals, "survey1", "a", trim_quantile=None)
    pb2 = Passband(table_vals, "survey1", "b", trim_quantile=None)
    pb_group = PassbandGroup(given_passbands=[pb1, pb2])
    _ = pb_group.plot()
