import numpy as np
import pytest
from lightcurvelynx.astro_utils.mag_flux import mag2flux
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.models.basic_models import StepModel
from lightcurvelynx.obstable.opsim import (
    _opsim_extinction_coeff,
    _opsim_zeropoint_per_sec_zenith,
)
from lightcurvelynx.utils.maf_api import MAFQueryTable, execute_maf_query


def test_create_maf_query_table():
    """Test that we can create a MAFQueryTable."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_mag": np.full(5, 19.0),
    }
    query_table = MAFQueryTable(values)
    assert len(query_table) == 5

    # We can query which columns the OpSim has by new or old name, using
    # the default OpSim column name mappings. The noise information (e.g., zp)
    # is also available.
    assert np.allclose(query_table["ra"], values["fieldRA"])
    assert np.allclose(query_table["dec"], values["fieldDec"])
    assert np.allclose(query_table["time"], values["observationStartMJD"])
    assert np.allclose(query_table["fieldRA"], values["fieldRA"])
    assert np.allclose(query_table["fieldDec"], values["fieldDec"])
    assert np.allclose(query_table["observationStartMJD"], values["observationStartMJD"])
    assert np.allclose(query_table["zp_mag"], values["zp_mag"])
    assert np.allclose(query_table["zp"], mag2flux(values["zp_mag"]))

    # We inherit the default survey values from OpSim.
    assert query_table.survey_values["dark_current"] == 0.2
    assert query_table.survey_values["ext_coeff"] == _opsim_extinction_coeff
    assert query_table.survey_values["pixel_scale"] == 0.2
    assert query_table.survey_values["radius"] == 1.75
    assert query_table.survey_values["read_noise"] == 8.8
    assert query_table.survey_values["zp_per_sec"] == _opsim_zeropoint_per_sec_zenith
    assert query_table.survey_values["survey_name"] == "LSST"

    # No spatial data is computed.
    assert query_table._spatial_data is None

    # Regardless of what we give for the spatial query, we should get all
    # the observations (this is how a MAFQueryTable is defined).
    all_inds = np.arange(5)
    query1 = query_table.range_search(0.0, 0.0)
    assert np.allclose(query1, all_inds)
    query2 = query_table.range_search(np.array([0.0, 15.0]), np.array([0.0, -10.0]))
    assert all(np.allclose(result, all_inds) for result in query2)


def test_execute_maf_query():
    """Test that we can execute a MAF Query."""
    # Assemble the minimum information for a MAF query. Note that this
    # needs to include the information for our default noise model.
    maf_query_data = {
        "time": np.array([0.0, 1.0, 2.0, 3.0]),
        "ra": np.array([15.0, 15.0, 15.0, 15.0]),
        "dec": np.array([-10.0, -10.0, -10.0, -10.0]),
        "filter": np.array(["r", "g", "r", "r"]),
        "zp": np.full(4, 1.0),
        "seeing": [1.12] * 4,
        "skybrightness": [20.0] * 4,
        "exptime": [29.2] * 4,
        "nexposure": [1] * 4,
    }

    # Create a toy model for the simulation.
    toy_model = StepModel(brightness=1000.0, ra=15.0, dec=-10.0, t0=0.5, t1=2.5)

    # Do the simulation. Check that we got expected results and parameters.
    result, params = execute_maf_query(toy_model, maf_query_data)
    assert len(result) == 4
    assert np.allclose(result["mjd"], maf_query_data["time"])
    assert np.allclose(result["flux_perfect"], [0.0, 1000.0, 1000.0, 0.0])
    assert np.all(result["fluxerr"] > 0.0)
    assert np.array_equal(result["filter"], maf_query_data["filter"])
    assert not np.allclose(result["flux_perfect"], result["flux"])
    assert isinstance(params, dict)
    assert len(params) > 0

    # We can presample the graph and run the simulation with a predefined graph state.
    # We should get the same flux_perfect (model parameters), but see different noise.
    toy_model2 = StepModel(
        brightness=NumpyRandomFunc("uniform", low=10.0, high=2000.0),
        ra=15.0,
        dec=-10.0,
        redshift=1e-8,
        t0=0.5,
        t1=2.5,
    )

    state1 = toy_model2.sample_parameters(num_samples=1)
    result1, _ = execute_maf_query(toy_model2, maf_query_data, graph_state=state1)
    result2, _ = execute_maf_query(toy_model2, maf_query_data, graph_state=state1)
    assert np.allclose(result1["flux_perfect"], result2["flux_perfect"])
    assert not np.allclose(result1["flux"], result2["flux"])


def test_execute_maf_query_empty_table():
    """Test that we fail if we provide an empty MAF query table."""
    # Test that we fail if we provide an empty MAF query table.
    empty_maf_query_data = {
        "time": np.array([]),
        "ra": np.array([]),
        "dec": np.array([]),
        "filter": np.array([]),
        "zp": np.array([]),
        "seeing": np.array([]),
        "skybrightness": np.array([]),
        "exptime": np.array([]),
        "nexposure": np.array([]),
    }
    toy_model = StepModel(brightness=1000.0, ra=15.0, dec=-10.0, t0=0.5, t1=2.5)

    with pytest.raises(ValueError, match="The MAF query table is empty."):
        execute_maf_query(toy_model, empty_maf_query_data)
