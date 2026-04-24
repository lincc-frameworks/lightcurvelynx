import numpy as np
import pytest
from lightcurvelynx.obstable.dummy_obstables import LookupOnlyObsTable


def test_lookup_only_obs_table():
    """Test that we can create and query a LookupOnlyObsTable."""
    data = {
        "time": [1.0, 2.0, 3.0],
        "filter": ["g", "r", "i"],
        "zp": [26.0, 27.0, 28.0],
    }
    table = LookupOnlyObsTable(data, survey_values={"gain": 2.0})
    assert len(table) == 3

    # Test "contains" and "getitem"
    assert "time" in table
    assert "filter" in table
    assert "zp" in table
    assert "nonexistent" not in table
    np.testing.assert_array_equal(table["time"].values, data["time"])
    np.testing.assert_array_equal(table["filter"].values, data["filter"])
    np.testing.assert_array_equal(table["zp"].values, data["zp"])
    np.testing.assert_array_equal(table["gain"], 2.0)
    with pytest.raises(KeyError, match="missing_key"):
        _ = table["missing_key"]

    # Test safe_get_survey_value
    assert table.safe_get_survey_value("gain") == 2.0
    with pytest.raises(ValueError, match="missing_key"):
        table.safe_get_survey_value("missing_key")

    # Test get_value_per_row
    result = table.get_value_per_row("zp", [0, 2])
    np.testing.assert_array_equal(result, [26.0, 28.0])

    result = table.get_value_per_row("gain", [0, 1, 2])
    np.testing.assert_array_equal(result, [2.0, 2.0, 2.0])

    with pytest.raises(KeyError, match="missing_key"):
        table.get_value_per_row("missing_key", [0, 1])
