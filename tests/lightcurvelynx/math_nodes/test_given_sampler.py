import pickle

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.graph_state import GraphState
from lightcurvelynx.math_nodes.given_sampler import (
    BinarySampler,
    GivenValueList,
    GivenValueSampler,
    GivenValueSelector,
    TableSampler,
)


def _test_func(value1, value2):
    """Return the sum of the two parameters.

    Parameters
    ----------
    value1 : `float`
        The first parameter.
    value2 : `float`
        The second parameter.
    """
    return value1 + value2


def test_binary_sampler():
    """Test that we can retrieve results from a BinarySampler."""
    binary_node = BinarySampler(0.75, node_label="test", seed=55)

    single_state = binary_node.sample_parameters(num_samples=1)
    assert isinstance(single_state["test"]["function_node_result"], bool)

    many_states = binary_node.sample_parameters(num_samples=10_000)
    assert len(many_states["test"]["function_node_result"]) == 10_000
    assert 7000 < np.count_nonzero(many_states["test"]["function_node_result"]) < 8000

    # We fail if we use an invalid probability.
    with pytest.raises(ValueError):
        _ = BinarySampler(1.5)


def test_given_value_list():
    """Test that we can retrieve numbers from a GivenValueList."""
    given_node = GivenValueList([1.0, 1.5, 2.0, 2.5, 3.0, -1.0, 3.5])

    # Check that we generate the correct result and save it in the GraphState.
    state1 = GraphState(num_samples=2)
    results = given_node.compute(state1)
    assert np.array_equal(results, [1.0, 1.5])
    assert np.array_equal(given_node.get_param(state1, "function_node_result"), [1.0, 1.5])

    state2 = GraphState(num_samples=1)
    results = given_node.compute(state2)
    assert results == 2.0
    assert given_node.get_param(state2, "function_node_result") == 2.0

    state3 = GraphState(num_samples=2)
    results = given_node.compute(state3)
    assert np.array_equal(results, [2.5, 3.0])
    assert np.array_equal(given_node.get_param(state3, "function_node_result"), [2.5, 3.0])

    # Check that GivenValueList raises an error when it has run out of samples.
    state4 = GraphState(num_samples=4)
    with pytest.raises(IndexError):
        _ = given_node.compute(state4)

    # Resetting the GivenValueList starts back at the beginning.
    given_node.reset()
    state5 = GraphState(num_samples=6)
    results = given_node.compute(state5)
    assert np.array_equal(results, [1.0, 1.5, 2.0, 2.5, 3.0, -1.0])
    assert np.array_equal(
        given_node.get_param(state5, "function_node_result"),
        [1.0, 1.5, 2.0, 2.5, 3.0, -1.0],
    )

    # We fail if we try to create a GivenValueList with an empty list.
    with pytest.raises(ValueError):
        _ = GivenValueList([])

    # GivenValueList cannot be used in distributed computation.
    with pytest.raises(RuntimeError):
        _ = pickle.dumps(given_node)


def test_given_value_list_offset():
    """Test that we can retrieve numbers from a GivenValueList when using a sample offset."""
    given_node = GivenValueList([1.0, 1.5, 2.0, 2.5, 3.0, -1.0, 3.5])

    # Check that we generate the correct result and save it in the GraphState.
    state1 = GraphState(num_samples=2, sample_offset=3)
    results = given_node.compute(state1)
    assert np.array_equal(results, [2.5, 3.0])
    assert np.array_equal(given_node.get_param(state1, "function_node_result"), [2.5, 3.0])

    state2 = GraphState(num_samples=1, sample_offset=3)
    results = given_node.compute(state2)
    assert results == -1.0
    assert given_node.get_param(state2, "function_node_result") == -1.0

    # Check that GivenValueList raises an error when it has run out of samples.
    state3 = GraphState(num_samples=2, sample_offset=3)
    with pytest.raises(IndexError):
        _ = given_node.compute(state3)


def test_test_given_value_list_compound():
    """Test that we can use the GivenValueList as input into another node."""
    values = [1.0, 1.5, 2.0, 2.5, 3.0, -1.0, 3.5, 4.0, 10.0, -2.0]
    given_node = GivenValueList(values)

    # Create a function node that takes the next value and adds 2.0.
    compound_node = FunctionNode(_test_func, value1=given_node, value2=2.0)
    for val in values:
        state = compound_node.sample_parameters()
        assert compound_node.get_param(state, "function_node_result") == val + 2.0

    # Reset the given node and try generating all the samples.
    given_node.reset()
    state2 = compound_node.sample_parameters(num_samples=8)
    assert np.array_equal(
        compound_node.get_param(state2, "function_node_result"),
        [3.0, 3.5, 4.0, 4.5, 5.0, 1.0, 5.5, 6.0],
    )


def test_given_value_list_of_list():
    """Test that we can retrieve lists from a GivenValueList."""
    given_node = GivenValueList([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], node_label="node")

    # Check that we generate the correct result and save it in the GraphState.
    state = given_node.sample_parameters(num_samples=3)
    assert len(state["node"]["function_node_result"]) == 3
    assert np.array_equal(state["node"]["function_node_result"][0], [0, 1])
    assert np.array_equal(state["node"]["function_node_result"][1], [1, 2])
    assert np.array_equal(state["node"]["function_node_result"][2], [2, 3])

    # When we generate a single sample, we should get a list instead of a list of lists.
    state2 = given_node.sample_parameters(num_samples=1)
    assert len(state2["node"]["function_node_result"]) == 2
    assert np.array_equal(state2["node"]["function_node_result"], [3, 4])


def test_given_value_sampler():
    """Test that we can retrieve numbers from a GivenValueSampler."""
    given_node = GivenValueSampler([1, 3, 5, 7])

    # Check that we have sampled uniformly from the given options.
    state = GraphState(num_samples=5_000)
    results = given_node.compute(state)
    assert len(results) == 5_000
    assert np.all((results == 1) | (results == 3) | (results == 5) | (results == 7))
    assert len(results[results == 1]) > 1000
    assert len(results[results == 3]) > 1000
    assert len(results[results == 5]) > 1000
    assert len(results[results == 7]) > 1000

    # We fail if we try to create a GivenValueSampler with an empty list.
    with pytest.raises(ValueError):
        _ = GivenValueSampler([])

    # We fail if the number of weights doesn't match the number of values.
    with pytest.raises(ValueError):
        _ = GivenValueSampler([1, 3, 5], weights=[0.5, 0.5])


def test_given_value_sampler_int():
    """Test that we can retrieve numbers from a GivenValueSampler representing a range."""
    given_node = GivenValueSampler(5)

    # Check that we have sampled uniformly from the given options.
    state = GraphState(num_samples=5_000)
    results = given_node.compute(state)
    assert len(results) == 5_000
    assert np.all((results >= 0) & (results < 5))
    for i in range(5):
        assert len(results[results == i]) > 500


def test_given_value_selector():
    """Test that we can retrieve numbers from a GivenValueSelector."""
    index_node = GivenValueList([0, 1, 2, 3, 2, 3, 1, 2], node_label="index_node")
    given_node = GivenValueSelector([10, 20, 30, 40], index_node, node_label="given_node")

    # Check that we have saampled from the given options based on index.
    state = given_node.sample_parameters(num_samples=8)
    assert len(state["given_node"]["function_node_result"]) == 8
    assert np.array_equal(
        state["given_node"]["function_node_result"],
        [10, 20, 30, 40, 30, 40, 20, 30],
    )

    # We fail if there are no values to select.
    with pytest.raises(ValueError):
        _ = GivenValueSelector([], index_node)

    # We fail if the index is out of bounds.
    with pytest.raises(IndexError):
        given_node2 = GivenValueSelector([10, 20, 30], GivenValueList([3]))
        _ = given_node2.sample_parameters(num_samples=1)


def test_given_value_sampler_weighted():
    """Test that we can retrieve numbers from a GivenValueSampler
    with a weighted distribution."""
    given_node = GivenValueSampler([1, 3, 5, 7], [0.1, 0.5, 0.3, 0.1])

    # Check that we have sampled uniformly from the given options
    # with approximately the given weights.
    state = GraphState(num_samples=10_000)
    results = given_node.compute(state)
    assert len(results) == 10_000
    assert np.all((results == 1) | (results == 3) | (results == 5) | (results == 7))
    assert len(results[results == 1]) > 500
    assert len(results[results == 3]) > 4000
    assert len(results[results == 5]) > 2000
    assert len(results[results == 7]) > 500


@pytest.mark.parametrize("test_data_type", ["dict", "ap_table", "pd_df"])
def test_table_sampler(test_data_type):
    """Test that we can retrieve numbers from a TableSampler from a
    dictionary, AstroPy Table, and Panda's DataFrame."""
    raw_data_dict = {
        "A": [1, 2, 3, 4, 5, 6, 7, 8],
        "B": [1, 1, 1, 1, 1, 1, 1, 1],
        "C": [3, 4, 5, 6, 7, 8, 9, 10],
    }

    # Convert the data type depending on the parameterized value.
    if test_data_type == "dict":
        data = raw_data_dict
    elif test_data_type == "ap_table":
        data = Table(raw_data_dict)
    elif test_data_type == "pd_df":
        data = pd.DataFrame(raw_data_dict)
    else:
        data = None

    # Create the table sampler from the data.
    table_node = TableSampler(data, in_order=True, node_label="node")
    assert set(table_node.columns) == {"A", "B", "C"}
    assert len(table_node) == 8

    state = table_node.sample_parameters(num_samples=2)
    assert len(state) == 3
    assert np.allclose(state["node"]["A"], [1, 2])
    assert np.allclose(state["node"]["B"], [1, 1])
    assert np.allclose(state["node"]["C"], [3, 4])

    # We can sample a single value. Note that the node is not
    # stateful, so it always returns the first N rows. We produce
    # a warning if we detect that the same offset is being used
    # multiple times.
    with pytest.warns(UserWarning):
        state = table_node.sample_parameters(num_samples=1)
    assert len(state) == 3
    assert state["node"]["A"] == 1
    assert state["node"]["B"] == 1
    assert state["node"]["C"] == 3

    # We can sample later values using a forced offset. No warning
    # should be produced here since the offset is different.
    state = table_node.sample_parameters(num_samples=4, sample_offset=2)
    assert len(state) == 3
    assert np.allclose(state["node"]["A"], [3, 4, 5, 6])
    assert np.allclose(state["node"]["B"], [1, 1, 1, 1])
    assert np.allclose(state["node"]["C"], [5, 6, 7, 8])

    # We go past the end of the data.
    with pytest.raises(IndexError):
        _ = table_node.sample_parameters(num_samples=100, sample_offset=5)


def test_table_sampler_fail():
    """Test failure cases when creating a TableSampler."""
    # Non-table data
    with pytest.raises(TypeError):
        _ = TableSampler("not a valid data type")

    # Empty data
    with pytest.raises(ValueError):
        _ = TableSampler({"a": [], "b": []})


def test_table_sampler_offset():
    """Test that we can retrieve numbers from a TableSampler with an offset."""
    data = {
        "A": [1, 2, 3, 4, 5, 6, 7, 8],
        "B": [1, 1, 1, 1, 1, 1, 1, 1],
        "C": [3, 4, 5, 6, 7, 8, 9, 10],
    }

    # Create the table sampler from the data.
    table_node = TableSampler(data, in_order=True, node_label="node")
    state = table_node.sample_parameters(num_samples=2, sample_offset=4)
    assert len(state) == 3
    assert np.allclose(state["node"]["A"], [5, 6])
    assert np.allclose(state["node"]["B"], [1, 1])
    assert np.allclose(state["node"]["C"], [7, 8])

    # We go past the end of the data.
    with pytest.raises(IndexError):
        _ = table_node.sample_parameters(num_samples=4, sample_offset=14)


def test_table_sampler_randomized():
    """Test that we can retrieve numbers from a TableSampler."""
    raw_data_dict = {
        "A": [1, 3, 5],
        "B": [2, 4, 6],
    }

    # Create the table sampler from the data.
    table_node = TableSampler(raw_data_dict, node_label="node")
    state = table_node.sample_parameters(num_samples=2000)

    # Check that we can sample a single point.
    single_state = table_node.sample_parameters(num_samples=1)
    idx = single_state["node"]["selected_table_index"]
    assert single_state["node"]["A"] == raw_data_dict["A"][idx]
    assert single_state["node"]["B"] == raw_data_dict["B"][idx]

    # We have sampled the a_vals roughly uniformly from the three options,
    # and that the row inds match what was sampled.
    a_vals = state["node"]["A"]
    row_inds = state["node"]["selected_table_index"]
    assert len(a_vals) == 2000
    assert np.all((a_vals == 1) | (a_vals == 3) | (a_vals == 5))
    assert np.array_equal((row_inds == 0), (a_vals == 1))
    assert np.array_equal((row_inds == 1), (a_vals == 3))
    assert np.array_equal((row_inds == 2), (a_vals == 5))
    assert len(a_vals[a_vals == 1]) > 500
    assert len(a_vals[a_vals == 3]) > 500
    assert len(a_vals[a_vals == 5]) > 500

    # We have sampled the b_vals roughly uniformly from the three options.
    b_vals = state["node"]["B"]
    assert len(b_vals) == 2000
    assert np.all((b_vals == 2) | (b_vals == 4) | (b_vals == 6))
    assert len(b_vals[b_vals == 2]) > 500
    assert len(b_vals[b_vals == 4]) > 500
    assert len(b_vals[b_vals == 6]) > 500

    # We always sample consistent ROWS of a and b.
    assert np.all(b_vals - a_vals == 1)
