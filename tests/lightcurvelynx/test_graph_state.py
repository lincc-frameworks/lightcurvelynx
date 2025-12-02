import matplotlib

matplotlib.use("Agg")  # Suppress the plots

import numpy as np
import pytest
from astropy.table import Table
from lightcurvelynx.graph_state import DependencyGraph, GraphState, transpose_dict_of_list


def test_create_single_sample_graph_state():
    """Test that we can create and access a single sample GraphState."""
    state = GraphState()
    assert len(state) == 0
    assert state.num_samples == 1
    assert state.get_all_params_names() == []
    assert state.sample_offset == 0
    assert state.sample_idx is None

    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("b", "v1", 3.0)
    assert len(state) == 3
    assert state["a"]["v1"] == 1.0
    assert state["a"]["v2"] == 2.0
    assert state["b"]["v1"] == 3.0

    with pytest.raises(KeyError):
        _ = state["a"]["v3"]
    with pytest.raises(KeyError):
        _ = state["c"]["v1"]

    # We can access the entries using the extended key name.
    assert state["a.v1"] == 1.0
    assert state["a.v2"] == 2.0
    assert state["b.v1"] == 3.0
    with pytest.raises(KeyError):
        _ = state["c.v1"]
    with pytest.raises(KeyError):
        _ = state["a.v1.v2"]

    # We can access a list of all the parameter names.
    assert state.get_all_params_names() == ["a.v1", "a.v2", "b.v1"]

    # We can create a human readable string representation of the GraphState.
    debug_str = str(state)
    assert debug_str == "a:\n    v1: 1.0\n    v2: 2.0\nb:\n    v1: 3.0"

    # Check that we can get all the values for a specific node.
    a_vals = state.get_node_state("a")
    assert len(a_vals) == 2
    assert a_vals["v1"] == 1.0
    assert a_vals["v2"] == 2.0

    # If the state only has a single node, we can access that node's variables
    # directly. But we get an error if we try to do this with multiple nodes.
    state2 = GraphState()
    state2.set("a", "v1", 1.0)
    state2.set("a", "v2", 2.0)
    assert state2["v1"] == 1.0
    assert state2["v2"] == 2.0

    state2.set("b", "v3", 3.0)
    with pytest.raises(KeyError):
        _ = state["v1"]

    # Can't access an out-of-bounds sample_num.
    with pytest.raises(ValueError):
        _ = state.get_node_state("a", 2)

    b_vals = state.get_node_state("b")
    assert len(b_vals) == 1
    assert b_vals["v1"] == 3.0

    with pytest.raises(KeyError):
        _ = state.get_node_state("c")

    # We can extract a single sample.
    new_state = state.extract_single_sample(0)
    assert len(new_state) == 3
    assert new_state.num_samples == 1
    assert new_state.sample_offset == 0
    assert new_state.sample_idx == 0
    assert state["a"]["v1"] == 1.0
    assert state["a"]["v2"] == 2.0
    assert state["b"]["v1"] == 3.0

    # Both nodes are in the fixed vars (empty sets).
    assert len(new_state.fixed_vars) == 2
    assert "a" in new_state.fixed_vars
    assert "b" in new_state.fixed_vars

    # We can overwrite settings.
    state.set("a", "v1", 10.0)
    assert len(state) == 3
    assert state["a"]["v1"] == 10.0

    # Test we cannot use a name containing the separator as a substring.
    with pytest.raises(ValueError):
        state.set("a.b", "v1", 10.0)
    with pytest.raises(ValueError):
        state.set("b", "v1.v3", 10.0)

    # We faile to create a GraphState with a negative number of samples.
    with pytest.raises(ValueError):
        _ = GraphState(-1)

    # We fail if we try to extract a sample that is out of bounds.
    with pytest.raises(ValueError):
        _ = state.extract_single_sample(-1)
    with pytest.raises(ValueError):
        _ = state.extract_single_sample(5)


def test_create_graph_state_offset():
    """Test that we can create a sample GraphState with an offset."""
    state = GraphState(sample_offset=5)
    assert len(state) == 0
    assert state.num_samples == 1
    assert state.sample_offset == 5


def test_create_single_graph_state_from_flattened_dict():
    """Test that we can create a single GraphState from a flattened dictionary."""
    input = {
        "a.v1": 1.0,
        "b.v1": 2.0,
        "c.v3": 3.0,
        "a.v2": 4.0,
        "c.v2": 5.0,
    }
    state = GraphState.from_dict(input, num_samples=1)

    assert len(state) == 5
    assert state.num_samples == 1
    assert state["a"]["v1"] == 1.0
    assert state["b"]["v1"] == 2.0
    assert state["c"]["v3"] == 3.0
    assert state["a"]["v2"] == 4.0
    assert state["c"]["v2"] == 5.0

    # We can access a list of all the parameter names.
    assert state.get_all_params_names() == ["a.v1", "a.v2", "b.v1", "c.v3", "c.v2"]


def test_create_multiple_graph_state_from_nested_dict():
    """Test that we can create a single GraphState from a nested dictionary."""
    input = {
        "a": {"v1": 1.0, "v2": 4.0},
        "b": {"v1": 2.0},
        "c": {"v3": 3.0, "v2": 5.0},
    }
    state = GraphState.from_dict(input, num_samples=1)

    assert len(state) == 5
    assert state.num_samples == 1
    assert state["a"]["v1"] == 1.0
    assert state["b"]["v1"] == 2.0
    assert state["c"]["v3"] == 3.0
    assert state["a"]["v2"] == 4.0
    assert state["c"]["v2"] == 5.0

    # We can access a list of all the parameter names.
    assert state.get_all_params_names() == ["a.v1", "a.v2", "b.v1", "c.v3", "c.v2"]

    # Fail is we use nested names (no separator), but do not have dictionaries.
    with pytest.raises(ValueError):
        GraphState.from_dict({"a": 1, "b": 2}, num_samples=1)


def test_graph_state_contains():
    """Test that we can use the 'in' operator in GraphState."""
    state = GraphState()
    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("b", "v1", 3.0)

    assert "a" in state
    assert "b" in state
    assert "c" not in state

    assert "a.v1" in state
    assert "a.v2" in state
    assert "a.v3" not in state
    assert "b.v1" in state
    assert "c.v1" not in state

    with pytest.raises(KeyError):
        assert "b.v1.v2" not in state

    # If the state only has a single node, we can access that node's variables directly.
    state2 = GraphState()
    state2.set("a", "v1", 1.0)
    state2.set("a", "v2", 2.0)
    assert "a" in state2
    assert "v1" in state2
    assert "v2" in state2


def test_create_multi_sample_graph_state():
    """Test that we can create and access a multi-sample GraphState."""
    state = GraphState(5)
    assert len(state) == 0
    assert state.num_samples == 5

    state.set("a", "v1", 1.0)
    state.set("a", "v2", np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    state.set("b", "v1", np.array([-2.0, -2.5, -3.0, -3.5, -4.0]))
    assert len(state) == 3
    assert np.allclose(state["a"]["v1"], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(state["a"]["v2"], [2.0, 2.5, 3.0, 3.5, 4.0])
    assert np.allclose(state["b"]["v1"], [-2.0, -2.5, -3.0, -3.5, -4.0])

    # We can access a list of all the parameter names.
    assert state.get_all_params_names() == ["a.v1", "a.v2", "b.v1"]

    # Check tht we can get all the values for a specific node and sample.
    a_vals = state.get_node_state("a")
    assert len(a_vals) == 2
    assert a_vals["v1"] == 1.0
    assert a_vals["v2"] == 2.0

    a_vals = state.get_node_state("a", 1)
    assert len(a_vals) == 2
    assert a_vals["v1"] == 1.0
    assert a_vals["v2"] == 2.5

    # We can extract a single sample.
    new_state = state.extract_single_sample(3)
    assert len(new_state) == 3
    assert new_state.num_samples == 1
    assert new_state.sample_offset == 0
    assert new_state.sample_idx == 3
    assert new_state["a"]["v1"] == 1.0
    assert new_state["a"]["v2"] == 3.5
    assert new_state["b"]["v1"] == -3.5

    # Error if we send an array of the wrong length.
    with pytest.raises(ValueError):
        state.set("b", "v2", [-2.0, -2.5, -3.0, -3.5, -4.0, 1.0])


def test_create_multi_graph_state_from_flattened_dict():
    """Test that we can create a multi-sample GraphState from a flattened dictionary."""
    input = {
        "a.v1": [1.0, 2.0, 3.0],
        "b.v1": [2.0, 4.0, 6.0],
        "c.v3": [3.0, 6.0, 9.0],
        "a.v2": [4.0, 3.0, 2.0],
        "c.v2": [5.0, 5.0, 5.0],
    }
    state = GraphState.from_dict(input, num_samples=3)

    assert len(state) == 5
    assert state.num_samples == 3
    assert np.array_equal(state["a"]["v1"], [1.0, 2.0, 3.0])
    assert np.array_equal(state["a"]["v2"], [4.0, 3.0, 2.0])
    assert np.array_equal(state["b"]["v1"], [2.0, 4.0, 6.0])
    assert np.array_equal(state["c"]["v2"], [5.0, 5.0, 5.0])
    assert np.array_equal(state["c"]["v3"], [3.0, 6.0, 9.0])

    # We raise an error if we get the number of samples wrong.
    with pytest.raises(ValueError):
        _ = GraphState.from_dict(input, num_samples=4)


def test_create_single_graph_state_from_nested_dict():
    """Test that we can create a single GraphState from a nested dictionary."""
    input = {
        "a": {"v1": [1.0, 2.0, 3.0], "v2": [4.0, 3.0, 2.0]},
        "b": {"v1": [2.0, 4.0, 6.0]},
        "c": {"v3": [3.0, 6.0, 9.0], "v2": [5.0, 5.0, 5.0]},
    }
    state = GraphState.from_dict(input, num_samples=3)

    assert len(state) == 5
    assert state.num_samples == 3
    assert np.array_equal(state["a"]["v1"], [1.0, 2.0, 3.0])
    assert np.array_equal(state["a"]["v2"], [4.0, 3.0, 2.0])
    assert np.array_equal(state["b"]["v1"], [2.0, 4.0, 6.0])
    assert np.array_equal(state["c"]["v2"], [5.0, 5.0, 5.0])
    assert np.array_equal(state["c"]["v3"], [3.0, 6.0, 9.0])

    # We raise an error if we get the number of samples wrong.
    with pytest.raises(ValueError):
        _ = GraphState.from_dict(input, num_samples=2)


def test_graph_state_copy():
    """Test that we can deep copy a GraphState."""
    state = GraphState()
    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("b", "v1", 3.0)

    state2 = state.copy()

    # Test that we copied over the meta-data
    assert state2.num_parameters == 3
    assert state2.sample_offset == 0
    assert state2.sample_idx is None
    assert len(state2.fixed_vars) == 2  # empty set for each node
    assert "a" in state2.fixed_vars
    assert "b" in state2.fixed_vars

    # Test with single values.
    state2.set("a", "v1", 10.0)
    state2.set("a", "v2", 20.0)
    state2.set("b", "v1", 30.0)

    # State 1 is unchanged
    assert state["a"]["v1"] == 1.0
    assert state["a"]["v2"] == 2.0
    assert state["b"]["v1"] == 3.0

    # State 2 has the new values.
    assert state2["a"]["v1"] == 10.0
    assert state2["a"]["v2"] == 20.0
    assert state2["b"]["v1"] == 30.0

    # Test with arrays.
    state = GraphState(3, sample_offset=2)
    state.set("a", "v1", np.array([1.0, 2.0, 3.0]))
    state.set("a", "v2", np.array([2.0, 3.0, 4.0]))
    state.set("b", "v1", np.array([3.0, 4.0, 5.0]))

    # Test that we copied over the meta-data
    state2 = state.copy()
    assert state2.num_parameters == 3
    assert state2.sample_offset == 2
    assert len(state2.fixed_vars) == 2  # empty set for each node

    state2["a.v1"][1] = 10.0
    state2["a.v2"][0] = 20.0
    state2["b.v1"][2] = 30.0

    # State 1 is unchanged
    assert np.array_equal(state["a"]["v1"], [1.0, 2.0, 3.0])
    assert np.array_equal(state["a"]["v2"], [2.0, 3.0, 4.0])
    assert np.array_equal(state["b"]["v1"], [3.0, 4.0, 5.0])

    # State 2 has the new values.
    assert np.array_equal(state2["a"]["v1"], [1.0, 10.0, 3.0])
    assert np.array_equal(state2["a"]["v2"], [20.0, 3.0, 4.0])
    assert np.array_equal(state2["b"]["v1"], [3.0, 4.0, 30.0])


def test_create_single_graph_state_from_list():
    """Test that we can create a single GraphState from a list of states."""
    state1 = GraphState(num_samples=1)
    state1.set("a", "v1", 1.0)
    state1.set("a", "v2", 2.0)
    state1.set("b", "v1", 3.0)

    state2 = GraphState(num_samples=2)
    state2.set("a", "v1", [1.0, 2.0])
    state2.set("a", "v2", [2.0, 3.0])
    state2.set("b", "v1", [3.0, 4.0])

    dict1 = {"a.v1": 10.0, "a.v2": 11.0, "b.v1": 12.0}
    state = GraphState.from_list([state1, dict1, state2])

    assert len(state) == 3
    assert state.num_samples == 4
    assert np.array_equal(state["a"]["v1"], [1.0, 10.0, 1.0, 2.0])
    assert np.array_equal(state["a"]["v2"], [2.0, 11.0, 2.0, 3.0])
    assert np.array_equal(state["b"]["v1"], [3.0, 12.0, 3.0, 4.0])

    # We raise an error if we get a state with different parameters.
    state_missing = GraphState.from_dict({"a.v1": 10.0, "a.v2": 11.0})
    with pytest.raises(ValueError):
        _ = GraphState.from_list([state1, dict1, state2, state_missing])

    state_extra = GraphState.from_dict({"a.v1": 10.0, "a.v2": 11.0, "b.v1": 12.0, "c.v1": 12.0})
    with pytest.raises(ValueError):
        _ = GraphState.from_list([state1, dict1, state2, state_extra])

    # Fail on invalid list item.
    with pytest.raises(TypeError):
        _ = GraphState.from_list([state1, dict1, state2, 1.0])

    # Fail on empty list.
    with pytest.raises(ValueError):
        _ = GraphState.from_list([])


def test_create_multi_sample_graph_state_reference():
    """Test that we can create and access a multi-sample GraphState with variables
    that were created from each other (with and without references)."""
    state = GraphState(5)
    state.set("a", "v1", 1.0)
    state.set("a", "v2", np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    state.set("b", "v1", state["a"]["v2"])
    assert len(state) == 3
    assert np.allclose(state["a"]["v1"], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(state["a"]["v2"], [2.0, 2.5, 3.0, 3.5, 4.0])
    assert np.allclose(state["b"]["v1"], [2.0, 2.5, 3.0, 3.5, 4.0])

    # Check that (b, v1) is just pointing to the data held in (a, v2).
    state["a"]["v2"][2] = 5.0
    assert np.allclose(state["a"]["v2"], [2.0, 2.5, 5.0, 3.5, 4.0])
    assert np.allclose(state["b"]["v1"], [2.0, 2.5, 5.0, 3.5, 4.0])

    # If we set force_copy to True then (b, v1) should point to its own
    # copy of the data and not inherit any changes to (a, v2).
    state2 = GraphState(5)
    state2.set("a", "v1", 1.0)
    state2.set("a", "v2", np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    state2.set("b", "v1", state2["a"]["v2"], force_copy=True)
    state2["a"]["v2"][2] = 5.0
    assert np.allclose(state2["a"]["v2"], [2.0, 2.5, 5.0, 3.5, 4.0])
    assert np.allclose(state2["b"]["v1"], [2.0, 2.5, 3.0, 3.5, 4.0])


def test_graph_state_iterate():
    """Test that we can use an iterator to transform a GraphState with
    multiple samples into a list of GraphStates each with a single sample.
    """
    state = GraphState(10)
    state.set("a", "v1", 1.0)
    state.set("a", "v2", np.arange(10))

    list_version = [x for x in state]
    assert len(list_version) == 10
    for i in range(10):
        sample_state = list_version[i]
        assert sample_state.num_samples == 1
        assert sample_state["a.v1"] == 1.0
        assert sample_state["a.v2"] == i


def test_graph_state_equal():
    """Test that we use == on GraphStates."""
    state1 = GraphState(num_samples=2)
    state1.set("a", "v1", [1.0, 2.0])
    state1.set("a", "v2", [2.0, 3.0])
    state1.set("b", "v1", [3.0, 4.0])

    state2 = GraphState(num_samples=2)
    state2.set("a", "v1", [1.0, 2.0])
    state2.set("a", "v2", [2.0, 3.0])
    state2.set("b", "v1", [3.0, 4.0])

    assert state1 == state2

    state2.set("b", "v1", [3.0, 5.0])
    assert state1 != state2

    state2.set("b", "v1", [3.0, 4.0])
    assert state1 == state2

    state2.set("b", "v2", [3.0, 4.0])
    assert state1 != state2

    state3 = GraphState(num_samples=3)
    state3.set("a", "v1", [1.0, 2.0, 3.0])
    state3.set("a", "v2", [2.0, 3.0, 4.0])
    state3.set("b", "v1", [3.0, 4.0, 5.0])
    assert state3 != state1
    assert state3 != state2

    # Test that equality works with a single sample (not arrays).
    state4 = GraphState(num_samples=1)
    state4.set("a", "v1", 1.0)
    state4.set("a", "v2", 2.0)
    state4.set("b", "v1", 3.0)

    state5 = GraphState(num_samples=1)
    state5.set("a", "v1", 1.0)
    state5.set("a", "v2", 2.0)
    state5.set("b", "v1", 3.0)

    assert state4 == state5

    state5.set("a", "v2", 2.5)
    assert state4 != state5

    # Two states are not equal if they have different numbers of samples.
    state5 = GraphState(num_samples=1)
    state6 = GraphState(num_samples=2)
    assert state5 != state6


def test_graph_state_fixed():
    """Test that we respected the 'fixed' flag for GraphState."""
    state = GraphState()
    assert len(state) == 0
    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("b", "v1", 3.0, fixed=True)
    assert len(state) == 3
    assert state["a"]["v1"] == 1.0
    assert state["a"]["v2"] == 2.0
    assert state["b"]["v1"] == 3.0

    # Try changing each of the states. Only two should actually change.
    state.set("a", "v1", 4.0)
    state.set("a", "v2", 5.0)
    state.set("b", "v1", 6.0)
    assert state["a"]["v1"] == 4.0
    assert state["a"]["v2"] == 5.0
    assert state["b"]["v1"] == 3.0


def test_graph_state_update():
    """Test that we can update a single sample GraphState."""
    state = GraphState()
    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("b", "v1", 3.0)

    state2 = GraphState()
    state2.set("a", "v1", 4.0)
    state2.set("a", "v3", 5.0)
    state2.set("c", "v1", 6.0)
    state2.set("c", "v2", 7.0)

    assert len(state) == 3
    assert len(state2) == 4

    # We set 3 new parameters and overwrite one.
    state.update(state2)
    assert len(state) == 6
    assert state["a"]["v1"] == 4.0
    assert state["a"]["v2"] == 2.0
    assert state["a"]["v3"] == 5.0
    assert state["b"]["v1"] == 3.0
    assert state["c"]["v1"] == 6.0
    assert state["c"]["v2"] == 7.0

    # We set 3 new parameters and overwrite one.
    state3 = {"a": {"v2": 8.0, "v4": 9.0}, "d": {"v1": 10.0}}
    state.update(state3)
    assert len(state) == 8
    assert state["a"]["v1"] == 4.0
    assert state["a"]["v2"] == 8.0
    assert state["a"]["v3"] == 5.0
    assert state["a"]["v4"] == 9.0
    assert state["b"]["v1"] == 3.0
    assert state["c"]["v1"] == 6.0
    assert state["c"]["v2"] == 7.0
    assert state["d"]["v1"] == 10.0

    # Test we cannot update with mismatched number of samples.
    state4 = GraphState(num_samples=2)
    state4.set("e", "v1", 1.0)
    with pytest.raises(ValueError):
        state.update(state4)


def test_graph_state_from_table():
    """Test that we can create a GraphState from an AstroPy Table."""
    input = Table(
        {
            GraphState.extended_param_name("a", "v1"): [1.0, 2.0, 3.0],
            GraphState.extended_param_name("a", "v2"): [4.0, 5.0, 6.0],
            GraphState.extended_param_name("b", "v1"): [7.0, 8.0, 9.0],
        }
    )
    state = GraphState.from_table(input)

    assert len(state) == 3
    assert state.num_samples == 3
    np.testing.assert_allclose(state["a"]["v1"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state["a"]["v2"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(state["b"]["v1"], [7.0, 8.0, 9.0])

    # Everything still works if we have only a single value.
    input2 = Table(
        {
            GraphState.extended_param_name("a", "v1"): [1.0],
            GraphState.extended_param_name("a", "v2"): [4.0],
            GraphState.extended_param_name("b", "v1"): [7.0],
        }
    )
    state2 = GraphState.from_table(input2)
    assert state2["a"]["v1"] == 1.0
    assert state2["a"]["v2"] == 4.0
    assert state2["b"]["v1"] == 7.0

    # We fail if given invalid parameter names.
    input3 = Table(
        {
            GraphState.extended_param_name("a", "v1.x"): [1.0],
            GraphState.extended_param_name("a", "v2"): [4.0],
        }
    )
    with pytest.raises(ValueError):
        _ = GraphState.from_table(input3)


def test_graph_state_to_dict():
    """Test that we can create a dictionary from a GraphState."""
    state = GraphState(num_samples=3)
    state.set("a", "v1", [1.0, 2.0, 3.0])
    state.set("a", "v2", [3.0, 4.0, 5.0])
    state.set("b", "v1", [6.0, 7.0, 8.0])

    result = state.to_dict()
    assert len(result) == 3
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("a", "v1")],
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("a", "v2")],
        [3.0, 4.0, 5.0],
    )
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("b", "v1")],
        [6.0, 7.0, 8.0],
    )

    # Everything still works if we have only a single value.
    state2 = GraphState(num_samples=1)
    state2.set("a", "v1", 1.0)
    state2.set("a", "v2", 3.0)
    state2.set("b", "v1", 6.0)

    result2 = state2.to_dict()
    assert len(result2) == 3
    assert result2[GraphState.extended_param_name("a", "v1")] == 1.0
    assert result2[GraphState.extended_param_name("a", "v2")] == 3.0
    assert result2[GraphState.extended_param_name("b", "v1")] == 6.0


def test_graph_state_to_table():
    """Test that we can create an AstroPy Table from a GraphState."""
    state = GraphState(num_samples=3)
    state.set("a", "v1", [1.0, 2.0, 3.0])
    state.set("a", "v2", [3.0, 4.0, 5.0])
    state.set("b", "v1", [6.0, 7.0, 8.0])

    result = state.to_table()
    assert len(result) == 3
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("a", "v1")].data,
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("a", "v2")].data,
        [3.0, 4.0, 5.0],
    )
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("b", "v1")].data,
        [6.0, 7.0, 8.0],
    )


def test_graph_state_update_multi():
    """Test that we can update a single sample GraphState."""
    state = GraphState(num_samples=3)
    state.set("a", "v1", [1.0, 2.0, 3.0])
    state.set("a", "v2", [3.0, 4.0, 5.0])
    state.set("b", "v1", [6.0, 7.0, 8.0])

    state2 = GraphState(num_samples=3)
    state2.set("a", "v1", [9.0, 10.0, 11.0])
    state2.set("c", "v1", [12.0, 13.0, 14.0])

    assert len(state) == 3
    assert len(state2) == 2

    # We set one new parameter and overwrite one.
    state.update(state2)
    assert len(state) == 4
    assert np.allclose(state["a"]["v1"], [9.0, 10.0, 11.0])
    assert np.allclose(state["a"]["v2"], [3.0, 4.0, 5.0])
    assert np.allclose(state["b"]["v1"], [6.0, 7.0, 8.0])
    assert np.allclose(state["c"]["v1"], [12.0, 13.0, 14.0])

    # If we add a parameter with sample_size = 1, we correctly expand it out.
    state3 = {"a": {"v2": 15.0}, "d": {"v1": 16.0}}
    state.update(state3)
    assert len(state) == 5
    assert np.allclose(state["a"]["v1"], [9.0, 10.0, 11.0])
    assert np.allclose(state["a"]["v2"], [15.0, 15.0, 15.0])
    assert np.allclose(state["b"]["v1"], [6.0, 7.0, 8.0])
    assert np.allclose(state["c"]["v1"], [12.0, 13.0, 14.0])
    assert np.allclose(state["d"]["v1"], [16.0, 16.0, 16.0])

    # Test we cannot update with mismatched number of samples.
    state4 = GraphState(num_samples=2)
    state4.set("e", "v1", 1.0)
    with pytest.raises(ValueError):
        state.update(state4)


def test_graph_to_from_file(tmp_path):
    """Test that we can create an AstroPy Table from a GraphState."""
    state = GraphState(num_samples=3)
    state.set("a", "v1", [1.0, 2.0, 3.0])
    state.set("a", "v2", [3.0, 4.0, 5.0])
    state.set("b", "v1", [6.0, 7.0, 8.0])

    file_path = tmp_path / "state.ecsv"
    assert not file_path.is_file()

    state.save_to_file(file_path)
    assert file_path.is_file()

    state2 = GraphState.from_file(file_path)
    assert state == state2

    # Cannot overwrite with it set to False, but works when set to True.
    with pytest.raises(OSError):
        state.save_to_file(file_path)
    state.save_to_file(file_path, overwrite=True)


def test_graph_state_extract_parameters():
    """Test that we can extract named parameters from a GraphState."""
    state = GraphState()
    state.set("a", "v0", 0.0)
    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("a", "v3", 3.0)
    state.set("b", "v1", 4.0)
    state.set("c", "v2", 5.0)
    state.set("c", "v3", 6.0)
    state.set("d", "v4", 7.0)
    state.set("e", "v3", 8.0)
    state.set("e", "v5", 9.0)
    state.set("e", "v6", 10.0)
    state.set("f", "v6", 11.0)

    # We can extract a mixture of unique parameters based on full and short names.
    results = state.extract_parameters(["a.v1", "c.v2", "v5"])
    assert len(results) == 3
    assert results["a.v1"] == 1.0
    assert results["c.v2"] == 5.0
    assert results["v5"] == 9.0

    # If we extract a parameter that appears in multiple nodes with its short
    # name, we expand the name for each instance.
    results = state.extract_parameters(["v2", "v3", "v4"])
    assert len(results) == 6
    assert results["a.v2"] == 2.0
    assert results["a.v3"] == 3.0
    assert results["c.v2"] == 5.0
    assert results["c.v3"] == 6.0
    assert results["e.v3"] == 8.0
    assert results["v4"] == 7.0  # We do not expand the name.

    # We can also provide a single parameter name as a string.
    results = state.extract_parameters("v2")
    assert len(results) == 2
    assert results["a.v2"] == 2.0
    assert results["c.v2"] == 5.0

    # Test a complicated list.
    results = state.extract_parameters(["v0", "c.v2", "e.v3", "v5", "v6", "a.v3"])
    assert len(results) == 7
    assert results["v0"] == 0.0
    assert results["c.v2"] == 5.0
    assert results["e.v3"] == 8.0
    assert results["v5"] == 9.0
    assert results["e.v6"] == 10.0
    assert results["f.v6"] == 11.0
    assert results["a.v3"] == 3.0

    # We raise a KeyError if we try to lookup a parameter that is not in the GraphState.
    with pytest.raises(KeyError):
        _ = state.extract_parameters(["v2", "v3", "c.v4"])

    with pytest.raises(KeyError):
        _ = state.extract_parameters(["v2", "v100"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_dependency_graph():
    """Test that we can create and access a dependency graph."""
    dep_graph = DependencyGraph()
    dep_graph.add_parameter("1", node_name="a")
    dep_graph.add_parameter("2", node_name="a")
    dep_graph.add_parameter("3", node_name="a")
    dep_graph.add_parameter("1", node_name="b")
    dep_graph.add_parameter("2", node_name="b")
    dep_graph.add_parameter("3", node_name="c")
    assert len(dep_graph) == 6
    assert set(dep_graph.all_nodes) == {"a", "b", "c"}

    # We have all the parameters, but no edges between them.
    all_params = set(["a.1", "a.2", "a.3", "b.1", "b.2", "c.3"])
    assert dep_graph.all_params == all_params
    for param in all_params:
        assert param in dep_graph
        assert len(dep_graph.incoming[param]) == 0
        assert len(dep_graph.outgoing[param]) == 0

    # Re-adding a parameter does nothing.
    dep_graph.add_parameter("1", node_name="a")
    assert dep_graph.all_params == all_params
    assert len(dep_graph) == 6

    # We can add edges between nodes.
    dep_graph.add_edge("a.1", "b.1")
    dep_graph.add_edge("a.1", "c.3")
    for param in all_params:
        if param == "a.1":
            assert dep_graph.outgoing[param] == set(["b.1", "c.3"])
        else:
            assert len(dep_graph.outgoing[param]) == 0

        if param in ["b.1", "c.3"]:
            assert dep_graph.incoming[param] == set(["a.1"])
        else:
            assert len(dep_graph.incoming[param]) == 0

    # If we add a constant it adds a node.
    const_name = dep_graph.add_constant(10)
    assert const_name == "const_0=10"
    assert const_name in dep_graph
    assert len(dep_graph) == 7

    # We can add edges from constants.
    dep_graph.add_edge(const_name, "a.2")
    assert dep_graph.outgoing[const_name] == set(["a.2"])
    assert dep_graph.incoming["a.2"] == set([const_name])

    const_name2 = dep_graph.add_constant(10)
    assert const_name2 == "const_1=10"
    dep_graph.add_edge(const_name2, "b.1")
    assert dep_graph.outgoing[const_name2] == set(["b.1"])
    assert dep_graph.incoming["b.1"] == set(["a.1", const_name2])

    # We can add edges multiple times without changing the lists
    dep_graph.add_edge(const_name, "a.2")
    dep_graph.add_edge(const_name, "a.2")
    assert dep_graph.outgoing[const_name] == set(["a.2"])
    assert dep_graph.incoming["a.2"] == set([const_name])

    # We can't add edges between parameters that do not exist.
    with pytest.raises(KeyError):
        dep_graph.add_edge("a.100", "b.1")

    # Test we can export a networkx graph.
    nx_graph = dep_graph.to_networkx()
    assert nx_graph is not None

    # Test that we can plot a dependency graph.
    dep_graph.draw()


def test_dependency_graph_depths():
    """Test that we can create subgraphs of the dependency graph."""
    # Create a graph as:
    # a.1 --> b.1 -> c.2 -> a.2
    #      /    \
    # c.1 /      \-> d.1
    #
    # e.1 --> d.2 -> a.3
    dep_graph = DependencyGraph()
    dep_graph.add_parameter("1", node_name="a")
    dep_graph.add_parameter("1", node_name="b")
    dep_graph.add_parameter("1", node_name="c")
    dep_graph.add_parameter("1", node_name="d")
    dep_graph.add_parameter("1", node_name="e")
    dep_graph.add_parameter("2", node_name="a")
    dep_graph.add_parameter("2", node_name="c")
    dep_graph.add_parameter("2", node_name="d")
    dep_graph.add_parameter("3", node_name="a")
    dep_graph.add_edge("a.1", "b.1")
    dep_graph.add_edge("c.1", "b.1")
    dep_graph.add_edge("b.1", "c.2")
    dep_graph.add_edge("b.1", "d.1")
    dep_graph.add_edge("e.1", "d.2")
    dep_graph.add_edge("c.2", "a.2")
    dep_graph.add_edge("d.2", "a.3")
    assert len(dep_graph) == 9
    assert set(dep_graph.all_nodes) == {"a", "b", "c", "d", "e"}
    assert set(dep_graph.all_params) == {"a.1", "b.1", "c.1", "c.2", "d.1", "d.2", "e.1", "a.2", "a.3"}

    depths = dep_graph.compute_depths()
    assert depths["a.1"] == 0
    assert depths["c.1"] == 0
    assert depths["b.1"] == 1
    assert depths["c.2"] == 2
    assert depths["d.1"] == 2
    assert depths["a.2"] == 3
    assert depths["e.1"] == 0
    assert depths["d.2"] == 1
    assert depths["a.3"] == 2


def test_dependency_graph_subgraphs():
    """Test that we can create subgraphs of the dependency graph."""
    # Create a graph as:
    # a.1 --> b.1 -> c.2
    #      /    \
    # c.1 /      \-> d.1
    #
    # e.1 --> d.2
    dep_graph = DependencyGraph()
    dep_graph.add_parameter("1", node_name="a")
    dep_graph.add_parameter("1", node_name="b")
    dep_graph.add_parameter("1", node_name="c")
    dep_graph.add_parameter("1", node_name="d")
    dep_graph.add_parameter("1", node_name="e")
    dep_graph.add_parameter("2", node_name="c")
    dep_graph.add_parameter("2", node_name="d")
    dep_graph.add_edge("a.1", "b.1")
    dep_graph.add_edge("c.1", "b.1")
    dep_graph.add_edge("b.1", "c.2")
    dep_graph.add_edge("b.1", "d.1")
    dep_graph.add_edge("e.1", "d.2")
    assert len(dep_graph) == 7
    assert set(dep_graph.all_nodes) == {"a", "b", "c", "d", "e"}
    assert set(dep_graph.all_params) == {"a.1", "b.1", "c.1", "c.2", "d.1", "d.2", "e.1"}

    # We can get the various types of subgraphs for a node, including:
    # 1) All nodes on which this node depends (incoming=True, outgoing=False)
    subgraph1 = dep_graph.build_subgraph("b.1", incoming=True, outgoing=False)
    assert len(subgraph1) == 3
    assert set(subgraph1.all_params) == {"a.1", "c.1", "b.1"}
    assert subgraph1.incoming["b.1"] == set(["a.1", "c.1"])
    assert subgraph1.outgoing["b.1"] == set()

    # 2) All nodes that depend on this node (incoming=False, outgoing=True)
    subgraph2 = dep_graph.build_subgraph("b.1", incoming=False, outgoing=True)
    assert len(subgraph2) == 3
    assert set(subgraph2.all_params) == {"b.1", "c.2", "d.1"}
    assert subgraph2.incoming["b.1"] == set()
    assert subgraph2.outgoing["b.1"] == set(["c.2", "d.1"])

    # 3) All nodes in this connected component (incoming=True, outgoing=True)
    subgraph3 = dep_graph.build_subgraph("b.1", incoming=True, outgoing=True)
    assert len(subgraph3) == 5
    assert set(subgraph3.all_params) == {"a.1", "c.1", "b.1", "c.2", "d.1"}
    assert subgraph3.incoming["b.1"] == set(["a.1", "c.1"])
    assert subgraph3.outgoing["b.1"] == set(["c.2", "d.1"])

    # Test that we can extract all connected components.
    components = dep_graph.build_connected_components()
    assert len(components) == 2
    if len(components[0]) == 5:
        assert set(components[0].all_params) == {"a.1", "b.1", "c.1", "c.2", "d.1"}
        assert set(components[1].all_params) == {"e.1", "d.2"}
    else:
        assert set(components[1].all_params) == {"a.1", "b.1", "c.1", "c.2", "d.1"}
        assert set(components[0].all_params) == {"e.1", "d.2"}


def test_transpose_dict_of_list():
    """Test the transpose_dict_of_list helper function"""
    input_dict = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
    }
    expected = [
        {"a": 1, "b": 4, "c": 7},
        {"a": 2, "b": 5, "c": 8},
        {"a": 3, "b": 6, "c": 9},
    ]
    output_list = transpose_dict_of_list(input_dict, 3)
    assert len(output_list) == 3
    for i in range(3):
        assert expected[i] == output_list[i]

    # We fail if num_elem does not match the list lengths.
    with pytest.raises(ValueError):
        _ = transpose_dict_of_list(input_dict, 0)
    with pytest.raises(ValueError):
        _ = transpose_dict_of_list(input_dict, 4)
