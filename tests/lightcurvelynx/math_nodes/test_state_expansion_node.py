import pytest
from lightcurvelynx.base_models import FunctionNode, ParameterizedNode
from lightcurvelynx.math_nodes.given_sampler import GivenValueList
from lightcurvelynx.math_nodes.single_value_node import SingleVariableNode
from lightcurvelynx.math_nodes.state_expansion_node import StateExpansionNode


class _AddModel(ParameterizedNode):
    """A test class for the ParameterizedNode that adds two values.

    Parameters
    ----------
    value1 : `float`
        The first value.
    value2 : `float`
        The second value.
    value_sum : `float`
        The sum of the two values.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, value1, value2, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("value1", value1, **kwargs)
        self.add_parameter("value2", value2, **kwargs)
        self.add_parameter(
            "value_sum",
            FunctionNode(self._test_func, value1=self.value1, value2=self.value2),
            **kwargs,
        )

    def _test_func(self, value1, value2, **kwargs):
        return value1 + value2


def test_state_expansion_node():
    """Test that we can create and query a StateExpansionNode with just the repeats."""
    exp_node = StateExpansionNode(repeats=3, node_label="exp")
    state = exp_node.sample_parameters(num_samples=2)

    # We ask for 2 samples, but force 3 repeats of each.
    assert state.num_samples == 6
    assert state["exp"]["org_inds"].tolist() == [0, 0, 0, 1, 1, 1]
    assert state["exp"]["sub_inds"].tolist() == [0, 1, 2, 0, 1, 2]
    assert state["exp"]["repeats"].tolist() == [3, 3, 3, 3, 3, 3]

    # If we resample, we retrigger the expansion and get new indices.
    new_state = exp_node.sample_parameters(num_samples=3)
    assert new_state.num_samples == 9
    assert new_state["exp"]["org_inds"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert new_state["exp"]["sub_inds"].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert new_state["exp"]["repeats"].tolist() == [3, 3, 3, 3, 3, 3, 3, 3, 3]

    # We can chain the nodes.
    exp_node2 = StateExpansionNode(repeats=2, node_label="exp2")
    add_node = _AddModel(
        value1=exp_node2.org_inds,
        value2=exp_node2.repeats,
        node_label="add",
    )
    state = add_node.sample_parameters(num_samples=2)
    assert state.num_samples == 4
    assert state["add"]["value1"].tolist() == [0, 0, 1, 1]
    assert state["add"]["value2"].tolist() == [2, 2, 2, 2]

    # The value sum should be the sum of the number of repeats and the original
    # indices (not meaningful beyond testing).
    assert state["add"]["value_sum"].tolist() == [2, 2, 3, 3]

    # We can use non-uniform repeats.
    repeats_list = GivenValueList([2, 0, 3])
    exp_node3 = StateExpansionNode(repeats=repeats_list, node_label="exp3")
    single_model = SingleVariableNode(name="value", value=exp_node3.org_inds, node_label="single")
    state = single_model.sample_parameters(num_samples=3)
    assert state.num_samples == 5
    assert state["exp3"]["org_inds"].tolist() == [0, 0, 2, 2, 2]
    assert state["exp3"]["sub_inds"].tolist() == [0, 1, 0, 1, 2]
    assert state["exp3"]["repeats"].tolist() == [2, 2, 3, 3, 3]
    assert state["single"]["value"].tolist() == [0, 0, 2, 2, 2]

    # We fail if we try to create a node without any repetition information.
    with pytest.raises(ValueError):
        _ = StateExpansionNode(node_label="bad_exp")


def test_state_expansion_node_subparameters():
    """Test that we can create and query a StateExpansionNode from subparameters."""
    sub_param_list = [
        {"a": [1, 2], "b": [3, 4]},
        {"a": [5, 6, 7], "b": [7, 8, 9]},
        {"a": [], "b": []},
        {"a": [9, 10], "b": [13, 14], "c": [15, 16]},  # Extra column should be ignored.
    ]
    sub_param_node = GivenValueList(sub_param_list)
    exp_node = StateExpansionNode(
        param_names=["a", "b"],
        param_values=sub_param_node,
        node_label="exp",
    )
    state = exp_node.sample_parameters(num_samples=4)

    # We ask for 4 samples, but after flattening the subparameters we have 2 + 3 + 0 + 2 = 7.
    # We have also flattened and appended the new column data.
    assert state.num_samples == 7
    assert state["exp"]["org_inds"].tolist() == [0, 0, 1, 1, 1, 3, 3]
    assert state["exp"]["sub_inds"].tolist() == [0, 1, 0, 1, 2, 0, 1]
    assert state["exp"]["repeats"].tolist() == [None, None, None, None, None, None, None]
    assert state["exp"]["a"].tolist() == [1, 2, 5, 6, 7, 9, 10]
    assert state["exp"]["b"].tolist() == [3, 4, 7, 8, 9, 13, 14]
    assert "c" not in state["exp"]

    # We fail if we give bad combinations of parameters to the node.
    with pytest.raises(ValueError):
        # repeats and param_values
        _ = StateExpansionNode(param_names=["a", "b"], repeats=2, param_values=sub_param_node)
    with pytest.raises(ValueError):
        # param_values but no names
        _ = StateExpansionNode(param_values=sub_param_node)
    with pytest.raises(ValueError):
        # param_names but no values
        _ = StateExpansionNode(param_names=["a", "b"])
    with pytest.raises(ValueError):
        # empty param_names
        _ = StateExpansionNode(param_names=[], param_values=sub_param_node)
    with pytest.raises(ValueError):
        # param_names is not a list of strings
        _ = StateExpansionNode(param_names=[1, "a"], param_values=sub_param_node)

    # We fail if the subparameters are missing a column.
    bad_subparameters = [
        {"a": [1, 2], "b": [3, 4]},
        {"a": [5, 6, 7], "c": [7, 8, 9]},  # Missing 'b'
    ]
    bad_node = StateExpansionNode(
        param_names=["a", "b"],
        param_values=GivenValueList(bad_subparameters),
    )
    with pytest.raises(ValueError):
        _ = bad_node.sample_parameters(num_samples=2)

    # We fail if the subparameters have different numbers of entries within a dictionary.
    bad_subparameters2 = [
        {"a": [1, 2], "b": [3, 4]},
        {"a": [5, 6, 7], "b": [7, 8]},
    ]
    bad_node2 = StateExpansionNode(
        param_names=["a", "b"],
        param_values=GivenValueList(bad_subparameters2),
    )
    with pytest.raises(ValueError):
        _ = bad_node2.sample_parameters(num_samples=2)


def test_state_expansion_node_subparameters_chaining():
    """Test that we can chain the StateExpansionNode."""
    # The first value is just a list of integers.
    value1 = GivenValueList([i * 10 for i in range(10)])

    # The second value provides TWO values to add to each value1.
    sub_param_list = GivenValueList([{"value2": [i, i + 1]} for i in range(10)])
    value2 = StateExpansionNode(param_names=["value2"], param_values=sub_param_list)

    # We add these two values together in a new node, which should trigger the expansion of
    # the second value and the addition of the new parameters to the graph state.
    add_node = _AddModel(
        value1=value1,
        value2=value2.value2,  # We access the newly added column.
        node_label="add",
    )

    # We sample 4 values, but expand those to 8 because of the StateExpansionNode.
    state = add_node.sample_parameters(num_samples=4)
    assert state.num_samples == 8
    assert state["add"]["value1"].tolist() == [0, 0, 10, 10, 20, 20, 30, 30]
    assert state["add"]["value2"].tolist() == [0, 1, 1, 2, 2, 3, 3, 4]
    assert state["add"]["value_sum"].tolist() == [0, 1, 11, 12, 22, 23, 33, 34]

    # We can do the same with uneven subparameters lengths.
    sub_param_list2 = [
        {"value2": [0, 1]},  # 2 repeats
        {"value2": [10, 11, 12]},  # 3 repeats
        {"value2": []},  # 0 repeats
        {"value2": [30], "value3": [40]},  # 1 repeat with extra (ignored) column
    ]
    value2_uneven = StateExpansionNode(
        param_names=["value2"],
        param_values=GivenValueList(sub_param_list2),
    )
    value1.reset()  # Reset the value1 node to trigger a new expansion with the new subparameters.
    add_node_uneven = _AddModel(
        value1=value1,
        value2=value2_uneven.value2,
        node_label="add_uneven",
    )

    state = add_node_uneven.sample_parameters(num_samples=4)
    assert state.num_samples == 6
    assert state["add_uneven"]["value1"].tolist() == [0, 0, 10, 10, 10, 30]
    assert state["add_uneven"]["value2"].tolist() == [0, 1, 10, 11, 12, 30]
    assert state["add_uneven"]["value_sum"].tolist() == [0, 1, 20, 21, 22, 60]
