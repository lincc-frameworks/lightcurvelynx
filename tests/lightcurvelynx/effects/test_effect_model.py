import numpy as np
import pytest
from lightcurvelynx.effects.effect_model import EffectModel
from lightcurvelynx.models.basic_models import ConstantSEDModel


def test_effect_model() -> None:
    """The base effect model is able to extract the parameters."""
    model = EffectModel(param1=1.0, param2=2.0)
    assert "param1" in model.parameters
    assert "param2" in model.parameters
    assert model.parameters["param1"] == 1.0
    assert model.parameters["param2"] == 2.0
    assert model.effect_name == "EffectModel"
    assert str(model) == "EffectModel"
    assert repr(model) == "EffectModel(param1,param2)"

    # We can override the effect's name.
    model2 = EffectModel(param3=3.0, effect_name="test_effect")
    assert str(model2) == "test_effect"
    assert repr(model2) == "test_effect(param3)"

    # We cannot call apply on the base EffectModel
    with pytest.raises(NotImplementedError):
        _ = model.apply(np.zeros((5, 3)))


def test_add_effect_model() -> None:
    """The test that we can add an effect model."""
    basic_model = ConstantSEDModel(brightness=1000.0, ra=15.0, dec=-10.0, node_label="test")
    effect_model1 = EffectModel(param1=1.0, param2=2.0)
    basic_model.add_effect(effect_model1)
    state = basic_model.sample_parameters(num_samples=1)
    assert state["test"]["brightness"] == 1000.0
    assert state["test"]["ra"] == 15.0
    assert state["test"]["dec"] == -10.0
    assert state["test"]["param1"] == 1.0
    assert state["test"]["param2"] == 2.0

    # We fail if we try to add a parameter that is already in the model.
    effect_model2 = EffectModel(brightness=25.0)
    with pytest.raises(ValueError):
        basic_model.add_effect(effect_model2)

    # However if we set it the parameter's value to None, then it will reuse the model's value.
    effect_model3 = EffectModel(brightness=None)
    basic_model.add_effect(effect_model3)
    state = basic_model.sample_parameters(num_samples=1)
    assert state["test"]["brightness"] == 1000.0
    assert state["test"]["ra"] == 15.0
    assert state["test"]["dec"] == -10.0
    assert state["test"]["param1"] == 1.0
    assert state["test"]["param2"] == 2.0
