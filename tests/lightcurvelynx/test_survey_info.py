import numpy as np
import pytest
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.noise_models.base_noise_models import ConstantFluxNoiseModel, PoissonFluxNoiseModel
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable
from lightcurvelynx.survey_info import SurveyInfo


def test_create_survey_info(passbands_dir):
    """Test that we can create a SurveyInfo object with explicitly
    specified and default parameters."""
    # Create fake passbands and noise model to use. Use the data in the test
    # data directory for the passbands, so we do not need to download it.
    pb_group = PassbandGroup.from_preset(preset="LSST", table_dir=passbands_dir)
    pb_group2 = PassbandGroup.from_preset(
        preset="LSST",
        filters=["r", "g"],
        table_dir=passbands_dir,
    )

    # Create a FakeObsTable
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    ops_data = FakeObsTable(values)

    # Create the survey info with the default noise model.
    survey_info = SurveyInfo(
        obstable=ops_data,
        passbands=pb_group,
        noise_model=ConstantFluxNoiseModel(1.0),
    )
    assert survey_info.obstable is ops_data
    assert survey_info.passbands is pb_group
    assert isinstance(survey_info.noise_model, ConstantFluxNoiseModel)
    assert survey_info.survey_name == "fake_survey"

    # Create the full SurveyInfo object with all parameters specified.
    survey_info_full = SurveyInfo(
        obstable=ops_data,
        passbands=pb_group,
        noise_model=ConstantFluxNoiseModel(1.0),
        survey_name="Test Survey",
    )
    assert survey_info_full.obstable is ops_data
    assert survey_info_full.passbands is pb_group
    assert isinstance(survey_info_full.noise_model, ConstantFluxNoiseModel)
    assert survey_info_full.survey_name == "test survey"

    # We fail with no passbands or incompatible passbands.
    with pytest.raises(ValueError):
        SurveyInfo(obstable=ops_data, passbands=None)
    with pytest.raises(ValueError):
        SurveyInfo(obstable=ops_data, passbands=pb_group2)

    # We fail with an incompatible noise model.
    with pytest.raises(ValueError):
        SurveyInfo(obstable=ops_data, passbands=pb_group, noise_model=PoissonFluxNoiseModel())

    # We fail if no obstable is given.
    with pytest.raises(ValueError):
        SurveyInfo(obstable=None, passbands=pb_group)


def test_create_survey_info_defaults(passbands_dir):
    """Test that we access the correct defaults when giving a survey name."""
    # Create a FakeObsTable with a survey name in the survey values.
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    ops_data = FakeObsTable(values)

    # We only test a few of the surveys because we don't want to force a download of
    # of the passbands for all of the surveys just for testing.

    # LSST survey (LSSTObsTable and OpSim) loaded from the test data directory.
    survey_info = SurveyInfo(
        obstable=ops_data,
        survey_name="LSST",
        validate=False,
        table_dir=passbands_dir,
    )
    assert isinstance(survey_info.noise_model, PoissonFluxNoiseModel)
    for passband in survey_info.passbands:
        assert passband.survey == "LSST"

    # Spectrograph survey (SpectrographObsTable).
    survey_info = SurveyInfo(obstable=ops_data, survey_name="Spectrograph", validate=False)
    assert survey_info.noise_model is None
    assert survey_info.passbands is None

    # ZTF survey (ZTFObsTable) loaded from sncosmo.
    survey_info = SurveyInfo(obstable=ops_data, survey_name="ZTF", validate=False)
    assert isinstance(survey_info.noise_model, PoissonFluxNoiseModel)
    for passband in survey_info.passbands:
        assert passband.survey == "ZTF"

    # Test a non-existent survey name, which should fail because there are no defaults.
    with pytest.raises(ValueError):
        SurveyInfo(obstable=ops_data, survey_name="NonExistentSurvey", validate=False)


def test_create_survey_unknown(passbands_dir):
    """Test that we can create a SurveyInfo with unknown default types."""
    # Create a FakeObsTable
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    ops_data = FakeObsTable(values)

    # Remove the default survey name.
    del ops_data.survey_values["survey_name"]

    # Create the survey info with the default noise model.
    pb_group = PassbandGroup.from_preset(preset="LSST", table_dir=passbands_dir)
    survey_info = SurveyInfo(
        obstable=ops_data,
        passbands=pb_group,
        noise_model=ConstantFluxNoiseModel(1.0),
    )
    assert survey_info.obstable is ops_data
    assert survey_info.passbands is pb_group
    assert isinstance(survey_info.noise_model, ConstantFluxNoiseModel)
    assert survey_info.survey_name == "unknown"

    # We fail if we don't specify the passbands.
    with pytest.raises(ValueError):
        SurveyInfo(obstable=ops_data, noise_model=ConstantFluxNoiseModel(1.0))

    # We fail if we don't specify the noise model.
    with pytest.raises(ValueError):
        SurveyInfo(obstable=ops_data, passbands=pb_group)


def test_create_survey_none(passbands_dir):
    """Test that when we create a survey with the name 'none', it does not
    require passbands or a noise model."""
    # Create a FakeObsTable
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    ops_data = FakeObsTable(values)

    # Create the survey info with the default noise model.
    survey_info = SurveyInfo(
        obstable=ops_data,
        survey_name="none",
    )
    assert survey_info.obstable is ops_data
    assert survey_info.passbands is None
    assert survey_info.noise_model is None
    assert survey_info.survey_name == "none"
