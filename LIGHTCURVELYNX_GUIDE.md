# LightCurveLynx Guide

Canonical reference for AI coding assistants working on LightCurveLynx. Tool-specific files (`CLAUDE.md`, `.github/copilot-instructions.md`) contain only tool-specific overrides and reference this file for shared guidance. **Edit this file** for changes that should apply to all AI assistants; edit tool-specific files only for tool-specific behavior.

## What Is LightCurveLynx

LightCurveLynx is an efficient, flexible, and extensible framework for forward simulation of variable and transient astronomical phenomena. Simulation consists of two phases. First, the model's parameters are sampled and stored in a `GraphState` object. Second, the model uses that object to generate fluxes.

## Design Goals and North Stars
**CRITICAL: Always keep these design principles in mind when making changes to LightCurveLynx.**

**Correctness is the Top Priority**
The accuracy of the simulation framework should be prioritized above aspects such as efficiency.

**Code should be modular and use a few consistent APIs.** Models, effects, and survey information are all implemented as classes with a consistent API to allow the user to plug in different components for their simulation.

**Configuration looks like code.** Instead of custom configuration files, the setup should be represented in pythonic code. This is why the `ParameterizedNode` objects set their attributes to be `_AttributeIndicatorNode`, which allows the user to use the dot notation to assign parameter dependencies.

**Results should be reproducible given a GraphState.** The sampling stage of the simulation fills out the values in a GraphState object which are then used to generate the simulated flux and noise values. The GraphState should include everything necessary to deterministically reproduce the simulation. If the simulation requires a random number (such as simulating an AGN), the GraphState for that node should save a random seed to use when creating the random number generator for that node.

**Make Easy Things Easy, Hard Things Possible.** Common use cases should require minimal configuration, but users should also be able to construct complex simulations.


## Coding advice
When changing code, ensure that the current assumptions of the change appear to have always been true. Leave code better than you find it over keeping old assumptions around.

## Development Setup

- **Python ≥ 3.11** (see `pyproject.toml` `requires-python`)
- Create a virtual environment: `python3 -m venv ~/envs/lcl && source ~/envs/lcl/bin/activate`
- Clone and install: `git clone https://github.com/lincc-frameworks/lightcurvelynx.git && cd lightcurvelynx`
- Install via pip: `pip install -e .'[dev]'`

## Common Commands

```bash
# All tests
python -m pytest

# Parallel tests
python -m pytest -n auto

# Lint and format (let the linter fix style — do not hand-tune)
ruff check src/ tests/
ruff format src/ tests/

# Pre-commit (runs ruff, mypy stubs, trailing whitespace, etc.)
pre-commit run --all-files

# Build docs
sphinx-build -M html ./docs ./_readthedocs
```

## Repository Structure

```
src/lightcurvelynx/               Main package
src/lightcurvelynx/astro_utils/   Functions and classes for astronomy specific functions
src/lightcurvelynx/effects/       EffectModel definitions
src/lightcurvelynx/math_nodes/    Utility nodes for generating data during sampling
src/lightcurvelynx/models/        Models of astronomical phenomena
src/lightcurvelynx/noise_models/  Noise model definitions
src/lightcurvelynx/obstable/      Survey specific classes and functionality
src/lightcurvelynx/utils/         General helper utilities
tests/lightcurvelynx/             Test suite
docs/                             Sphinx documentation sources
benchmarks/                       ASV performance benchmarks
```

Key files:

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, ruff/pytest config |
| `src/lightcurvelynx/graph_state.py` | The `GraphState` class |
| `src/lightcurvelynx/base_models.py` | The `ParameterizedNode`, `FunctionNode` and related helper classes |
| `src/lightcurvelynx/models/physical_model.py` | The base classes for the models of physical phenomena, including `BasePhysicalModel`, `SEDModel`, and `BandfluxModel` |
| `src/lightcurvelynx/simulate.py` | The functions to run a simulation of the model |
| `.setup_dev.sh` | Development environment bootstrap |


## Architecture: ParameterizedNodes

Parameterized nodes (sometimes just called nodes) are Python objects that are a subclass of the ``ParameterizedNode`` class and produce or use model parameters during the simulation. Each parameterized node object registers zero or more parameters using the `add_parameter` function. These parameters represent the sampled information the parameterized node needs to complete the simulation. The parameterized node uses the `_sample_helper` function to compute the value of each parameter based on any relevant inputs (recursively calling `_sample_helper` for the node's dependencies) and writes those parameter values to the current `GraphState` object. The parameterized nodes structure is designed to enable simple and consistent sampling.

### ParameterizedNode

Key file: `src/lightcurvelynx/base_models.py`

The base class for all parameterized nodes.

**Key method groups:**
- **Parameter queries:** `list_params`, `describe_params`, `get_param`, `get_local_params`, `set_parameter`, `add_parameter`
- **Sampling functions:** `compute`, `sample_parameters`

### FunctionNode

Key file: `src/lightcurvelynx/base_models.py`

The base class for all parameterized nodes that perform some in-node computation via the compute function.

**Key method groups:**
- **Compute functions:** `compute`, `generate`


## Architecture: Physical Models

A physical model is an astronomical phenomenon that is modeled using a subclass of `BasePhysicalModel` (usually also subclasses of `SEDModel` or `BandfluxModel`) that represents a physical phenomenon that produces flux. All physical model classes are subclasses of `ParameterizedNode`.

### BasePhysicalModel

Key file: `src/lightcurvelynx/models/physical_model.py`

The abstract base class used to represent a physical model of a source of flux. 

**Key method groups:**
- **Effects:** `add_effect`, `add_parameter_offset`
- **Evaluation:** `evaluate_bandfluxes`, `evaluate_spectra`
- **Validity queries:** `minwave`, `maxwave`, `minphase`, `maxphase`

### SEDModel

Key file: `src/lightcurvelynx/models/physical_model.py`

The base class for a model of a source of flux that is defined at the SED level.

**Key method groups:**
- **Effects:** `add_effect`, `list_effects`
- **Evaluation:** `evaluate_sed`, `compute_sed`

### BandfluxModel

Key file: `src/lightcurvelynx/models/physical_model.py`

The base class for a model of a source of flux that is defined by band pass values in the observer frame.

**Key method groups:**
- **Effects:** `add_effect`, `list_effects`
- **Evaluation:** `compute_bandflux`


## Architecture: EffectModel

Key file: `src/lightcurvelynx/effects/effect_model.py`

A physical or systematic effect to apply to an observation. Effects are added to physical model nodes via the `add_effect` function.

**Key method groups:**
- **Evaluation:** `apply`, `apply_bandflux`


## Architecture: ObsTable

Key file: `src/lightcurvelynx/obstable/obs_table.py`

`ObsTable` is the parent class for storing information about the observations in a survey, such as pointing and observation conditions. Subclasses of `ObsTable` are defined for different surveys and include information about that survey (e.g. default noise parameters, default filter characteristics, etc.).

**Key method groups:**

- **Introspection:** `head`, `__len__`, `time_bounds`
- **Data Access:** `__getitem__`, `__contains__`, `get_value_per_row`, `safe_get_survey_value`, `add_column`
- **Detector Footprints:** `clear_detector_footprint`, `set_detector_footprint`, `uses_footprint`
- **Filtering:** `filter_rows`
- **Read from file:** `from_db`, `from_parquet`
- **Resampling:** `make_resampled_table`
- **Saturation:** `compute_saturation`
- **Spatial search:** `is_observed`, `range_search`, `get_observations`
- **Survey footprint:** `estimate_coverage`, `build_moc`, `plot_footprint`
- **Write to file:** `write_db`, `write_parquet`


## Typical LightCurveLynx Workflow

A typical LightCurveLynx workflow involves:

1. Loading one or more `ObsTable` for the surveys that you want to simulate.
2. Loading a `PassbandGroup` for each survey you want to simulate.
3. Defining the sampling of the parameters via the use of one or more `ParameterizedNode` objects.
4. Creating a physical model object of the source to simulate. Setting its parameters from the previously created parameterized nodes. Common parameters include ra, dec, redshift, and t0.
5. Creating and adding zero or more `EffectModel`s and attaching them to your model object with the `add_effect` function.
6. Calling `simulate_lightcurves` to create a `nested_pandas.NestedFrame` with the results. Each `ObsTable`/`PassbandGroup` pair should be wrapped in a `SurveyInfo` object.

## Testing Conventions

- **File naming:** `tests/lightcurvelynx/<SUBDIR>/test_<name>.py`, mirroring the `src/lightcurvelynx/` layout.
- **Fixtures:** defined in `tests/conftest.py`. Use existing fixtures, do not duplicate test data.
- Default test run: `python -m pytest`
