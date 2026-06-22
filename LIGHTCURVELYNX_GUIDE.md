# LightCurveLynx Guide

Canonical reference for working on LightCurveLynx.

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

**Units should be consistent and documented.** New code and docstrings should describe the expected units coming into and out of each function. Calls to each function should provide the values in the correct units.

**Citation information should be available.** References for used code and data should be made available via docstrings and `citation_compass`.


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

# Pre-commit (runs ruff lint/format, workflow/pyproject schema checks, pytest, etc.)
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
| `src/lightcurvelynx/effects/effect_model.py` | The `EffectModel` class |
| `src/lightcurvelynx/obstable/obs_table.py` | The `ObsTable` class |
| `src/lightcurvelynx/models/physical_model.py` | The base classes for the models of physical phenomena, including `BasePhysicalModel`, `SEDModel`, and `BandfluxModel` |
| `src/lightcurvelynx/simulate.py` | The functions to run a simulation of the model |


## Architecture: GraphState

Key file: `src/lightcurvelynx/graph_state.py`

The `GraphState` object stores the information about the parameter values. Information is stored in this object, instead of in the parameterized nodes themselves, to keep the main objects stateless. In addition a `GraphState` object can be saved and used to deterministically re-run a simulation or part of the simulation.

The `GraphState` object stores the information about the parameter values. It is organized as a doubly nested dictionary. The outer layer is keyed by the node's name and has values that correspond to that node's parameter dictionary. The inner layer is keyed by the parameter name and maps to the value(s). If `num_samples > 1` the values are stored in a numpy array. If `num_samples==1` the individual (often scalar) value is stored directly. If a function returns an array of length `num_samples` and `num_samples==1`, the value should be extracted from the array *before* `set` is called so that only the individual value is saved.

Example of accessing a parameter:
```python
param_val = graph_state["node_name"]["param_name"]
```

Note that the values within `GraphState` should only be set by the `sample_parameters` of a `ParameterizedNode` object. Users should not manually modify these values.


## Architecture: ParameterizedNodes

Parameterized nodes (sometimes just called nodes) are Python objects that are a subclass of the ``ParameterizedNode`` class and produce or use model parameters during the simulation. Each parameterized node object registers zero or more parameters using the `add_parameter` function. These parameters represent the sampled information the parameterized node needs to complete the simulation. The parameterized node uses the `_sample_helper` function to compute the value of each parameter based on any relevant inputs (recursively calling `_sample_helper` for the node's dependencies) and writes those parameter values to the current `GraphState` object. The parameterized nodes structure is designed to enable simple and consistent sampling.

### ParameterizedNode

Key file: `src/lightcurvelynx/base_models.py`

The base class for all parameterized nodes. If a node uses the `compute` function to set some parameters, it should be a `FunctionNode` subclass.

**Key methods:** `add_parameter`, `sample_parameters`, `get_local_params`

### FunctionNode

Key file: `src/lightcurvelynx/base_models.py`

The base class for all parameterized nodes that perform some in-node computation via the compute function.

**Key methods:** `compute`, `generate`

## Architecture: Physical Models

A physical model is an astronomical phenomenon that is modeled using a subclass of `BasePhysicalModel` (usually also subclasses of `SEDModel` or `BandfluxModel`) that represents a physical phenomenon that produces flux. All physical model classes are subclasses of `ParameterizedNode`.

### BasePhysicalModel

Key file: `src/lightcurvelynx/models/physical_model.py`

The abstract base class used to represent a physical model of a source of flux.

**Key methods:** `add_effect`, `evaluate_bandfluxes`, `evaluate_spectra`

### SEDModel

Key file: `src/lightcurvelynx/models/physical_model.py`

The base class for a model of a source of flux that is defined at the SED level (flux per unit wavelength in the rest frame).

**Key methods:** `compute_sed`, `add_effect`

### BandfluxModel

Key file: `src/lightcurvelynx/models/physical_model.py`

The base class for a model of a source of flux that is defined by band pass values in the observer frame.

**Key methods:** `compute_bandflux`, `add_effect`

## Architecture: EffectModel

Key file: `src/lightcurvelynx/effects/effect_model.py`

A physical or systematic effect to apply to an observation. Effects are added to physical model nodes via the `add_effect` function.

**Key methods:** `apply`, `apply_bandflux`

## Architecture: ObsTable

Key file: `src/lightcurvelynx/obstable/obs_table.py`

`ObsTable` is the parent class for storing information about the observations in a survey, such as pointing and observation conditions. Subclasses of `ObsTable` are defined for different surveys and include information about that survey (e.g. default noise parameters, default filter characteristics, etc.).

**Key methods:** `from_db`, `from_parquet`, `get_observations`, `compute_saturation`

## Extension & Customization

### Decision Tree: What to Extend

Choose the right extension point:

- **Need to sample new parameters?** → Subclass `ParameterizedNode` (or `FunctionNode` if sampling requires computation)
- **Need to add a new flux-producing phenomenon?** → Subclass `SEDModel` or `BandfluxModel`
  - Use `SEDModel` for models defined at the SED level (flux per unit wavelength in rest frame)
  - Use `BandfluxModel` for models defined in observer-frame bandpass fluxes
- **Need to modify flux after generation (dust, reddening, etc.)?** → Subclass `EffectModel`
- **Need to support a new survey?** → Subclass `ObsTable`

### New SED-level Physical Models

**Required methods:**
- `__init__`: Call `super().__init__()` and register all parameters via `add_parameter`
- `compute_sed`: Return shape (T, N) array of flux densities in nJy

**Checklist:**
- [ ] All sampled parameters registered with `add_parameter` in `__init__` (or in one of the parent classes)
- [ ] `compute_sed` accepts (times, wavelengths, graph_state, **kwargs)
- [ ] Return type is numpy array with shape (len(times), len(wavelengths))
- [ ] Tests cover sampling and evaluation in `tests/lightcurvelynx/models/`
- [ ] Run tests: `python -m pytest tests/lightcurvelynx/models/`

**Example:** SinModel emitting a sine wave
```python
class SinModel(SEDModel):
    """A model that emits a sine wave: flux = brightness * sin(2π * frequency * (time - t0))"""

    def __init__(self, brightness, frequency, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("brightness", brightness, description="The inherent brightness")
        self.add_parameter("frequency", frequency, description="The frequency of the sine wave")

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Return effect-free SED values (T x N matrix).
        
        Parameters: times (T array, days), wavelengths (N array, Angstroms), graph_state (GraphState)
        Returns: flux_density (T x N array, nJy)
        """
        params = self.get_local_params(graph_state)
        phases = 2.0 * np.pi * params["frequency"] * (times - params["t0"])
        single_wave = params["brightness"] * np.sin(phases)
        return np.tile(single_wave[:, np.newaxis], (1, len(wavelengths)))
```

### New Bandflux-level Physical Models

**Required methods:**
- `__init__`: Call `super().__init__()` and register all parameters via `add_parameter`
- `compute_bandflux`: Return shape (T,) array of bandpass flux values

**Checklist:**
- [ ] All sampled parameters registered with `add_parameter` in `__init__` (or in one of the parent classes)
- [ ] `compute_bandflux` accepts (times, bandpass, graph_state, **kwargs)
- [ ] Return type is 1D numpy array with shape (len(times),)
- [ ] Tests cover sampling and evaluation in `tests/lightcurvelynx/models/`

### New Effect Models

**Required methods:**
- `__init__`: Call `super().__init__()` and register all parameters via `add_effect_parameter`
- `apply` and/or `apply_bandflux`: Modify and return input flux density array

**Checklist:**
- [ ] All sampled parameters registered with `add_effect_parameter` in `__init__`
- [ ] `apply` accepts (flux_density, times, wavelengths, graph_state, **kwargs)
- [ ] `apply` has keyword arguments for all the effect's parameters
- [ ] Returns array with same shape as input flux_density
- [ ] Tests verify effect modifies flux correctly in `tests/lightcurvelynx/effects/`

### New Survey Information (ObsTable Subclasses)

**Required methods:**
- `__init__`: Call `super().__init__()` and initialize survey-specific defaults
- Set `_default_colnames` mapping (dict from input names to standard names)

**Checklist:**
- [ ] Subclass `ObsTable` from `src/lightcurvelynx/obstable/obs_table.py`
- [ ] Define survey-specific defaults (pixel_scale, read_noise, etc.)
- [ ] Map input column names to standard names via `_default_colnames`
- [ ] Tests verify data loading and metadata access in `tests/lightcurvelynx/obstable/`

### Common Pitfalls to Avoid

❌ **Don't:** Modify GraphState parameters outside your node's `sample_parameters` method
- → Use `add_parameter` during `__init__` and set values within the node

❌ **Don't:** Forget to register parameters with `add_parameter`
- → Unregistered parameters won't appear in GraphState; sampling will fail

❌ **Don't:** Return arrays with incorrect shapes or types
- → SEDModel.compute_sed must return (T, N) float array; BandfluxModel.compute_bandflux must return (T,) float array

❌ **Don't:** Ignore units in docstrings and code
- → Always document input/output units; use nJy for flux, days for time, Angstroms for wavelength

❌ **Don't:** Modify input arrays in-place (especially in effects)
- → Always return a new array: `return flux_density.copy() + modification`

### Success Criteria for Custom Extensions

Your implementation is correct if:
- ✓ All existing tests still pass: `python -m pytest`
- ✓ Custom tests pass: `python -m pytest tests/lightcurvelynx/<your_module>/`
- ✓ Same GraphState + code → same output (deterministic given seeds)
- ✓ Output arrays have expected shapes and finite values
- ✓ Docstrings document units for all inputs/outputs

### Adding new dependencies

When creating a custom extension requires the addition of a new package dependency, the user should consider the package's license, the package's size, whether it is actively maintained, etc.

- **The package is not open source** → Do not use this package.
- **The package is out of data and unmaintained* → Do not use this package.
- **The package has a compatible open source license** → You can (optionally) add the dependency:
  - Add to core dependencies **only** when the package is core to LightCurveLynx.
  - Add to "dev" and "all" when a few optional modules depend on the package. This will be the most common case.
  - Imports outside the code dependencies imports should only be called when needed. The code should use a `try` block and, upon failure, provide instructions on how to install the package.
  - Tests requiring any package outside the core dependencies should use `pytest.importorskip` to skip tests where the package is not included in either the core or "dev" dependencies.


## Typical LightCurveLynx Workflow

A typical LightCurveLynx workflow involves:

1. Loading one or more `ObsTable` for the surveys that you want to simulate.
2. Loading a `PassbandGroup` for each survey you want to simulate.
3. Defining the sampling of the parameters via the use of one or more `ParameterizedNode` objects. You can add explicit dependencies between the sampled values in one node (`model`) and another node (`myotherobject`) using the dot notation:
```python
model = MyObject(
    param1=myotherobject.its_param,
    ...
)
```
4. Creating a physical model object of the source to simulate. Setting its parameters from the previously created parameterized nodes. Common parameters include ra, dec, redshift, and t0.
5. Creating and adding zero or more `EffectModel`s and attaching them to your model object with the `add_effect` function.
6. Calling `simulate_lightcurves` to create a `nested_pandas.NestedFrame` with the results. Each `ObsTable`/`PassbandGroup` pair should be wrapped in a `SurveyInfo` object.


## Debugging

The first step in debugging is often examining the parameters within a `GraphState` object named `state`. Users can do this by printing the object (`print(state)`). Users can also access a dictionary of all parameters for a single node with `state[<node_name>]`.


## Testing Conventions

- **File naming:** `tests/lightcurvelynx/<SUBDIR>/test_<name>.py`, mirroring the `src/lightcurvelynx/` layout.
- **Fixtures:** defined in `tests/conftest.py`. Use existing fixtures, do not duplicate test data.
- Default test run: `python -m pytest`


## Units

The majority of the LightCurveLynx code uses the same units:

- `Flux`: nJy
- `Time`: days
- `RA`: Degrees
- `Dec`: Degrees
- `Wavelength`: Angstroms

Some functions that use alternate units. These should be clearly documented in the docstrings and in comments.

For efficiency reasons LightCurveLynx uses astropy unit objects only when necessary.
