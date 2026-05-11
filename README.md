# LightCurveLynx

A Fast and Nimble Package for Time Domain Astronomy

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/lightcurvelynx?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/lightcurvelynx/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/lightcurvelynx.svg)](https://anaconda.org/conda-forge/lightcurvelynx)

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/lightcurvelynx/smoke-test.yml)](https://github.com/lincc-frameworks/lightcurvelynx/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/lincc-frameworks/lightcurvelynx/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/lightcurvelynx)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/lightcurvelynx/asv-main.yml?label=benchmarks)](https://lincc-frameworks.github.io/lightcurvelynx/)
[![Read the Docs](https://img.shields.io/readthedocs/lightcurvelynx)](https://lightcurvelynx.readthedocs.io/)


## Introduction

Realistic light curve simulations are essential to many time-domain problems. 
Simulations are needed to evaluate observing strategy, characterize biases, 
and test pipelines. LightCurveLynx aims to provide a flexible, scalable, and user-friendly
time-domain simulation software with realistic effects and survey strategies.

The software package consists of multiple stages:
1. A flexible framework for consistently sampling model parameters (and hyperparameters),
2. Realistic models of time varying phenomena (such as supernovae and AGNs),
3. Effect models (such as dust extinction), and
4. Survey characteristics (such as cadence, filters, and noise).

For an overview of the package, we recommend starting with [introduction notebook](https://lightcurvelynx.readthedocs.io/en/latest/notebooks/introduction.html).


## Installation

Install from PyPI or conda-forge:

```
pip install lightcurvelynx
```

```
conda install conda-forge::lightcurvelynx
```

Since LightCurveLynx relies on a large number of existing packages, not all of the packages
are installed in the default configuration. You can install most of the optional depenencies
with the "dev" or "all" extras:

```
pip install 'lightcurvelynx[all]'
```

If you need a package that is not installed as part of the default or all configurations,  LightCurveLynx will provide a message with the information on which packages you need to install and how to install them.


## Example Usage

The [tutorial notebooks documentation page](https://lightcurvelynx.readthedocs.io/en/latest/notebooks.html) provides a variety of usage examples and technical deep dives.

If you have questions, check out the [FAQ page](https://lightcurvelynx.readthedocs.io/en/latest/faq.html) or the [getting help page](https://lightcurvelynx.readthedocs.io/en/latest/getting_help.html)


## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment such as `venv`

```
>> python3 -m venv ~/envs/lightcurvelynx
>> source ~/envs/lightcurvelynx/bin/activate
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
```

Notes:
1. The single quotes around `'[dev]'` may not be required for your operating system.
2. `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)

If you are interested in contributing directly to the package, see our [contribution guide](https://lightcurvelynx.readthedocs.io/en/latest/contributing.html).


## Advisories

This project is under active development and may see API changes.

**Users should always carefully validate the science outputs for their use case.**
Please reach out to the team if you find any problems.


## Acknowledgements

This project is supported by Schmidt Sciences.