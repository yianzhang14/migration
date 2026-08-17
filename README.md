# migration

This repository holds the code for creating migration logit models using the PUMS data. The directory tree is as follows:

- data/: holds code for cleaning up raw census/external data and making it into a form compatible for fitting the logit model.
- lib/: contains shared code used throughout the codebase. `io.py` and `census.py` provide shared functionality for loading in data from `data/`, and `model_spec.py, modeling_util.py`
are used to define the model programmatically.
- ors/: contains the Docker image used to generate the distance/time matrices between PUMAs and MIGPUMAs using OpenStreetMap.
- us/: contains the core notebooks for fitting the logit model. Also contains various notebooks that produce the statistics and figures used in the paper.

To install the environment, install `pixi` and run `pixi install`. All the packages are listed in the `pyproject.toml`.
