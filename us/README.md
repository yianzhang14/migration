# us

This directory contains the core notebooks used to generate the results of the paper. The notebooks are listed below:

- The `modeling*` notebooks (`larch, biogeme, torch_choice`) all fit the logit model, but they all use different libraries. Currently, the `larch` notebook is preferred as the
library is relatively easy to work with and is efficient. The `larch` notebook outputs its results into the results/ folder. The results used in the paper are described in `us_mnl_12.html`.
- The `grav_model.ipynb` is used to train the reference gravity model used to benchmark the logit model.
- The `simulation.ipynb` notebook is used to run inference. It takes in data files and estimated coefficients and computes utilities, translating them into probabilities.
- `exploration.ipynb` and `other_paper_figs.ipynb` contain code to generate some of the exploratory statistics and figures seen in the paper.
