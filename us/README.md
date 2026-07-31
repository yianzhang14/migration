# US Migration Modeling

<!-- This repository contains all the code used to create the logit models for disaggregate human migration in the US.

The contents of the various folders are detailed below:

1. auxiliary - This folder contains various jupyter notebook scripts to do miscellaneous tasks on the input data--e.g., recoding things or cleaning up things when necessary.
2. data - This folder contains (most) of the inputs for the pipeline/other tasks, excluding the very large files that could not be pushed to git.
3. experiments - This folder contains various experiments used during the modeling process to guide decision.
4. pipeline - This folder contains the main pipeline of the modeling process. First, the ```data_cleanup.ipynb``` script takes in all the input data and converts it into a form that the model creation scripts excpe. After this, the ```modeling.ipynb``` notebook does a few final cleanup steps, creates the logit model, and fits it on the data. This folder also contains intermediate steps of iteration.
    - To add additional fields, you can edit the cleanup notebook and then have it reflect accordingly in the modeling notebook.
5. result_files - This folder contains various output files from modeling detailing coefficients & significance.
6. visualization - This folder contains various notebooks used to generate figures/graphs useful for visualization of results, some of which are found in the paper -->
