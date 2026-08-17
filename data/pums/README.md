# pums

This folder contains the main logic to clean the PUMS data. There are two primary notebooks:

1. `unify_pums.ipynb`: this notebook uses the IPUMS PUMS extract and performs the decision unit aggregation described in the paper.
2. `unify_pums_old.ipynb`: this notebook uses the raw census PUMS extract and outputs the old version of the cleaned PUMS data (i.e., one entry for each individual, no decision unit logic or handling)

The `group_analysis.ipynb` notebook contains code to analyze various decision unit aggregations mechanisms and quantify how effective they are. It is also the source of some statistics used
in the decision unit aggregation section of the paper.
