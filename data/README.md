# data

This folder contains the primary code to do data cleaning. The directory tree is described below:

- `acs/`, `cbp/`, `lodes/`, `pums/` all process the raw census data and spit out cleaned versions.
- `geometry` contains various shape files and equivalency files used throughout the data cleaning process. Critical ones include CBSA shapefiles, PUMA/MIGPUMA shapefiles, PUMA/MIGPUMA equivalency files, and tract to PUMA equivalency files.
- `distances/` contains code to generate the distance/time matrices between MIGPUMAs and PUMAs.

The `create_estdata.ipynb` then reads in all the results of this cleaning and merges them into an estdata parquet, which is what the `modeling*` notebooks in the top-level `us/` folder use. Note: if
changes are made to the data cleaning logic, this notebook must be re-ran to regenerate the estdata.
