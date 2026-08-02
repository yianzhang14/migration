# ACS data

This data was sourced from (Social Explorer)[https://www.socialexplorer.com/my-home], which automatically compiles the census data to PUMA geographies. When exporting to CSV, all the columns derived by Social Explorer were included, and the "Output column labels in the first row" checkbox was selected. These CSVs were placed into the raw/ directory, where process_raw.ipynb processes them.

In addition to this, the list of tables were exported from Social Explorer's data dictionary pages by coping the table contents--e.g., the (5 year 2018 ACS estimates)[https://www.socialexplorer.com/data/ACS2018_5yr/metadata/?ds=SE]. These are stored into csvs in raw/ as well and parsed using fixed-width columns.