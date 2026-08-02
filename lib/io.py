"""Raw-data loaders shared between the kentucky/ and us/ pipelines.

Each function is a direct extraction of code that used to be copy-pasted,
with small variations, across both projects' notebooks -- behavior is
preserved exactly; parameters cover the variation (footnote-stripping,
index suffixes, dtype casting) that existed between call sites so nothing
downstream changes silently.
"""

import numpy as np
import pandas as pd

from .model_spec import required_alt_suffixes, required_individual_columns


def read_puma_pop(path: str) -> pd.DataFrame:
    puma_pop = pd.read_csv(path).set_index("Geo__geoid_")
    puma_pop.index.name = "PUMA"
    puma_pop.index = puma_pop.index.astype(str).str.zfill(7)
    puma_pop = puma_pop.rename(columns={"SE_A00001_001": "TOT_POP"})
    return puma_pop


def read_estdata(path: str, num_alternatives: int) -> pd.DataFrame:
    """Read the `INDIV_COLS` + `ALT{i}_PUMA` + `ALT{i}_<suffix>` subset of `path`,
    clean dtypes, assign a 0-indexed `person_id`, and build `ALT_CHOICE`.

    Reads only the columns in this module's `INDIV_COLS`/`ALT_VARYING_SUFFIXES` instead of the full
    ~3,800-column file, since most of it is unused census fields -- `INDIV_COLS` covers everything not
    alternative-varying (plus whatever's needed to build a choice indicator, e.g. `CHOSEN`/`STAY`),
    `ALT_VARYING_SUFFIXES` covers the `ALT{i}_<suffix>` destination columns actually used, for `i` in
    `1..num_alternatives`. `ALT{i}_PUMA` is always included (needed to match a mover's true chosen
    destination).

    Dtype cleanup: `category`/`object` columns (which includes `ALT{i}_STATE`, `CHOSEN`, and
    `ALT{i}_PUMA` in the current parquet) are cast to `int`. `CHOSEN`/`ALT{i}_PUMA` end up numeric too,
    which is fine -- they're only ever compared to each other, and equality holds either way.

    `ALT_CHOICE` is 0 for stayers, and for movers the index (`1..num_alternatives`) of whichever
    `ALT{i}_PUMA` matches `CHOSEN` -- same convention used throughout the modeling notebooks.
    """
    needed_cols = (
        list(required_individual_columns())
        + [f"ALT{i}_PUMA" for i in range(1, num_alternatives + 1)]
        + [
            f"ALT{i}_{suf}"
            for i in range(1, num_alternatives + 1)
            for suf in required_alt_suffixes()
        ]
    )
    needed_cols = list(dict.fromkeys(needed_cols))

    df = pd.read_parquet(path, columns=needed_cols)

    cat_cols = df.dtypes[df.dtypes == "category"].keys()
    df[cat_cols] = df[cat_cols].apply(lambda x: x.astype(int))

    obj_cols = list(df.dtypes[df.dtypes == "object"].keys())
    df[obj_cols] = df[obj_cols].apply(lambda x: x.astype(int))

    df["person_id"] = np.arange(len(df))

    # add alternatives
    df["ALT_CHOICE"] = 0
    for i in range(1, num_alternatives + 1):
        var = f"ALT{i}_PUMA"
        df["ALT_CHOICE"] = np.where(df[var] == df["CHOSEN"], i, df["ALT_CHOICE"])
    df["ALT_CHOICE"] = np.where(df["STAY"] == 1, 0, df["ALT_CHOICE"])
    assert (df["ALT_CHOICE"] > 0).sum() == (df["STAY"] == 0).sum(), (
        "every mover must match exactly one ALT{i}_PUMA"
    )

    return df


def load_puma_migpuma(
    path: str,
) -> pd.DataFrame:
    """Load and index the PUMA<->MIGPUMA equivalency csv."""
    puma_migpuma = pd.read_csv(path, dtype="str").set_index("PUMA")
    return puma_migpuma[puma_migpuma.index.notna()]


def load_distance_matrix(
    path: str,
    index_dtype: type | str | None = None,
    columns_dtype: type | str | None = None,
    zfill: int | None = None,
) -> pd.DataFrame:
    """Load a PUMA/MIGPUMA distance (or time) matrix CSV."""
    distances = pd.read_csv(path, index_col=0)
    if zfill:
        distances.index = distances.index.astype(str).str.zfill(zfill)
        distances.columns = distances.columns.map(str).str.zfill(zfill)
    else:
        if index_dtype is not None:
            distances.index = distances.index.astype(index_dtype)
        if columns_dtype is not None:
            distances.columns = distances.columns.astype(columns_dtype)
    return distances
